"""
arena.py — the measurement instrument for RL. Without this you can't tell if a
training run is helping or hurting, so it's built to be rigorous, not quick.

Provides:
  * match(make_A, make_B, ...)   head-to-head, alternating seats, fresh per-game state
  * elo_delta / wilson           turn win-rate into an Elo gap with a 95% CI
  * gate(cand, anchor, ...)      AlphaGo-style promotion test (mirror, threshold)
  * field_eval(make_agent, ...)  our deck vs the varied opponent pool (per-matchup)

Agents are passed as *factories* (zero-arg callables returning a fresh policy
function), because combat/deck-inference carry per-game state that must reset.
"""
from __future__ import annotations
import argparse, glob, json, math, os, random, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch

import features as fx
import combat
import deck_inference as DI
import policy_heuristic as H
import model as M
import stats
from evaluate import CardDB
from train_bc import collate, device
from cg import game


# ---------- net inference ----------
def _infer(net, enc, dev):
    X, _ = collate([{**enc, "target": [0]}])
    X = {k: v.to(dev) for k, v in X.items()}
    with torch.no_grad():
        logits, value = net(X)
    m = X["option_mask"][0]
    p = torch.softmax(logits[0].masked_fill(m < 0.5, -1e9), 0).cpu().numpy()
    return p, float(value[0]), int(m.sum().item())


def load_net(path, dev, num_ids=1268):
    net = M.PointerPolicyValueNet(num_card_ids=num_ids, d=96).to(dev)
    net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    return net


# ---------- agent factories ----------
def make_net_agent(net, dev, db, atk, deck, use_combat=False):
    """Return a factory: each call yields a fresh per-game policy fn(obs)->action."""
    def factory():
        trk = DI.OpponentTracker()
        lib = DI.library_from_pool(deck) if use_combat else DI.ArchetypeLibrary().fit([("our", deck)])
        predictor = (lambda o: DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3)) \
            if use_combat else None

        def policy(obs):
            sel = obs.get("select")
            if sel is None:
                return deck
            O = len(sel.get("option", []))
            if (sel.get("minCount", 1) or 0) > 1 or O == 0:
                return H.select(obs, db=db, attack_db=atk)
            if use_combat:
                trk.update(obs)
            p, _, _ = _infer(net, fx.encode_observation(obs, attack_lookup=atk), dev)
            base = [int(np.argmax(p[:O]))]
            if use_combat and combat.available():
                return combat.refine_selection(obs, deck, db, atk, base,
                                                opp_predictor=predictor, plies=1)
            return base
        return policy
    return factory


def make_heuristic_agent(db, atk, deck, use_combat=False):
    def factory():
        trk = DI.OpponentTracker()
        lib = DI.library_from_pool(deck) if use_combat else DI.ArchetypeLibrary().fit([("our", deck)])
        predictor = (lambda o: DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3)) \
            if use_combat else None

        def policy(obs):
            sel = obs.get("select")
            if sel is None:
                return deck
            if use_combat:
                trk.update(obs)
            base = H.select(obs, db=db, attack_db=atk)
            if use_combat and combat.available():
                return combat.refine_selection(obs, deck, db, atk, base,
                                                opp_predictor=predictor, plies=1)
            return base
        return policy
    return factory


def make_ismcts_agent(net, dev, db, atk, deck, worlds=4, sims=32, c_puct=1.5):
    """Net wrapped in ISMCTS search at play time (the AlphaZero 'search improves the net' agent)."""
    import ismcts
    def factory():
        trk = DI.OpponentTracker()
        lib = DI.library_from_pool(deck)
        predictor = lambda o: DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3)

        def policy(obs):
            sel = obs.get("select")
            if sel is None:
                return deck
            O = len(sel.get("option", []))
            if (sel.get("minCount", 1) or 0) > 1 or O == 0:
                return H.select(obs, db=db, attack_db=atk)
            trk.update(obs)
            if O == 1:
                return [0]
            pol, _, ch = ismcts.search(obs, deck, db, atk, net, dev, predictor,
                                       n_worlds=worlds, n_sims=sims, c_puct=c_puct)
            if ch is not None:
                return [ch]
            p, _, _ = _infer(net, fx.encode_observation(obs, attack_lookup=atk), dev)
            return [int(np.argmax(p[:O]))]
        return policy
    return factory


# ---------- stats ----------
def wilson(wins, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def elo_delta(winrate):
    w = min(max(winrate, 1e-4), 1 - 1e-4)
    return -400.0 * math.log10(1.0 / w - 1.0)


# ---------- play ----------
def play_one(polA, polB, deckA, deckB, seatA, max_steps=4000):
    pol = {seatA: polA, 1 - seatA: polB}
    decks = [None, None]; decks[seatA] = deckA; decks[1 - seatA] = deckB
    obs, _ = game.battle_start(decks[0], decks[1]); s = 0
    while s < max_steps:
        cur = obs.get("current") or {}
        if cur.get("result", -1) >= 0:
            break
        sel = obs.get("select")
        yi = cur.get("yourIndex", 0)
        action = pol[yi](obs) if sel is not None else decks[yi]
        obs = game.battle_select(action); s += 1
    res = (obs.get("current") or {}).get("result", -1)
    game.battle_finish()
    return res  # 0/1 winner, 2 draw, -1 unfinished


def match(make_A, make_B, deckA, deckB, games, progress_every=0, label=""):
    wa = wb = draw = 0
    for i in range(games):
        seatA = i % 2
        res = play_one(make_A(), make_B(), deckA, deckB, seatA)
        if res == 2 or res < 0:
            draw += 1
        elif res == seatA:
            wa += 1
        else:
            wb += 1
        if progress_every and (i + 1) % progress_every == 0:
            print(f"      {label}{i+1}/{games}: {wa}-{wb}-{draw} "
                  f"({wa/max(i+1,1):.0%})", flush=True)
    return wa, wb, draw


# ---------- gate & field eval ----------
def gate(make_cand, make_anchor, our_deck, games, threshold=0.55, log=None, tag="gate"):
    """Mirror match candidate vs anchor; promote if score >= threshold."""
    wa, wb, draw = match(make_cand, make_anchor, our_deck, our_deck, games,
                         progress_every=max(games // 5, 1), label=f"{tag} ")
    dec = wa + wb
    score = (wa + 0.5 * draw) / max(games, 1)
    wr = wa / max(dec, 1)
    p, lo, hi = wilson(wa, dec)
    elo = elo_delta(wr)
    passed = score >= threshold and lo > 0.5      # CI must clear coin-flip
    print(f"  {tag}: {wa}-{wb}-{draw}  score {score:.0%}  vs-anchor winrate {wr:.0%} "
          f"[{lo:.0%},{hi:.0%}]  Elo {elo:+.0f}  -> {'PROMOTE' if passed else 'keep best'}")
    if log:
        stats.log(log, event="gate", wins=wa, losses=wb, draws=draw, score=round(score, 3),
                  winrate=round(wr, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                  elo=round(elo, 1), promoted=passed)
    return passed, score, elo


def field_eval(make_agent, our_deck, pool, games, opp_make=None, db=None, atk=None, log=None,
               tag="field eval (our deck vs pool)", verbose=True):
    """Our agent (our_deck) vs each opponent deck in the pool, per-matchup win-rate."""
    overall_w = overall_n = 0
    if verbose:
        print(f"  {tag}:")
    for name, opp_deck in pool:
        mk_opp = opp_make if opp_make else make_heuristic_agent(db, atk, opp_deck)
        wa, wb, draw = match(make_agent, mk_opp, our_deck, opp_deck, games)
        wr = (wa + 0.5 * draw) / max(games, 1)
        overall_w += wa + 0.5 * draw; overall_n += games
        p, lo, hi = wilson(wa, wa + wb)
        if verbose:
            print(f"    vs {name:<20} {wa}-{wb}-{draw}  {wr:.0%} [{lo:.0%},{hi:.0%}]", flush=True)
        if log:
            stats.log(log, event="field_matchup", opponent=name, winrate=round(wr, 3),
                      ci_low=round(lo, 3), ci_high=round(hi, 3))
    overall = overall_w / max(overall_n, 1)
    if verbose:
        print(f"    overall {overall:.0%}", flush=True)
    if log:
        stats.log(log, event="field_summary", winrate=round(overall, 3), n=overall_n)
    return overall


def _load_ctx():
    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in json.load(open("attack_table.json")).items()}
    our = [int(x) for x in open("deck.csv").read().split()][:60]
    pool = []
    for f in sorted(glob.glob("decks/*.csv")):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            pool.append((os.path.basename(f)[:-4], d))
    return db, atk, our, pool


def load_pool(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            out.append((os.path.basename(f)[:-4], d))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", help="net .pt to evaluate")
    ap.add_argument("--anchor", help="baseline net .pt (e.g. the BC model)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--combat", action="store_true", help="wrap nets with combat search")
    ap.add_argument("--ismcts", action="store_true", help="wrap the candidate net with ISMCTS search")
    ap.add_argument("--ismcts-worlds", type=int, default=4)
    ap.add_argument("--ismcts-sims", type=int, default=32)
    ap.add_argument("--c-puct", type=float, default=1.5, help="ISMCTS exploration (higher = more off-prior)")
    ap.add_argument("--field", action="store_true", help="also eval vs the deck pool")
    ap.add_argument("--adversary", action="store_true", help="also eval vs off-meta adversary decks")
    ap.add_argument("--log", default="logs/arena.jsonl")
    a = ap.parse_args()

    dev = device()
    db, atk, our, pool = _load_ctx()
    cand = load_net(a.candidate, dev) if a.candidate else None
    anch = load_net(a.anchor, dev) if a.anchor else None
    mode = (f"ISMCTS {a.ismcts_worlds}w x {a.ismcts_sims}s (c_puct {a.c_puct})" if (cand is not None and a.ismcts)
            else ("net+combat" if a.combat else "raw net"))
    print(f"arena: candidate={mode} | {a.games} games | device={dev}", flush=True)

    if cand is not None and a.ismcts:
        mk_cand = make_ismcts_agent(cand, dev, db, atk, our, a.ismcts_worlds, a.ismcts_sims, a.c_puct)
    elif cand is not None:
        mk_cand = make_net_agent(cand, dev, db, atk, our, a.combat)
    else:
        mk_cand = make_heuristic_agent(db, atk, our, a.combat)
    if anch is not None:
        mk_anch = make_net_agent(anch, dev, db, atk, our, a.combat)
        gate(mk_cand, mk_anch, our, a.games, log=a.log, tag="candidate vs anchor")
    if a.field or anch is None:
        field_eval(mk_cand, our, pool, max(a.games // max(len(pool), 1), 10),
                   db=db, atk=atk, log=a.log)
    if a.adversary:
        adv = load_pool("decks/adversary/*.csv")
        if adv:
            field_eval(mk_cand, our, adv, max(a.games // max(len(adv), 1), 10),
                       db=db, atk=atk, log=a.log,
                       tag="ADVERSARY eval (our deck vs off-meta exploiters)")
