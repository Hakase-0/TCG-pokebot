"""
selfplay_rl.py — Expert-Iteration self-play RL (AlphaZero-style, imperfect-info).

This is the rung past behavioral cloning: the policy/value net plays itself in
the cabt engine and learns from OUTCOMES, so it can exceed the data it was
cloned from instead of merely imitating it.

Because PTCG is imperfect-information, we don't use vanilla AlphaZero MCTS. We
use DETERMINIZED search (PIMC / information-set search): combat.py samples the
opponent's hidden cards (via deck_inference) and simulates each candidate move
through the engine. That search is the "expert" — it yields a policy stronger
than the raw net, which becomes the training target (Anthony et al. ExIt;
Silver et al. AlphaZero; for imperfect info cf. AlphaHearts Zero / AlphaJust4Fun
PIMC-ISMCTS).

Loop per iteration:
  1. self-play G games: at each of our decisions, run determinized search over
     the top-k net moves -> improved policy target; play it; record
     (state, target_policy). Opponent samples from a varied deck pool and is
     piloted by a past checkpoint (league) or the heuristic.
  2. label every recorded decision with the game OUTCOME z (1 win / 0 loss / .5 draw).
  3. train: policy head -> cross-entropy to search target; value head -> MSE to z.
  4. checkpoint; occasionally snapshot to the league.

DESIGN CHOICE: we train to pilot ONE fixed deck (--our-deck) against a VARIED
opponent pool (--opp-decks). Specializing is how limited compute buys strength;
deck *selection* is left to the human expert. The net stays deck-agnostic
(card embeddings), so a deck swap = fine-tune from the warm start, not a restart.

This is a runnable SKELETON: correct end-to-end, intentionally small. Strength
comes from scaling games/search/iterations on Kaggle or cloud (see notes).

Usage:
  python selfplay_rl.py --iters 5 --games 40 --warm model.pt \
      --our-deck deck.csv --opp-decks decks/ --out rl_model.pt
"""
from __future__ import annotations
import argparse, glob, os, random, sys, time, warnings
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


# ----- net helpers -----
def load_net(num_ids, warm, dev):
    net = M.PointerPolicyValueNet(num_card_ids=num_ids, d=96).to(dev)
    if warm and os.path.exists(warm):
        try:
            net.load_state_dict(torch.load(warm, map_location=dev))
            print(f"warm-started from {warm}")
        except Exception as e:
            print(f"warm start failed ({e}); training from scratch")
    else:
        print("no warm start; training from scratch (BC warm start strongly recommended)")
    return net


def infer(net, enc, dev):
    """Return (policy over options [O], value scalar, O)."""
    X, _ = collate([{**enc, "target": [0]}])
    X = {k: v.to(dev) for k, v in X.items()}
    with torch.no_grad():
        logits, value = net(X)
    m = X["option_mask"][0]
    lg = logits[0].masked_fill(m < 0.5, -1e9)
    p = torch.softmax(lg, 0).cpu().numpy()
    return p, float(value[0]), int(m.sum().item())


# ----- the determinized-search "expert" -----
def expert(obs, deck, db, atk, net, dev, predictor, topk=3, temp=0.4, plies=1):
    """
    Policy improvement: evaluate the net's top-k options with a determinized
    engine rollout, return (encoded_state, improved_target[O], chosen_index).
    Falls back to the raw net policy when search isn't applicable.
    """
    enc = fx.encode_observation(obs, attack_lookup=atk)
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    O = len(opts)
    p, _, _ = infer(net, enc, dev)
    target = p[:O].copy()
    if combat.available() and O > 1 and sel.get("type") == 0 and (sel.get("minCount", 1) or 0) <= 1:
        order = list(np.argsort(-p[:O]))[:max(topk, 2)]
        vals = {}
        for i in order:
            _, vi = combat.simulate_line(obs, [int(i)], deck, db, atk, predictor, plies=plies)
            if vi is not None:
                vals[int(i)] = vi
        if len(vals) >= 2:
            arr = np.array([vals.get(i, min(vals.values()) - 0.5) for i in range(O)], dtype=np.float64)
            ex = np.exp((arr - arr.max()) / max(temp, 1e-3))
            if ex.sum() > 0:
                target = ex / ex.sum()
    if target.sum() <= 0:
        target = np.ones(O) / max(O, 1)
    choice = int(np.argmax(target))
    return enc, target, choice


def opp_move(obs, db, atk, opp_net, dev):
    sel = obs.get("select") or {}
    O = len(sel.get("option", []))
    if opp_net is None or (sel.get("minCount", 1) or 0) > 1 or O == 0:
        return H.select(obs, db=db, attack_db=atk)
    p, _, _ = infer(opp_net, fx.encode_observation(obs, attack_lookup=atk), dev)
    return [int(np.argmax(p[:O]))]


def play_game(net, our_deck, opp_deck, our_seat, db, atk, dev, opp_net, topk, plies):
    trk = DI.OpponentTracker()
    lib = DI.ArchetypeLibrary().fit([("our", our_deck)])
    predictor = lambda o: (DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3))
    decks = [None, None]; decks[our_seat] = our_deck; decks[1 - our_seat] = opp_deck
    obs, _ = game.battle_start(decks[0], decks[1])
    samples = []; s = 0
    while s < 4000:
        cur = obs.get("current") or {}
        if cur.get("result", -1) >= 0:
            break
        sel = obs.get("select")
        if sel is None:
            yi = cur.get("yourIndex", 0); obs = game.battle_select(decks[yi]); s += 1; continue
        yi = cur.get("yourIndex", 0)
        if yi == our_seat:
            trk.update(obs)
            if (sel.get("minCount", 1) or 0) <= 1 and len(sel.get("option", [])) > 0:
                enc, target, choice = expert(obs, our_deck, db, atk, net, dev, predictor, topk, plies=plies)
                samples.append([enc, target, None])
                act = [choice]
            else:
                act = H.select(obs, db=db, attack_db=atk)
            obs = game.battle_select(act); s += 1
        else:
            obs = game.battle_select(opp_move(obs, db, atk, opp_net, dev)); s += 1
    res = (obs.get("current") or {}).get("result", -1)
    z = 1.0 if res == our_seat else (0.5 if res == 2 else 0.0)
    for smp in samples:
        smp[2] = z
    game.battle_finish()
    return samples, (res == our_seat)


def train_on(net, samples, dev, epochs, bs, lr, vcoef, log):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data = [s for s in samples if s[1] is not None and s[2] is not None]
    net.train()
    for ep in range(epochs):
        random.shuffle(data)
        pl_sum = vl_sum = n = 0
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            X, _ = collate([{**enc, "target": [0]} for enc, _, _ in batch])
            X = {k: v.to(dev) for k, v in X.items()}
            logits, value = net(X)
            mask = X["option_mask"]
            logp = torch.log_softmax(logits.masked_fill(mask < 0.5, -1e9), 1)
            W = logits.shape[1]
            tgt = torch.zeros(len(batch), W)
            for b, (_, td, _) in enumerate(batch):
                tgt[b, :len(td)] = torch.tensor(np.asarray(td), dtype=torch.float32)
            tgt = tgt.to(dev)
            policy_loss = -(tgt * logp).sum(1).mean()
            z = torch.tensor([s[2] for s in batch], dtype=torch.float32, device=dev)
            value_loss = ((value - z) ** 2).mean()
            loss = policy_loss + vcoef * value_loss
            opt.zero_grad(); loss.backward(); opt.step()
            pl_sum += policy_loss.item() * len(batch); vl_sum += value_loss.item() * len(batch); n += len(batch)
        if n and log:
            stats.log(log, event="rl_train", policy_loss=round(pl_sum / n, 4),
                      value_loss=round(vl_sum / n, 4), samples=n)
        if n:
            print(f"    train: policy_loss {pl_sum/n:.4f}  value_loss {vl_sum/n:.4f}  ({n} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--games", type=int, default=40, help="self-play games per iteration")
    ap.add_argument("--warm", default="model.pt", help="BC checkpoint to warm-start from")
    ap.add_argument("--our-deck", default="deck.csv")
    ap.add_argument("--opp-decks", default="decks/", help="dir of opponent deck .csv (the field)")
    ap.add_argument("--out", default="rl_model.pt")
    ap.add_argument("--topk", type=int, default=3, help="options searched per decision")
    ap.add_argument("--plies", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--vcoef", type=float, default=1.0)
    ap.add_argument("--opponent", choices=["league", "heuristic"], default="league")
    ap.add_argument("--log", default="logs/rl.jsonl")
    a = ap.parse_args()

    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in __import__("json").load(open("attack_table.json")).items()}
    our_deck = [int(x) for x in open(a.our_deck).read().split()][:60]
    pool = []
    for f in sorted(glob.glob(os.path.join(a.opp_decks, "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            pool.append((os.path.basename(f)[:-4], d))
    if not pool:
        pool = [("mirror", our_deck)]
    print(f"opponent pool ({len(pool)}): {[n for n,_ in pool]}")

    dev = device()
    net = load_net(1268, a.warm, dev)
    league = [a.out]                       # past checkpoints to play against

    for it in range(a.iters):
        t0 = time.time(); samples = []; wins = 0
        opp_net = None
        if a.opponent == "league":         # play vs a past snapshot of ourselves
            snap = M.PointerPolicyValueNet(num_card_ids=1268, d=96).to(dev)
            snap.load_state_dict(net.state_dict()); snap.eval(); opp_net = snap
        for g in range(a.games):
            opp_name, opp_deck = random.choice(pool)
            smp, won = play_game(net, our_deck, opp_deck, g % 2, db, atk, dev,
                                 opp_net, a.topk, a.plies)
            samples += smp; wins += int(won)
        wr = wins / a.games
        print(f"iter {it+1}/{a.iters}: {a.games} games, winrate {wr:.0%}, "
              f"{len(samples)} decisions, {(time.time()-t0)/max(a.games,1):.2f}s/game")
        stats.log(a.log, event="rl_iter", iter=it + 1, winrate=round(wr, 3),
                  games=a.games, decisions=len(samples))
        train_on(net, samples, dev, a.epochs, a.bs, a.lr, a.vcoef, a.log)
        torch.save(net.state_dict(), a.out)
        __import__("json").dump({"num_card_ids": 1268, "d": 96}, open("model_meta.json", "w"))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
