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
import ismcts
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
def expert(obs, deck, db, atk, net, dev, predictor, topk=3, temp=0.4, plies=1,
           explore=False, play_temp=1.0, dir_eps=0.25, dir_alpha=0.3,
           search_mode="flat", ismcts_worlds=3, ismcts_sims=16, leaf_eval="value"):
    """
    Policy improvement, returning (encoded_state, improved_target[O], chosen_index).
    search_mode="flat": net's top-k options scored by a 1-ply determinized rollout.
    search_mode="ismcts": full per-determinization MCTS; target = visit distribution.
    The CLEAN improved target trains the net; the PLAYED move adds AlphaZero-style
    exploration (Dirichlet root noise + temperature) when explore=True.
    """
    enc = fx.encode_observation(obs, attack_lookup=atk)
    sel = obs.get("select") or {}
    opts = sel.get("option", [])
    O = len(opts)
    p, _, _ = infer(net, enc, dev)
    target = p[:O].copy()
    single = (sel.get("minCount", 1) or 0) <= 1
    sval = None                                    # search-backed value estimate (low-variance target)

    if search_mode == "ismcts" and ismcts.available() and O > 1 and single:
        pol, sval, _ = ismcts.search(obs, deck, db, atk, net, dev, predictor,
                                  n_worlds=ismcts_worlds, n_sims=ismcts_sims, leaf_eval=leaf_eval)
        if pol is not None and len(pol) >= O:
            target = np.asarray(pol[:O], dtype=np.float64)
    elif combat.available() and O > 1 and sel.get("type") == 0 and single:
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

    # played move: clean greedy by default; explore with noise + temperature
    if explore and O > 1:
        play = target.copy()
        if dir_eps > 0:
            noise = np.random.dirichlet([dir_alpha] * O)
            play = (1 - dir_eps) * play + dir_eps * noise
        if play_temp > 1e-3:
            pt = play ** (1.0 / play_temp)
            pt = pt / pt.sum() if pt.sum() > 0 else np.ones(O) / O
            choice = int(np.random.choice(O, p=pt))
        else:
            choice = int(np.argmax(play))
    else:
        choice = int(np.argmax(target))
    return enc, target, choice, sval


def opp_move(obs, db, atk, opp_net, dev):
    sel = obs.get("select") or {}
    O = len(sel.get("option", []))
    if opp_net is None or (sel.get("minCount", 1) or 0) > 1 or O == 0:
        return H.select(obs, db=db, attack_db=atk)
    p, _, _ = infer(opp_net, fx.encode_observation(obs, attack_lookup=atk), dev)
    return [int(np.argmax(p[:O]))]


def play_game(net, our_deck, opp_deck, our_seat, db, atk, dev, opp_net, topk, plies,
              explore=True, greedy_after=8, library=None,
              search_mode="flat", ismcts_worlds=3, ismcts_sims=16, leaf_eval="value"):
    trk = DI.OpponentTracker()
    lib = library if library is not None else DI.ArchetypeLibrary().fit([("our", our_deck)])
    predictor = lambda o, rng=None: (DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3, rng=rng))
    decks = [None, None]; decks[our_seat] = our_deck; decks[1 - our_seat] = opp_deck
    obs, _ = game.battle_start(decks[0], decks[1])
    samples = []; s = 0; our_moves = 0
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
                # temperature 1.0 early (explore), greedy later (exploit)
                ptemp = 1.0 if our_moves < greedy_after else 0.0
                enc, target, choice, sval = expert(obs, our_deck, db, atk, net, dev, predictor,
                                             topk, plies=plies, explore=explore, play_temp=ptemp,
                                             search_mode=search_mode, ismcts_worlds=ismcts_worlds,
                                             ismcts_sims=ismcts_sims, leaf_eval=leaf_eval)
                samples.append([enc, target, None, sval])
                act = [choice]; our_moves += 1
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
            X, _ = collate([{**s[0], "target": [0]} for s in batch])
            X = {k: v.to(dev) for k, v in X.items()}
            logits, value = net(X)
            mask = X["option_mask"]
            logp = torch.log_softmax(logits.masked_fill(mask < 0.5, -1e9), 1)
            W = logits.shape[1]
            tgt = torch.zeros(len(batch), W)
            for b, s in enumerate(batch):
                td = s[1]
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


def _snapshot(net, dev):
    snap = M.PointerPolicyValueNet(num_card_ids=1268, d=96).to(dev)
    snap.load_state_dict(net.state_dict()); snap.eval()
    return snap


def main():
    import arena
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=80, help="self-play games per iteration")
    ap.add_argument("--warm", default="model.pt", help="BC checkpoint to warm-start from")
    ap.add_argument("--our-deck", default="deck.csv")
    ap.add_argument("--opp-decks", default="decks/", help="dir of opponent deck .csv (the field)")
    ap.add_argument("--out", default="rl_model.pt", help="best (gated) checkpoint is written here")
    ap.add_argument("--topk", type=int, default=3, help="options searched per decision (flat)")
    ap.add_argument("--plies", type=int, default=1)
    ap.add_argument("--search", choices=["flat", "ismcts"], default="flat",
                    help="expert search: flat 1-ply rollout, or full per-determinization MCTS")
    ap.add_argument("--ismcts-worlds", type=int, default=3)
    ap.add_argument("--ismcts-sims", type=int, default=16)
    ap.add_argument("--leaf", choices=["value", "rollout"], default="value",
                    help="ISMCTS leaf eval: net value head, or engine-oracle heuristic rollout")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vcoef", type=float, default=1.0)
    ap.add_argument("--gate-games", type=int, default=60, help="games for the promotion gate")
    ap.add_argument("--gate-threshold", type=float, default=0.55)
    ap.add_argument("--league-size", type=int, default=5)
    ap.add_argument("--field-every", type=int, default=3, help="detailed field-eval cadence (iters)")
    ap.add_argument("--field-games", type=int, default=6, help="games/deck for the per-iter field check")
    ap.add_argument("--early-stop-margin", type=float, default=0.05,
                    help="stop if field win-rate falls this far below the BC baseline (2 iters running)")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--adversary-frac", type=float, default=0.10,
                    help="fraction of self-play games vs off-meta adversary decks")
    ap.add_argument("--log", default="logs/rl.jsonl")
    ap.add_argument("--time-budget-min", type=float, default=0.0,
                    help="stop before starting an iteration once this many wall-clock minutes elapse "
                         "(0 = unlimited). Use ~660 on Kaggle's 12h kernels to checkpoint before the kill.")
    ap.add_argument("--latest-out", default="",
                    help="path for the continuously-trained net, saved every iteration for crash-safe "
                         "resume (default: <out>.latest.pt). Resume next session by warming from this.")
    ap.add_argument("--dump-samples", default="",
                    help="dir to dump self-play samples (enc, policy_target, outcome_z, search_value) each "
                         "iteration, as value-head training data for fit_value.py (byproduct, no extra cost).")
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
    adv_pool = []
    for f in sorted(glob.glob(os.path.join(a.opp_decks, "adversary", "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            adv_pool.append((os.path.basename(f)[:-4], d))
    print(f"opponent pool ({len(pool)} meta + {len(adv_pool)} adversary)")
    library = DI.library_from_pool(our_deck, a.opp_decks)   # opponent inference coverage
    print(f"deck_inference library: {len(library.decks)} archetypes")
    if a.search == "ismcts":
        print(f"SEARCH: ISMCTS ({a.ismcts_worlds} worlds x {a.ismcts_sims} sims) | "
              f"lr {a.lr} | {a.iters} iters x {a.games} games", flush=True)
    else:
        print(f"SEARCH: flat (top-{a.topk}, {a.plies}-ply) | lr {a.lr} | "
              f"{a.iters} iters x {a.games} games", flush=True)

    dev = device()
    net = load_net(1268, a.warm, dev)            # the continuously-trained candidate
    best = _snapshot(net, dev)                    # gated best == league anchor + output
    league = [best]                               # past gated checkpoints (AlphaStar-style)
    torch.save(best.state_dict(), a.out)
    __import__("json").dump({"num_card_ids": 1268, "d": 96}, open("model_meta.json", "w"))

    # baseline: how the BC warm-start plays the field, BEFORE any RL touches it.
    print("\n=== baseline: BC net vs the field (this is the bar to beat) ===")
    baseline_field = arena.field_eval(arena.make_net_agent(best, dev, db, atk, our_deck),
                                      our_deck, pool, a.field_games, db=db, atk=atk,
                                      log=a.log)
    stats.log(a.log, event="rl_baseline", field_winrate=round(baseline_field, 3))
    below = 0
    session_t0 = time.time()
    latest_out = a.latest_out or (os.path.splitext(a.out)[0] + ".latest.pt")
    print()
    for it in range(a.iters):
        if a.time_budget_min and (time.time() - session_t0) / 60.0 >= a.time_budget_min:
            print(f"\nTIME BUDGET reached ({a.time_budget_min:.0f} min) before iter {it+1}. "
                  f"Stopping cleanly; checkpoints are safe (best={a.out}, latest={latest_out}). "
                  f"Resume next session by warming from {latest_out}.", flush=True)
            stats.log(a.log, event="rl_time_budget_stop", iter=it,
                      elapsed_min=round((time.time() - session_t0) / 60.0, 1))
            break
        t0 = time.time(); samples = []; wins = 0
        for g in range(a.games):
            opp_net = random.choice(league)           # league opponent (avoids cycling)
            if adv_pool and random.random() < a.adversary_frac:
                _, opp_deck = random.choice(adv_pool)  # off-meta exploiter (robustness)
            else:
                _, opp_deck = random.choice(pool)      # the meta field
            smp, won = play_game(net, our_deck, opp_deck, g % 2, db, atk, dev,
                                 opp_net, a.topk, a.plies, explore=True, library=library,
                                 search_mode=a.search, ismcts_worlds=a.ismcts_worlds,
                                 ismcts_sims=a.ismcts_sims, leaf_eval=a.leaf)
            samples += smp; wins += int(won)
            if (g + 1) % max(a.games // 8, 1) == 0:
                el = time.time() - t0
                print(f"    [iter {it+1}] self-play {g+1}/{a.games}  "
                      f"winrate {wins/(g+1):.0%}  {el/(g+1):.1f}s/game  "
                      f"~{el/(g+1)*(a.games-g-1)/60:.0f}m left in iter", flush=True)
        wr = wins / a.games
        print(f"iter {it+1}/{a.iters}: {a.games} games, self-play winrate {wr:.0%}, "
              f"{len(samples)} decisions, {(time.time()-t0)/max(a.games,1):.2f}s/game, league={len(league)}")
        stats.log(a.log, event="rl_iter", iter=it + 1, winrate=round(wr, 3),
                  games=a.games, decisions=len(samples), league=len(league))
        train_on(net, samples, dev, a.epochs, a.bs, a.lr, a.vcoef, a.log)
        # crash-safe: persist the continuously-trained net every iteration (independent of the gate),
        # so a 12h kill mid-run never loses progress and resume can continue this trajectory.
        torch.save(net.state_dict(), latest_out)
        # byproduct harvest: dump this iteration's samples as value-head training data (free).
        if a.dump_samples:
            os.makedirs(a.dump_samples, exist_ok=True)
            import pickle
            with open(os.path.join(a.dump_samples, f"iter{it+1:03d}.pkl"), "wb") as f:
                pickle.dump(samples, f)

        # CI-gated promotion: candidate must beat best in a mirror match
        mk_cand = arena.make_net_agent(net, dev, db, atk, our_deck)
        mk_best = arena.make_net_agent(best, dev, db, atk, our_deck)
        promoted, score, elo = arena.gate(mk_cand, mk_best, our_deck, a.gate_games,
                                          a.gate_threshold, log=a.log, tag=f"iter{it+1} gate")
        if promoted:
            best = _snapshot(net, dev)
            league.append(best)
            league[:] = league[-a.league_size:]
            torch.save(best.state_dict(), a.out)
            print(f"  promoted -> new best saved to {a.out} (Elo +{elo:.0f} vs prev best)")
        # per-iteration field check vs the BC baseline (catch degradation early)
        detailed = (it + 1) % a.field_every == 0
        cand_field = arena.field_eval(mk_cand, our_deck, pool, a.field_games,
                                      db=db, atk=atk, log=a.log, verbose=detailed,
                                      tag=f"iter{it+1} field eval")
        stats.log(a.log, event="rl_field", iter=it + 1, field_winrate=round(cand_field, 3),
                  baseline=round(baseline_field, 3))
        print(f"  field: {cand_field:.0%} (BC baseline {baseline_field:.0%})")
        if detailed and adv_pool:
            arena.field_eval(mk_cand, our_deck, adv_pool, a.field_games,
                             db=db, atk=atk, log=a.log, tag="ADVERSARY eval (off-meta)")
        if not a.no_early_stop:
            if cand_field < baseline_field - a.early_stop_margin:
                below += 1
                if below >= 2:
                    print(f"\nEARLY STOP: field win-rate below BC baseline for 2 iters running "
                          f"({cand_field:.0%} < {baseline_field:.0%}). The RL signal is degrading "
                          f"the net — best gated checkpoint ({a.out}) is unchanged. "
                          f"This is the trigger to move to ISMCTS (see docs/roadmap.md).")
                    stats.log(a.log, event="rl_early_stop", iter=it + 1,
                              field_winrate=round(cand_field, 3), baseline=round(baseline_field, 3))
                    break
            else:
                below = 0
    print(f"done. best gated checkpoint: {a.out}")


if __name__ == "__main__":
    main()
