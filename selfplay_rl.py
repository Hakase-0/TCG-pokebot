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
  4. checkpoint the latest net every iter (gateless, AlphaZero/KataGo-style);
     periodically snapshot it into the league for opponent diversity.

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
from train_bc import collate, device, is_xla, xla_mark_step
from cg import game


# ----- net helpers -----
def load_net(num_ids, warm, dev):
    net = M.from_meta(dev, warm=warm, num_ids_default=num_ids)
    if warm and os.path.exists(warm):
        print(f"warm-started from {warm}")
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


def opp_move(obs, db, atk, opp_net, dev, opp_deck=None, predictor=None,
             search_mode="raw", ismcts_worlds=2, ismcts_sims=24, leaf_eval="value"):
    """Opponent's move. Default 'raw' = net argmax (fast sparring partner, original
    behavior). With search_mode='ismcts' the opponent ALSO runs ISMCTS — symmetric
    search-vs-search self-play: a stronger opponent and more honest value targets,
    at ~2x the self-play cost (pair with --leaf value to keep that affordable).
    Falls back to raw argmax whenever search can't run (engine missing, <=1 option,
    multi-select, or no opp_deck)."""
    sel = obs.get("select") or {}
    O = len(sel.get("option", []))
    if opp_net is None or (sel.get("minCount", 1) or 0) > 1 or O == 0:
        return H.select(obs, db=db, attack_db=atk)
    if search_mode == "ismcts" and O > 1 and opp_deck is not None and ismcts.available():
        _, _, ch = ismcts.search(obs, opp_deck, db, atk, opp_net, dev, predictor,
                                 n_worlds=ismcts_worlds, n_sims=ismcts_sims, leaf_eval=leaf_eval)
        if ch is not None:
            return [int(ch)]
    p, _, _ = infer(opp_net, fx.encode_observation(obs, attack_lookup=atk), dev)
    return [int(np.argmax(p[:O]))]


def play_game(net, our_deck, opp_deck, our_seat, db, atk, dev, opp_net, topk, plies,
              explore=True, greedy_after=8, library=None,
              search_mode="flat", ismcts_worlds=3, ismcts_sims=16, leaf_eval="value",
              opp_search="raw", opp_ismcts_worlds=2, opp_ismcts_sims=24, opp_leaf="value"):
    trk = DI.OpponentTracker()
    lib = library if library is not None else DI.ArchetypeLibrary().fit([("our", our_deck)])
    predictor = lambda o, rng=None: (DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3, rng=rng))
    # opponent-side inference state, built only when the opponent itself runs ISMCTS
    opp_trk = DI.OpponentTracker() if opp_search == "ismcts" else None
    opp_predictor = (lambda o, rng=None: DI.predict_opponent_zones(o, opp_trk, lib, card_db=db, min_conf=0.3, rng=rng)) \
        if opp_search == "ismcts" else None
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
            if opp_trk is not None:
                opp_trk.update(obs)                      # track US from the opponent's view, for its search
            obs = game.battle_select(opp_move(obs, db, atk, opp_net, dev,
                                              opp_deck=opp_deck, predictor=opp_predictor,
                                              search_mode=opp_search, ismcts_worlds=opp_ismcts_worlds,
                                              ismcts_sims=opp_ismcts_sims, leaf_eval=opp_leaf)); s += 1
    res = (obs.get("current") or {}).get("result", -1)
    z = 1.0 if res == our_seat else (0.5 if res == 2 else 0.0)
    for smp in samples:
        smp[2] = z
    game.battle_finish()
    return samples, (res == our_seat)


def train_on(net, samples, dev, epochs, bs, lr, vcoef, log):
    # TPU: park the net on the XLA device for the batched train step only. Inference
    # (self-play, field-eval, snapshots) runs batch-1 on CPU where XLA's per-shape
    # recompiles would hurt; only this dense, batched step benefits from the TPU.
    on_xla = is_xla(dev)
    if on_xla:
        net.to(dev)
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
            if on_xla: xla_mark_step()   # flush this step's graph onto the TPU
            pl_sum += policy_loss.item() * len(batch); vl_sum += value_loss.item() * len(batch); n += len(batch)
        if n and log:
            stats.log(log, event="rl_train", policy_loss=round(pl_sum / n, 4),
                      value_loss=round(vl_sum / n, 4), samples=n)
        if n:
            print(f"    train: policy_loss {pl_sum/n:.4f}  value_loss {vl_sum/n:.4f}  ({n} samples)")
    if on_xla:
        net.to("cpu")   # hand trained weights back on CPU for inference/snapshots/saves


def _snapshot(net, dev):
    import copy
    snap = copy.deepcopy(net).to(dev); snap.eval()
    return snap


# ===================== parallel self-play =====================
# The cabt engine keeps per-PROCESS global state (game.battle_* / ismcts._api), so
# only ONE game can run per process — parallelism must be across PROCESSES, not
# threads. Self-play here is CPU/engine-bound (each ISMCTS sim interleaves a native
# engine search_step with a batch-1 net forward), so the GPU sits nearly idle during
# self-play; the win is filling the otherwise-unused CPU cores. Workers therefore run
# net inference on CPU (cheap at batch 1, and frees the single GPU for the training
# step that actually benefits from it). 'spawn' gives each worker a fresh interpreter
# (no inherited CUDA context / native engine handles); weights are shipped per
# iteration as picklable CPU state_dicts. If anything in this path fails we fall back
# to the original sequential loop, so correctness never depends on it.
_WK = {}   # per-worker state, populated by _worker_init in each child process


def _cpu_sd(net):
    """CPU copy of a net's state_dict, safe to pickle to worker processes."""
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


def _worker_init(cand_sd, league_sds, opp_decks_dir, our_deck_path, cfg):
    """Runs once per worker process. Fresh engine (auto-inits on import), CPU nets
    rebuilt from state_dicts, deck pool + inference library loaded from disk (cheap;
    avoids pickling big objects). Mirrors the parent's net modes exactly so this is a
    pure throughput change: the candidate stays in train() mode (as the parent's
    continuously-trained net is during self-play), league nets are eval() (as
    _snapshot makes them)."""
    import glob as _g, json as _j
    import torch as _t
    import model as _M, deck_inference as _DI
    from evaluate import CardDB
    _t.set_num_threads(1)                       # processes provide the parallelism; don't let each
    dev = _t.device("cpu")                      # worker's BLAS fan out and oversubscribe the cores
    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in _j.load(open("attack_table.json")).items()}
    pool = []
    for f in sorted(_g.glob(os.path.join(opp_decks_dir, "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            pool.append((os.path.basename(f)[:-4], d))
    our_deck = None
    if our_deck_path and os.path.exists(our_deck_path):
        our_deck = [int(x) for x in open(our_deck_path).read().split()][:60]
    if not pool and our_deck is not None:
        pool = [("mirror", our_deck)]
    if our_deck is None and pool:
        our_deck = pool[0][1]
    library = _DI.library_from_pool(our_deck, opp_decks_dir)

    def _mk(sd, ev):
        n = _M.from_meta(dev, warm=None)
        n.load_state_dict(sd)
        n.eval() if ev else n.train()
        return n
    _WK.clear()
    _WK.update(dict(dev=dev, db=db, atk=atk, library=library, cfg=cfg,
                    cand=_mk(cand_sd, False), league=[_mk(sd, True) for sd in league_sds]))


def _play_one(task):
    """Play ONE self-play game on CPU. task = (our_deck, opp_deck, our_seat, league_idx, seed).
    Per-task seeding makes every game diverse (workers would otherwise share RNG state)
    and reproducible."""
    import random as _r, numpy as _np, torch as _t
    our_deck, opp_deck, our_seat, league_idx, seed = task
    _r.seed(seed); _np.random.seed(seed % (2**32 - 1)); _t.manual_seed(seed)
    w = _WK; c = w["cfg"]
    smp, won = play_game(w["cand"], our_deck, opp_deck, our_seat, w["db"], w["atk"], w["dev"],
                         w["league"][league_idx], c["topk"], c["plies"], explore=True, library=w["library"],
                         search_mode=c["search"], ismcts_worlds=c["ismcts_worlds"],
                         ismcts_sims=c["ismcts_sims"], leaf_eval=c["leaf"],
                         opp_search=c["opp_search"], opp_ismcts_worlds=c["opp_worlds"],
                         opp_ismcts_sims=c["opp_sims"], opp_leaf=c["opp_leaf"])
    return smp, int(won)


def _selfplay_assignments(a, pool, adv_pool, n_league):
    """Per-game (our_deck, opp_deck, our_seat, league_idx, seed). One sampling routine
    shared by both the parallel and sequential paths so they draw the same distribution
    (deck sampled from the pool, off-meta adversary at --adversary-frac, alternating seat,
    league opponent by index)."""
    out = []
    for g in range(a.games):
        league_idx = random.randrange(n_league)
        _, da = random.choice(pool)
        if adv_pool and random.random() < a.adversary_frac:
            _, opp_deck = random.choice(adv_pool)
        else:
            _, opp_deck = random.choice(pool)
        out.append((da, opp_deck, g % 2, league_idx, random.randrange(2**31)))
    return out


def _progress(it, done, games, wins, t0, suffix=""):
    if games and done % max(games // 8, 1) == 0:
        el = time.time() - t0
        print(f"    [iter {it+1}] self-play {done}/{games}  winrate {wins/max(done,1):.0%}  "
              f"{el/max(done,1):.1f}s/game  ~{el/max(done,1)*(games-done)/60:.0f}m left in iter{suffix}",
              flush=True)


def _run_selfplay_parallel(a, assignments, cand_sd, league_sds, cfg, it):
    """Distribute the iteration's games across worker processes; aggregate samples as
    they complete (order-independent — training shuffles a replay buffer anyway)."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    W = max(1, min(a.workers, len(assignments)))
    samples = []; wins = done = 0; t0 = time.time()
    with ctx.Pool(W, initializer=_worker_init,
                  initargs=(cand_sd, league_sds, a.opp_decks, a.our_deck, cfg)) as pool:
        for smp, won in pool.imap_unordered(_play_one, assignments):
            samples += smp; wins += won; done += 1
            _progress(it, done, a.games, wins, t0, suffix=f"  ({W} workers)")
    return samples, wins


def _run_selfplay_sequential(a, assignments, net, league, db, atk, dev, library, cfg, it):
    """Original single-process self-play (the safe fallback / --workers 1)."""
    samples = []; wins = 0; t0 = time.time()
    for g, (da, opp_deck, our_seat, league_idx, seed) in enumerate(assignments):
        random.seed(seed); np.random.seed(seed % (2**32 - 1)); torch.manual_seed(seed)
        smp, won = play_game(net, da, opp_deck, our_seat, db, atk, dev,
                             league[league_idx], cfg["topk"], cfg["plies"], explore=True, library=library,
                             search_mode=cfg["search"], ismcts_worlds=cfg["ismcts_worlds"],
                             ismcts_sims=cfg["ismcts_sims"], leaf_eval=cfg["leaf"],
                             opp_search=cfg["opp_search"], opp_ismcts_worlds=cfg["opp_worlds"],
                             opp_ismcts_sims=cfg["opp_sims"], opp_leaf=cfg["opp_leaf"])
        samples += smp; wins += int(won)
        _progress(it, g + 1, a.games, wins, t0)
    return samples, wins


def main():
    import arena
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--games", type=int, default=80, help="self-play games per iteration")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel self-play worker PROCESSES (0 = auto = os.cpu_count(); 1 = original "
                         "single-process loop). Self-play is CPU/engine-bound, so on a multi-core kernel "
                         "this multiplies games/hour ~linearly with cores; workers run net inference on CPU "
                         "and the GPU is reserved for the training step. Falls back to sequential on any "
                         "multiprocessing error.")
    ap.add_argument("--warm", default="model.pt", help="BC checkpoint to warm-start from")
    ap.add_argument("--our-deck", default="deck.csv")
    ap.add_argument("--opp-decks", default="decks/", help="dir of opponent deck .csv (the field)")
    ap.add_argument("--out", default="rl_model.pt", help="output checkpoint (latest trained net, written every iter)")
    ap.add_argument("--topk", type=int, default=3, help="options searched per decision (flat)")
    ap.add_argument("--plies", type=int, default=1)
    ap.add_argument("--search", choices=["flat", "ismcts"], default="flat",
                    help="expert search: flat 1-ply rollout, or full per-determinization MCTS")
    ap.add_argument("--ismcts-worlds", type=int, default=3)
    ap.add_argument("--ismcts-sims", type=int, default=16)
    ap.add_argument("--leaf", choices=["value", "rollout"], default="value",
                    help="ISMCTS leaf eval: net value head, or engine-oracle heuristic rollout")
    ap.add_argument("--opp-search", choices=["raw", "ismcts"], default="raw",
                    help="opponent move policy: 'raw' net argmax (fast, default, original behavior), "
                         "or 'ismcts' (symmetric search-vs-search self-play; stronger sparring + more "
                         "honest value targets, ~2x self-play cost — pair with --leaf value).")
    ap.add_argument("--opp-ismcts-worlds", type=int, default=0, help="opponent ISMCTS worlds (0 = reuse --ismcts-worlds)")
    ap.add_argument("--opp-ismcts-sims", type=int, default=0, help="opponent ISMCTS sims (0 = reuse --ismcts-sims)")
    ap.add_argument("--opp-leaf", choices=["same", "value", "rollout"], default="same",
                    help="opponent ISMCTS leaf eval ('same' = match --leaf)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vcoef", type=float, default=1.0)
    ap.add_argument("--replay-buffer", type=int, default=30000,
                    help="max self-play decisions retained ACROSS iterations for training (sliding "
                         "window, AlphaZero/KataGo-style). Each iter trains on a shuffled draw from this "
                         "buffer instead of only the freshest games, decorrelating minibatches so a single "
                         "iteration's self-correlated data can't overfit/forget the net. ~6-7 iters of "
                         "history at these settings; 0 = disabled (train only on the current iter — "
                         "original behavior).")
    ap.add_argument("--replay-preload", default="",
                    help="dir of prior-session sample .pkl dumps to PRELOAD into the replay buffer at "
                         "startup (typically the resumed --dump-samples dir). Lets training continue on "
                         "games harvested in EARLIER sessions instead of an empty buffer; the merged set "
                         "is trimmed to --replay-buffer most-recent decisions.")
    ap.add_argument("--gate-games", type=int, default=60,
                    help="(deprecated: promotion gate removed; accepted but ignored)")
    ap.add_argument("--gate-threshold", type=float, default=0.55,
                    help="(deprecated: promotion gate removed; accepted but ignored)")
    ap.add_argument("--league-every", type=int, default=2,
                    help="snapshot the candidate into the league every N iters for opponent "
                         "diversity (replaces gate-based promotion; costs no arena games). 0 disables.")
    ap.add_argument("--league-size", type=int, default=5)
    ap.add_argument("--field-every", type=int, default=3, help="detailed field-eval cadence (iters)")
    ap.add_argument("--field-games", type=int, default=6, help="games/deck for the per-iter field check")
    ap.add_argument("--early-stop-margin", type=float, default=0.10,
                    help="stop if field win-rate falls this far below the baseline "
                         "(for --early-stop-patience iters running)")
    ap.add_argument("--early-stop-patience", type=int, default=3,
                    help="consecutive iters below the early-stop floor before bailing "
                         "(higher = more tolerant of field-eval noise, esp. on a cold start)")
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
    opp_worlds = a.opp_ismcts_worlds or a.ismcts_worlds
    opp_sims = a.opp_ismcts_sims or a.ismcts_sims
    opp_leaf = a.leaf if a.opp_leaf == "same" else a.opp_leaf
    if a.workers <= 0:
        a.workers = max(1, (os.cpu_count() or 1))

    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in __import__("json").load(open("attack_table.json")).items()}
    pool = []
    for f in sorted(glob.glob(os.path.join(a.opp_decks, "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            pool.append((os.path.basename(f)[:-4], d))
    # AGNOSTIC: the net is trained to pilot the WHOLE pool, not one deck. our_deck is
    # optional (only a fallback / library seed); both seats are sampled from the pool.
    our_deck = None
    if a.our_deck and os.path.exists(a.our_deck):
        our_deck = [int(x) for x in open(a.our_deck).read().split()][:60]
    if not pool:
        if our_deck is None:
            raise SystemExit("no decks/*.csv pool and no --our-deck; build the deck pool first")
        pool = [("mirror", our_deck)]
    if our_deck is None:
        our_deck = pool[0][1]
    adv_pool = []
    for f in sorted(glob.glob(os.path.join(a.opp_decks, "adversary", "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            adv_pool.append((os.path.basename(f)[:-4], d))
    print(f"AGNOSTIC training over {len(pool)} meta decks (+{len(adv_pool)} adversary)", flush=True)
    # FIELD METRIC = meta + off-meta: fold adversaries into the eval pool so the field win-rate
    # (and the early-stop keyed off it) reflects off-meta robustness, not just the meta. Training
    # still over-samples adversaries via --adversary-frac; this changes only what we MEASURE.
    field_pool = pool + adv_pool
    if adv_pool:
        print(f"FIELD eval pool: {len(field_pool)} decks "
              f"({len(pool)} meta + {len(adv_pool)} off-meta)", flush=True)
    library = DI.library_from_pool(our_deck, a.opp_decks)   # opponent inference coverage
    print(f"deck_inference library: {len(library.decks)} archetypes")
    if a.search == "ismcts":
        print(f"SEARCH: ISMCTS ({a.ismcts_worlds} worlds x {a.ismcts_sims} sims) | "
              f"lr {a.lr} | {a.iters} iters x {a.games} games", flush=True)
        if a.opp_search == "ismcts":
            print(f"  OPPONENT also searches: ISMCTS ({opp_worlds}w x {opp_sims}s, "
                  f"leaf={opp_leaf}) — symmetric self-play (~2x cost)", flush=True)
    else:
        print(f"SEARCH: flat (top-{a.topk}, {a.plies}-ply) | lr {a.lr} | "
              f"{a.iters} iters x {a.games} games", flush=True)

    train_dev = device()                          # may be an XLA/TPU device when TCG_DEVICE=tpu
    dev = "cpu" if is_xla(train_dev) else train_dev   # inference/snapshots/saves stay off XLA;
    net = load_net(1268, a.warm, dev)            # only the batched train step parks on train_dev
    best = _snapshot(net, dev)                    # initial anchor: league seed + fixed baseline yardstick
    league = [best]                               # past checkpoints as self-play opponents (AlphaStar-style)
    torch.save(best.state_dict(), a.out)          # seed the output; rewritten with the latest net every iter
    if not os.path.exists("model_meta.json"):    # preserve BC's architecture meta; never clobber dims
        __import__("json").dump({"num_card_ids": 1268, "d": net.d}, open("model_meta.json", "w"))

    # baseline: how the warm-started net pilots the FIELD, before any RL touches it.
    print("\n=== baseline: net pilots the field vs heuristic reference (the bar to beat) ===", flush=True)
    baseline_field = arena.field_eval_agnostic(best, dev, db, atk, field_pool, a.field_games * 3,
                                               log=a.log)
    stats.log(a.log, event="rl_baseline", field_winrate=round(baseline_field, 3))
    below = 0
    replay = []                                   # AlphaZero-style sliding-window replay buffer (see --replay-buffer)
    if a.replay_buffer and a.replay_preload and os.path.isdir(a.replay_preload):
        import pickle as _pk
        pre = []
        for f in sorted(glob.glob(os.path.join(a.replay_preload, "*.pkl"))):
            try:
                with open(f, "rb") as fh:
                    pre += _pk.load(fh)
            except Exception as e:
                print(f"  replay preload skip {f}: {e}", flush=True)
        if pre:
            replay = pre[-a.replay_buffer:]       # cross-session continuity: warm the buffer from past games
            print(f"replay preload: {len(pre)} decisions from {a.replay_preload} "
                  f"-> buffer starts at {len(replay)} (cap {a.replay_buffer})", flush=True)
    _SESS = time.strftime("%y%m%d_%H%M%S")        # unique tag so this session's dumps don't collide with resumed ones
    session_t0 = time.time()
    latest_out = a.latest_out or (os.path.splitext(a.out)[0] + ".latest.pt")
    # picklable scalar config shared by both self-play paths (db/library are reloaded per worker)
    cfg = dict(topk=a.topk, plies=a.plies, search=a.search,
               ismcts_worlds=a.ismcts_worlds, ismcts_sims=a.ismcts_sims, leaf=a.leaf,
               opp_search=a.opp_search, opp_worlds=opp_worlds, opp_sims=opp_sims, opp_leaf=opp_leaf)
    use_parallel = a.workers > 1
    if use_parallel:
        print(f"PARALLEL self-play: up to {a.workers} worker processes (cpu_count={os.cpu_count()}); "
              f"CPU inference in workers, GPU reserved for training.", flush=True)
    print()
    for it in range(a.iters):
        if a.time_budget_min and (time.time() - session_t0) / 60.0 >= a.time_budget_min:
            print(f"\nTIME BUDGET reached ({a.time_budget_min:.0f} min) before iter {it+1}. "
                  f"Stopping cleanly; checkpoints are safe (best={a.out}, latest={latest_out}). "
                  f"Resume next session by warming from {latest_out}.", flush=True)
            stats.log(a.log, event="rl_time_budget_stop", iter=it,
                      elapsed_min=round((time.time() - session_t0) / 60.0, 1))
            break
        t0 = time.time()
        assignments = _selfplay_assignments(a, pool, adv_pool, len(league))
        if use_parallel:
            try:
                samples, wins = _run_selfplay_parallel(a, assignments, _cpu_sd(net),
                                                       [_cpu_sd(s) for s in league], cfg, it)
            except Exception as e:
                print(f"  [warn] parallel self-play failed ({type(e).__name__}: {e}); "
                      f"falling back to sequential for the rest of the run.", flush=True)
                use_parallel = False
                samples, wins = _run_selfplay_sequential(a, assignments, net, league, db, atk, dev, library, cfg, it)
        else:
            samples, wins = _run_selfplay_sequential(a, assignments, net, league, db, atk, dev, library, cfg, it)
        wr = wins / a.games
        print(f"iter {it+1}/{a.iters}: {a.games} games, self-play winrate {wr:.0%}, "
              f"{len(samples)} decisions, {(time.time()-t0)/max(a.games,1):.2f}s/game, league={len(league)}")
        stats.log(a.log, event="rl_iter", iter=it + 1, winrate=round(wr, 3),
                  games=a.games, decisions=len(samples), league=len(league))
        # sliding-window replay buffer: train on a decorrelated mix of RECENT iterations, not just
        # this iteration's freshly-generated (and highly self-correlated) games. AlphaZero/KataGo sample
        # minibatches from a large buffer for exactly this reason; training on one small fresh batch — and
        # especially for several epochs — overfits the net to that batch and degrades general strength.
        if a.replay_buffer:
            replay.extend(samples)
            if len(replay) > a.replay_buffer:
                replay = replay[-a.replay_buffer:]          # drop the oldest (most off-policy) decisions
            print(f"    replay buffer: {len(replay)} decisions "
                  f"(this iter +{len(samples)}, cap {a.replay_buffer})", flush=True)
            train_set = replay
        else:
            train_set = samples
        train_on(net, train_set, train_dev, a.epochs, a.bs, a.lr, a.vcoef, a.log)
        # GATELESS (AlphaZero/KataGo-style): the continuously-trained net IS the output. Persist it every
        # iteration to BOTH the resume checkpoint (latest_out) and the published output (a.out), so a 12h
        # kill never loses progress and whatever loads rl_model.pt gets the latest training. We accept the
        # per-iter noise (no promotion gate): the replay buffer decorrelates updates and the field-eval
        # early-stop below still catches sustained degradation.
        torch.save(net.state_dict(), latest_out)
        torch.save(net.state_dict(), a.out)
        # byproduct harvest: dump this iteration's samples as value-head training data (free).
        if a.dump_samples:
            os.makedirs(a.dump_samples, exist_ok=True)
            import pickle
            with open(os.path.join(a.dump_samples, f"{_SESS}_iter{it+1:03d}.pkl"), "wb") as f:
                pickle.dump(samples, f)

        # opponent diversity: periodically fold the current net into the league (a deepcopy, NO arena games
        # unlike the old gate). Self-play then mixes in older snapshots as opponents (fictitious play).
        if a.league_every and (it + 1) % a.league_every == 0:
            league.append(_snapshot(net, dev))
            league[:] = league[-a.league_size:]
            print(f"  league snapshot -> {len(league)} members (every {a.league_every} iters)")
        # per-iteration field check vs the baseline (catch degradation early)
        detailed = (it + 1) % a.field_every == 0
        cand_field = arena.field_eval_agnostic(net, dev, db, atk, field_pool, a.field_games * 3,
                                               log=a.log, verbose=detailed,
                                               tag=f"iter{it+1} field eval")
        stats.log(a.log, event="rl_field", iter=it + 1, field_winrate=round(cand_field, 3),
                  baseline=round(baseline_field, 3))
        print(f"  field: {cand_field:.0%} (baseline {baseline_field:.0%})")
        if not a.no_early_stop:
            if cand_field < baseline_field - a.early_stop_margin:
                below += 1
                if below >= a.early_stop_patience:
                    print(f"\nEARLY STOP: field win-rate below baseline for {a.early_stop_patience} iters running "
                          f"({cand_field:.0%} < {baseline_field:.0%}). The RL signal is degrading "
                          f"the net. NOTE: gateless mode means {a.out}/{latest_out} hold the LATEST "
                          f"(now-regressed) net — resume from an earlier value-data dump if needed. "
                          f"This is the trigger to move to ISMCTS (see docs/roadmap.md).")
                    stats.log(a.log, event="rl_early_stop", iter=it + 1,
                              field_winrate=round(cand_field, 3), baseline=round(baseline_field, 3))
                    break
            else:
                below = 0
    print(f"done. output checkpoint (latest trained net): {a.out}  |  resume from: {latest_out}")


if __name__ == "__main__":
    main()
