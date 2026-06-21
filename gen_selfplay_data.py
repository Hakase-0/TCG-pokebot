"""
gen_selfplay_data.py — build a behavioral-cloning dataset by self-play.

Each decision becomes a training sample: the encoded observation plus the option
index/indices the policy chose. Distilling the strong `combat` agent into the
fast pointer net is the goal; use --policy heuristic for quick/cheap data.
The SAME sample format ingests real ladder replays later (swap the data source).

Usage:
  python gen_selfplay_data.py --games 200 --policy combat --out data/bc.pkl
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from cg import game
import features, policy_heuristic as H, stats, glob, random
from evaluate import CardDB


def _load_pool(decks_dir="decks"):
    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in json.load(open("attack_table.json")).items()}
    pool = []
    for f in sorted(glob.glob(os.path.join(decks_dir, "*.csv"))):
        d = [int(x) for x in open(f).read().split()][:60]
        if len(d) == 60:
            pool.append((os.path.basename(f)[:-4], d))
    return pool, db, atk


def collect(n_games, out, log, policy="heuristic", max_steps=4000, decks_dir="decks"):
    """Agnostic BC: each game samples BOTH decks from the pool and records the teacher's
    decisions piloting whichever deck it was dealt — so the net learns to pilot the field,
    not one deck. policy='combat' uses the engine-oracle teacher (recommended)."""
    import arena
    pool, db, atk = _load_pool(decks_dir)
    if not pool:
        raise SystemExit(f"no decks/*.csv in {decks_dir} — build the pool first")
    use_combat = (policy == "combat")
    samples = []
    for g in range(n_games):
        (_, da), (_, dbk) = random.choice(pool), random.choice(pool)
        decks = {0: da, 1: dbk}
        pols = {0: arena.make_heuristic_agent(db, atk, da, use_combat=use_combat)(),
                1: arena.make_heuristic_agent(db, atk, dbk, use_combat=use_combat)()}
        obs, _ = game.battle_start(da, dbk)
        steps = 0
        while steps < max_steps:
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                break
            sel = obs.get("select")
            yi = cur.get("yourIndex", 0)
            if sel is None:
                choice = decks[yi]
            else:
                enc = features.encode_observation(obs, attack_lookup=atk)
                choice = pols[yi](obs)
                if enc is not None and choice and isinstance(choice, list):
                    enc["target"] = list(choice)
                    samples.append(enc)
            obs = game.battle_select(choice)
            steps += 1
        game.battle_finish()
        if (g + 1) % 10 == 0 or g + 1 == n_games:
            print(f"  BC data: {g+1}/{n_games} games, {len(samples)} samples", flush=True)
            stats.log(log, event="selfplay", policy=policy, games=g + 1, total_samples=len(samples))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    print(f"collected {len(samples)} decision samples over {n_games} agnostic games -> {out}")
    return len(samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--policy", choices=["heuristic", "combat"], default="heuristic")
    ap.add_argument("--out", default="data/bc.pkl")
    ap.add_argument("--log", default="logs/selfplay.jsonl")
    a = ap.parse_args()
    collect(a.games, a.out, a.log, a.policy)
