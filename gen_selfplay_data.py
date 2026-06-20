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
import argparse, json, pickle, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from cg import game
import features, main, policy_heuristic as H, stats
from evaluate import CardDB


def _load():
    deck = [int(x) for x in open("deck.csv").read().split()][:60]
    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in json.load(open("attack_table.json")).items()}
    return deck, db, atk


def _policy(name, deck, db, atk):
    if name == "combat":
        return main.agent                      # heuristic + engine-oracle refinement
    return lambda o: (H.select(o, db, atk) if o.get("select") is not None else deck)


def collect(n_games, out, log, policy="heuristic", max_steps=4000):
    deck, db, atk = _load()
    pol = _policy(policy, deck, db, atk)
    samples = []
    for g in range(n_games):
        main._TRACKER = None                   # reset per-game opponent inference
        obs, _ = game.battle_start(deck, deck)
        steps = 0
        while steps < max_steps:
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                break
            sel = obs.get("select")
            if sel is not None:
                enc = features.encode_observation(obs, attack_lookup=atk)
                choice = pol(obs)
                if enc is not None and choice:
                    enc["target"] = list(choice)
                    samples.append(enc)
            else:
                choice = deck
            obs = game.battle_select(choice)
            steps += 1
        game.battle_finish()
        if (g + 1) % 10 == 0 or g + 1 == n_games:
            stats.log(log, event="selfplay", policy=policy,
                      games=g + 1, total_samples=len(samples))
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(samples, f)
    print(f"collected {len(samples)} decision samples over {n_games} games -> {out}")
    return len(samples)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--policy", choices=["heuristic", "combat"], default="heuristic")
    ap.add_argument("--out", default="data/bc.pkl")
    ap.add_argument("--log", default="logs/selfplay.jsonl")
    a = ap.parse_args()
    collect(a.games, a.out, a.log, a.policy)
