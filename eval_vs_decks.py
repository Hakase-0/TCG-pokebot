"""
eval_vs_decks.py — measure our agent against a pool of varied opponent decks.

Our side is main.agent (heuristic + engine-oracle combat) on our deck.csv. Each
opponent deck (engine .csv produced by import_deck.py) is piloted by the plain
heuristic so it plays its own list competently. Seats alternate to cancel
first-player bias. Reports per-matchup and overall win rate, logs JSONL.

Usage:
  python eval_vs_decks.py --decks decks/ --games 20
"""
from __future__ import annotations
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from cg import game
import main, policy_heuristic as H, stats
from evaluate import CardDB


def _load():
    db = CardDB.load("capability_table.json")
    atk = {int(k): v for k, v in json.load(open("attack_table.json")).items()}
    return db, atk


def _read_deck(path):
    return [int(x) for x in open(path).read().split()][:60]


def play(our_deck, opp_deck, our_seat, db, atk, max_steps=4000):
    """One game; returns True if our agent wins."""
    main._TRACKER = None
    decks = [None, None]
    decks[our_seat] = our_deck
    decks[1 - our_seat] = opp_deck
    obs, _ = game.battle_start(decks[0], decks[1])
    steps = 0
    while steps < max_steps:
        cur = obs.get("current") or {}
        if cur.get("result", -1) >= 0:
            break
        sel = obs.get("select")
        if sel is None:
            yi = cur.get("yourIndex", 0)
            obs = game.battle_select(decks[yi]); steps += 1; continue
        yi = cur.get("yourIndex", 0)
        if yi == our_seat:
            choice = main.agent(obs)                        # us: heuristic + combat
        else:
            choice = H.select(obs, db=db, attack_db=atk)    # opponent: plain heuristic
        obs = game.battle_select(choice); steps += 1
    res = (obs.get("current") or {}).get("result", -1)
    game.battle_finish()
    return res == our_seat


def evaluate(decks_dir, games, log):
    db, atk = _load()
    our_deck = _read_deck("deck.csv")
    files = sorted(f for f in glob.glob(os.path.join(decks_dir, "*.csv")))
    if not files:
        print(f"no opponent decks in {decks_dir} (run import_deck.py first)"); return
    print(f"our deck vs {len(files)} opponent decks, {games} games each\n")
    overall_w = overall_n = 0
    rows = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        opp = _read_deck(f)
        if opp == our_deck:
            continue
        w = 0
        for i in range(games):
            w += play(our_deck, opp, our_seat=i % 2, db=db, atk=atk)
        wr = w / games
        overall_w += w; overall_n += games
        rows.append((name, w, games, wr))
        print(f"  vs {name:<22} {w:>3}/{games}  {wr:5.0%}")
        stats.log(log, event="matchup", opponent=name, games=games, wins=w, winrate=round(wr, 3))
    if overall_n:
        print(f"\n  overall {overall_w}/{overall_n} = {overall_w/overall_n:.0%}")
        stats.log(log, event="eval_summary", games=overall_n, wins=overall_w,
                  winrate=round(overall_w / overall_n, 3), n_decks=len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="decks/")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--log", default="logs/matchups.jsonl")
    a = ap.parse_args()
    evaluate(a.decks, a.games, a.log)
