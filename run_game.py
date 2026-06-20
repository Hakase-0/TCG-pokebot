"""
run_game.py — drive a cabt battle with our agent and watch it play.

Works against the REAL engine or the mock, with the same loop:
    python run_game.py --mock            # end-to-end protocol check, no engine
    CG_LIB_PATH=/path/to/cg python run_game.py          # real engine, low-level
    CG_LIB_PATH=/path/to/cg python run_game.py --kaggle  # via kaggle_environments

The real engine's `game` module exposes battle_start / battle_select /
battle_finish (see https://matsuoinstitute.github.io/cabt/game.html); the loop
calls our agent (main.agent) for whichever player must select, validates the
returned indices are legal, logs the decision, and stops at a result.
"""

from __future__ import annotations
import argparse, os, sys

import main as AGENT   # main.agent is seat-agnostic (reads yourIndex from obs)

# human-readable names for logging
SELECT_TYPE = {0: "MAIN", 1: "CARD", 2: "ATTACHED", 3: "CARD/ATT", 4: "ENERGY",
               5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT", 9: "YES_NO",
               10: "SPECIAL"}
OPTION_TYPE = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL", 5: "ENERGY_CARD",
               6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
               11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END", 15: "SKILL",
               16: "SPECIAL"}


def get_engine(use_mock):
    if not use_mock:
        for p in (os.environ.get("CG_LIB_PATH"), ".", "..", "./sample_submission"):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        try:
            from cg import game           # engine ships as a `cg` package
            return game, "real"
        except Exception:
            try:
                import game
                return game, "real"
            except Exception as e:
                print(f"[!] real engine not importable ({e}); using --mock", file=sys.stderr)
    import mock_engine
    return mock_engine, "mock"


def load_deck():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    with open(path) as f:
        return [int(x) for x in f if x.strip()]


def describe(sel):
    st = SELECT_TYPE.get(sel.get("type"), sel.get("type"))
    opts = sel.get("option", []) or []
    kinds = ",".join(OPTION_TYPE.get(o.get("type"), "?") for o in opts[:8])
    return f"{st}/ctx{sel.get('context')} [{len(opts)} opts: {kinds}] " \
           f"min{sel.get('minCount')} max{sel.get('maxCount')}"


def run_one(engine, deck, verbose=True, max_steps=2000):
    obs, start = engine.battle_start(deck, deck)
    if obs is None:
        print(f"[X] battle_start failed: {start}")
        return None
    steps = 0
    while steps < max_steps:
        cur = obs.get("current") or {}
        if cur.get("result", -1) is not None and cur.get("result", -1) >= 0:
            break
        sel = obs.get("select")
        if sel is None:                      # deck-selection style turn (kaggle wrapper)
            choice = AGENT.agent(obs)        # returns the 60-card deck
        else:
            choice = AGENT.agent(obs)
            n = len(sel.get("option", []))
            assert all(0 <= i < n for i in choice), f"ILLEGAL selection {choice}"
            if verbose:
                who = cur.get("yourIndex", "?")
                print(f"  t{cur.get('turn','?')} P{who}: {describe(sel)} -> {choice}")
        obs = engine.battle_select(choice)
        steps += 1
    result = (obs.get("current") or {}).get("result", -1)
    engine.battle_finish()
    winner = {0: "Player 0", 1: "Player 1", 2: "Draw"}.get(result, "unfinished")
    print(f"[=] game over in {steps} decisions -> {winner}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="use the mock engine")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--log", default=None, help="append per-game results as JSONL")
    args = ap.parse_args()

    engine, kind = get_engine(args.mock)
    deck = load_deck()
    print(f"engine={kind}  deck={len(deck)} cards  games={args.games}\n")
    results = [run_one(engine, deck, verbose=not args.quiet) for _ in range(args.games)]
    ok = sum(1 for r in results if r is not None and r >= 0)
    p0 = sum(1 for r in results if r == 0)
    print(f"\ncompleted {ok}/{args.games} games cleanly  (P0 won {p0})")
    if args.log:
        try:
            import stats
            stats.log(args.log, event="eval", engine=kind, games=args.games,
                      completed=ok, p0_wins=p0, p0_winrate=round(p0 / max(args.games, 1), 3))
        except Exception:
            pass
    if kind == "mock" and ok == args.games:
        print("run_game.py (mock): protocol loop OK")


if __name__ == "__main__":
    main()
