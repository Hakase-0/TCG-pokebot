"""
stats_ui.py — a tiny terminal dashboard for the JSONL logs.

Renders BC training curves (loss, top-1 match) and eval win-rates as unicode
sparklines. No dependencies. One-shot, or --watch to refresh live.

  python stats_ui.py                          # default logs/
  python stats_ui.py --watch
  python stats_ui.py --train logs/train.jsonl --eval logs/eval.jsonl
"""
from __future__ import annotations
import argparse, os, time
import stats

BLOCKS = "▁▂▃▄▅▆▇█"


def spark(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(BLOCKS[min(7, int((v - lo) / rng * 7))] for v in vals)


def render(train_path, eval_path):
    out = []
    out.append("┌─ PTCG agent · training & eval ─────────────────────────────┐")

    tr = [r for r in stats.read(train_path) if r.get("event") == "bc_epoch"]
    if tr:
        loss = [r.get("loss") for r in tr]
        acc = [r.get("top1_match") for r in tr]
        out.append(f"│ BC epochs: {len(tr):<3}                                            │")
        out.append(f"│   loss      {spark(loss):<28} {loss[-1]:.3f} (was {loss[0]:.3f}) │")
        out.append(f"│   top1-match{spark(acc):<28} {acc[-1]:.3f}            │")
    else:
        out.append("│ BC: no training epochs logged yet                          │")

    sp = [r for r in stats.read(train_path.replace('train', 'selfplay'))
          if r.get("event") == "selfplay"]
    if sp:
        out.append(f"│ self-play data: {sp[-1].get('total_samples',0)} samples "
                   f"over {sp[-1].get('games',0)} games            │")

    ev = [r for r in stats.read(eval_path) if r.get("event") == "eval"]
    if ev:
        wr = [r.get("p0_winrate") for r in ev]
        out.append(f"│ eval runs: {len(ev):<3}                                            │")
        out.append(f"│   p0 winrate{spark(wr):<28} {wr[-1]:.0%}             │")
        last = ev[-1]
        out.append(f"│   last: {last.get('completed')}/{last.get('games')} clean, "
                   f"engine={last.get('engine')}                       │")
    else:
        out.append("│ eval: no runs logged yet (run_game.py --log logs/eval.jsonl)│")

    out.append("└────────────────────────────────────────────────────────────┘")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="logs/train.jsonl")
    ap.add_argument("--eval", default="logs/eval.jsonl")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()
    if not a.watch:
        print(render(a.train, a.eval))
        return
    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(render(a.train, a.eval))
            print("\n(watching — Ctrl-C to stop)")
            time.sleep(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
