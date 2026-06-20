#!/usr/bin/env bash
# build_submission.sh — package the agent for the Kaggle ladder.
#
# Produces submission.tar.gz containing main.py (entrypoint), our modules,
# the precomputed tables, deck.csv, and the engine's cg/ folder.
#
# Usage:
#   CG_LIB_PATH=/path/to/cg ./build_submission.sh
#
# Before submitting, generate the tables once against the real engine:
#   CG_LIB_PATH=/path/to/cg python inspect_cards.py    # writes *_table.json
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/submission"
rm -rf "$OUT" "$HERE/submission.tar.gz"
mkdir -p "$OUT"

# our code
for f in main.py policy_heuristic.py evaluate.py features.py combat.py deck.csv; do
  cp "$HERE/$f" "$OUT/"
done

# precomputed knowledge (optional but strongly recommended — the policy is much
# stronger with them; it degrades to safe defaults if absent)
for f in capability_table.json attack_table.json; do
  [ -f "$HERE/$f" ] && cp "$HERE/$f" "$OUT/" || echo "[!] $f missing — run inspect_cards.py against the engine first"
done

# trained network, if present (POLICY=nn). Heuristic-only submissions skip these.
for f in model.py model.pt model_meta.json; do
  [ -f "$HERE/$f" ] && cp "$HERE/$f" "$OUT/"
done

# the engine
if [ -n "${CG_LIB_PATH:-}" ] && [ -d "$CG_LIB_PATH" ]; then
  cp -r "$CG_LIB_PATH" "$OUT/cg"
else
  echo "[!] CG_LIB_PATH not set to the engine's cg/ folder — submission will lack the engine"
fi

# deck must be exactly 60 lines
LINES=$(grep -c . "$OUT/deck.csv" || true)
[ "$LINES" = "60" ] || { echo "[X] deck.csv has $LINES cards, must be 60"; exit 1; }

( cd "$OUT" && tar -czf "$HERE/submission.tar.gz" . )
echo "[=] wrote submission.tar.gz ($(du -h "$HERE/submission.tar.gz" | cut -f1))"
echo "    submit with: kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m 'msg'"
