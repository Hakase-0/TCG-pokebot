#!/usr/bin/env bash
# pack_for_kaggle.sh — zip the code for upload as a Kaggle DATASET.
# Use this because the GitHub repo is private (Kaggle can't git-clone it).
#
# Workflow each time you change code:
#   1. ./pack_for_kaggle.sh                      # makes tcg-pokebot-code.zip
#   2. Kaggle -> your dataset -> "New Version" -> upload the zip   (first time: New Dataset)
#   3. In the notebook the input auto-updates on next run; re-run cell 1.
#
# Excludes the competition engine/data (gitignored) and local artifacts.
set -e
OUT="tcg-pokebot-code.zip"
rm -f "$OUT"
zip -r "$OUT" . \
  -x '.git/*' '*/__pycache__/*' '__pycache__/*' '*.pyc' \
     'data/*' 'logs/*' 'replays/*' '*.pt' 'model_meta.json' \
     'cg/*' '*.so' '*.dll' '*.pdf' '*_Card_Data.csv' 'sample_submission/*' \
     'capability_table.json' 'attack_table.json' 'gotchas.csv' \
     '*.bundle' 'submission/*' 'submission.tar.gz' >/dev/null
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)) — upload as a Kaggle dataset (new version each time)"
echo "includes: $(unzip -l "$OUT" | grep -c '\.py$') python files, $(unzip -l "$OUT" | grep -c 'decks/.*\.txt') decklists"
