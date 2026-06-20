# Running locally

The competition engine is **Linux/Windows x86-64 only** (`cg/libcg.so`, `cg/cg.dll`)
— there is no macOS build, so on an Apple Silicon Mac the engine runs in a Linux
container, not natively.

## 1. macOS venv — code, offline tests, and NN training (native, uses MPS)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# these need no engine:
python selftest.py
python test_behaviors.py
python run_game.py --mock
python train_bc.py --data data/bc.pkl --epochs 20   # if you have a dataset
python stats_ui.py --watch
```

## 2. Run the ENGINE via Docker (Linux x86-64) on the Mac
Put the Kaggle `cg/` engine folder and the generated `*_table.json` in the
project root first (both are git-ignored). Then:
```bash
docker build --platform linux/amd64 -t tcg-pokebot .
# generate tables from the engine (one-time):
docker run --rm --platform linux/amd64 -v "$PWD":/app tcg-pokebot python inspect_cards.py
# watch a real game:
docker run --rm --platform linux/amd64 -v "$PWD":/app tcg-pokebot python run_game.py
# generate BC data with the strong agent, then train:
docker run --rm --platform linux/amd64 -v "$PWD":/app tcg-pokebot \
    python gen_selfplay_data.py --games 500 --policy combat --out data/bc.pkl
python train_bc.py --data data/bc.pkl --epochs 20   # train natively on the Mac (MPS)
```

## 3. No-Docker alternative: Kaggle Notebooks
Kaggle is Linux x86-64, so the engine loads there directly — good for heavier
self-play/data-gen runs.
