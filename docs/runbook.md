# Setup runbook — from a fresh engine to a measured RL run

The goal of this document: take someone from "just cloned the repo" to a first
**measured self-play RL run** (`selfplay_rl.py --search flat`), and tell them
exactly when to flip on ISMCTS (`--search ismcts`). Every command below is the
real CLI of the tool it names.

## The one hard constraint

The competition engine (`cg/`, `libcg.so`/`cg.dll`) is **Linux/Windows x86-64
only — there is no macOS build.** Anything that imports the engine
(`inspect_cards.py`, `gen_selfplay_data.py`, `run_game.py`, `arena.py`,
`ismcts.py`, and `selfplay_rl.py` — it does `from cg import game` at import time)
**cannot run natively on a Mac.** Run those in one of:

- **Docker (`linux/amd64`) on the Mac** — full control, slower (emulated). Good
  for tables + short runs.
- **Kaggle notebook** — native x86-64 Linux, has internet (needed for
  `build_deck_pool.py`), best for heavy self-play. See
  `kaggle_train_pokebot.ipynb`.

Pure-PyTorch steps (`train_bc.py`, `fit_value.py`, `stats_ui.py`) and the offline
tests **do** run natively on the Mac (MPS) — see `DEV.md`.

---

## Step 0 — environment

Pick **one** of:

**Docker path (Mac).** Build the image once, then run everything through it:
```bash
docker build --platform linux/amd64 -t tcg-pokebot .
# convenience: every "engine" command below is  $RUN <cmd>
RUN='docker run --rm --platform linux/amd64 -v '"$PWD"':/app tcg-pokebot'
```

**Kaggle path.** Open `kaggle_train_pokebot.ipynb`, *Add Input* → the competition
(engine + card data), enable Internet, and run top-to-bottom — it performs Steps
1–7 below. The rest of this runbook is the explicit version of that notebook.

---

## Step 1 — get the competition IP (not in git)

From the Kaggle competition page (`pokemon-tcg-ai-battle`), download into the
project root:

| File / folder | What it is |
|---|---|
| `cg/` | the engine (`libcg.so` etc.) |
| `EN_Card_Data.csv` (`*_Card_Data.csv`) | card database for deck import |

All of these are **git-ignored** (`.gitignore`) — they are competition IP, do not
commit or redistribute them. Then install the harness (already in the Docker
image; on Kaggle add it, natively use `requirements.txt`):
```bash
pip install kaggle-environments==1.30.1
```

---

## Step 2 — generate the knowledge tables (one-time, needs the engine)

```bash
$RUN python inspect_cards.py        # Docker
# (Kaggle:  python inspect_cards.py)
```
Writes `capability_table.json` (1267 cards) + `attack_table.json` (1556 attacks)
+ `gotchas.csv`. The policy and the net are markedly stronger with these; they
degrade to safe defaults without them. **Verify:**
```bash
ls -l capability_table.json attack_table.json
```

---

## Step 3 — build the opponent deck pool (`decks/`)

The repo ships 30 LimitlessTCG decklists as `decks/*.txt`; training needs them as
engine decks (`decks/*.csv`, also git-ignored). `import_deck.py` reads only the
card-data CSV (**no engine needed**), so this step runs natively too.

**Offline (use the shipped `.txt`):**
```bash
for f in decks/*.txt; do
  python import_deck.py "$f" --csv EN_Card_Data.csv --out "${f%.txt}.csv"
done
```

**Or refresh from Limitless (needs internet — Kaggle/Docker):**
```bash
$RUN python build_deck_pool.py --top 30 --csv EN_Card_Data.csv   # re-fetches + imports
```

**Verify** (expect ~30; some lists may be skipped if they include post-POR cards
outside the engine pool):
```bash
ls decks/*.csv | wc -l
```

**Optional — adversary (off-meta) decks** for robustness eval. `selfplay_rl.py`
reads them from `decks/adversary/*.csv` and uses them for `--adversary-frac` of
self-play games; the run works fine with none.

---

## Step 4 — our deck (`deck.csv`)

`deck.csv` holds our 60-card list. All readers — `selfplay_rl.py`,
`import_deck.py`, and `main.py` (the submission) — parse it with `.read().split()`,
so **whitespace- or newline-separated IDs both work**. The file must contain
exactly 60 IDs; otherwise the agent falls back to a `[1]*60` placeholder deck.

---

## Step 5 — (recommended) behavioral-cloning warm start

RL is much more stable warm-started from a BC net than from scratch (the loop
prints a warning and trains from random weights otherwise).

```bash
# data-gen needs the engine (heavy — run in the container / on Kaggle):
$RUN python gen_selfplay_data.py --games 500 --policy combat --out data/bc.pkl
# training is pure PyTorch — run it natively on the Mac (MPS) if you prefer:
python train_bc.py --data data/bc.pkl --epochs 20 --out model.pt
```
Produces `model.pt` + `model_meta.json`. (Better still, clone from real ladder
replays with `ingest_replays.py` when you have them — see `README.md`.)

---

## Step 6 — first RL run: **Step 1 of the roadmap (flat search)**

This is the measured baseline — "does the self-play flywheel turn at all?" Start
with a tiny smoke test, then the real run.

```bash
# smoke test (a few minutes): confirms the loop completes end-to-end
$RUN python selfplay_rl.py --search flat --iters 1 --games 4 \
    --warm model.pt --our-deck deck.csv --opp-decks decks/ --out rl_model.pt

# the actual Step-1 run
$RUN python selfplay_rl.py --search flat --iters 5 --games 40 \
    --warm model.pt --our-deck deck.csv --opp-decks decks/ --out rl_model.pt
```

What to watch (the run logs to `logs/rl.jsonl`):
```bash
python stats_ui.py --watch        # loss / win-rate sparklines (runs natively)
```
The loop prints, every iteration: self-play win-rate, the **CI-gated promotion**
result (candidate vs. current best — promotes only if it clears the threshold
*and* the Wilson lower bound beats a coin flip), and a **field-eval** win-rate
against the deck pool with the **BC baseline** to beat. The best gated checkpoint
is `rl_model.pt`; `rl_model.latest.pt` is the crash-safe continuously-trained net.

---

## Step 7 — read the gate, then enable ISMCTS (roadmap Step 2)

The program tells you when to escalate. Two signals from Step 6:

- **Flywheel turns** (gate Elo trends up across iterations; field win-rate ≥ the
  BC baseline) → the simple loop works; ISMCTS should make it climb *faster*.
- **Flat / regressing** → the loop early-stops with: *"The RL signal is degrading
  the net … This is the trigger to move to ISMCTS."* A flat result means the
  search/target is the bottleneck — which **is** the case ISMCTS addresses.

Either way, flip the search on (warm-start from the flat checkpoint):
```bash
$RUN python selfplay_rl.py --search ismcts \
    --ismcts-worlds 6 --ismcts-sims 32 --leaf rollout \
    --iters 30 --games 400 \
    --warm rl_model.pt --our-deck deck.csv --opp-decks decks/ --out rl_model.pt
```
- **Cost:** ISMCTS is the compute driver — each searched option is an engine
  rollout (`worlds × sims` per decision). Scale `--ismcts-worlds/--ismcts-sims`
  and `--games` up on a cloud box, not on the emulated Mac. `--leaf value` (net
  value head) is cheaper than `--leaf rollout` (engine-oracle rollout, a stronger
  compass while the value head is weak).
- **Optional symmetric self-play:** add `--opp-search ismcts` so the opponent
  also searches (stronger sparring, more honest value targets, ~2× cost).
- **Win check / kill** (from `docs/roadmap.md`): keep ISMCTS only if the
  MCTS-wrapped net beats heuristic+combat in the arena and the flywheel improves
  faster than Step 1; if it's slower with no Elo gain after tuning sim count,
  revisit determinization / the value head.

Standalone arena check anytime:
```bash
$RUN python arena.py --candidate rl_model.pt --anchor model.pt --games 200 --field
```

---

## Step 8 — package & submit

```bash
# heuristic+combat submission (no net):
CG_LIB_PATH=cg ./build_submission.sh        # -> submission.tar.gz
# for a learned agent, ship model.pt/rl_model.pt too and set POLICY=nn (see main.py)
kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "rl v1"
```
To iterate code on Kaggle without git access, `./pack_for_kaggle.sh` zips the
code for upload as a Kaggle dataset.

---

## Artifact / environment quick reference

| Artifact | Produced by | Needs engine? | Where to run |
|---|---|---|---|
| `cg/`, `EN_Card_Data.csv` | download from Kaggle | — | — |
| `capability_table.json`, `attack_table.json` | `inspect_cards.py` | **yes** | Docker / Kaggle |
| `decks/*.csv` | `import_deck.py` / `build_deck_pool.py` | no (build_deck_pool needs internet) | anywhere / Kaggle |
| `model.pt` (BC) | `gen_selfplay_data.py` → `train_bc.py` | gen: **yes**; train: no | Docker(gen) + Mac(train) |
| `rl_model.pt` (RL) | `selfplay_rl.py` | **yes** | Docker / Kaggle |
| `submission.tar.gz` | `build_submission.sh` | no | anywhere |
