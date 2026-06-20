# PTCG AI Battle — agent scaffold

A crash-proof, card-aware Pokémon TCG agent for the cabt engine, with the
network skeleton for the imitation/RL phases. Built against the official API
(https://matsuoinstitute.github.io/cabt/).

## Status — validated against the real engine
The competition engine (`cg/`, `libcg.so`) runs in-container. Verified:
- Self-play games complete cleanly (no crashes, no infinite loops).
- Win rates on the live engine: heuristic **75%** vs random; **+ combat oracle
  → 90%** vs random and **66%** vs the plain heuristic (combat changes the
  turn-ending decision on ~24% of those decisions).
- `capability_table.json` (1267 cards) + `attack_table.json` (1556 attacks) are
  generated from the real `all_card_data()` / `all_attack()`.
- `submission.tar.gz` (~1.1 MB) packages and runs a full game standalone.
- `deck.csv` is the official sample meta deck — a real, legal 60 to tune later.

The real pool is **1267 cards** (121 ex, 30 Mega ex, 32 Tera, 29 ACE SPEC), and
**673 of them have text-driven damage/effects** — which is why `combat.py` uses
the engine as a damage oracle (via `search_begin`/`search_step`) rather than
trusting printed `Attack.damage`.

## Files
| file | role |
|---|---|
| `main.py` | **submission entrypoint.** `agent(obs_dict)` — deck handling, legality normalization, END-preferring fallback, optional timeout. Loads the tables + (optional) NN. Never raises. |
| `policy_heuristic.py` | card-aware policy: real move order (value/draw first, energy attached last), declines bad optional actions (over-benching liabilities, optional discards), quality targeting. |
| `evaluate.py` | prize-trade position evaluator + `CardDB` (reads `capability_table.json`). Value in [-1,1]; also the NN value-head target. |
| `combat.py` | **engine oracle.** `find_lethal`, `should_attack`, `refine` — simulates actions via `search_begin`/`search_step` and scores results with `evaluate_state`. Upgrades the heuristic's turn-ending decision; safely no-ops if the engine isn't importable. |
| `deck_inference.py` | opponent model: `OpponentTracker` (reveal accumulation by serial), `ArchetypeLibrary` (match revealed cards to known decklists), `predict_opponent_zones` (counts-matched hidden zones for `search_begin`). Feeds combat's 2-ply and gust targeting. |
| `gen_selfplay_data.py` | builds a behavioral-cloning dataset by self-play (`--policy combat` to distill the strong agent). Same format ingests ladder replays later. |
| `train_bc.py` | behavioral cloning of the pointer net (masked cross-entropy over legal options); writes `model.pt` + `model_meta.json`; MPS/CPU auto. |
| `stats.py` / `stats_ui.py` | JSONL metric logging + a dependency-free terminal dashboard (`python stats_ui.py --watch`) with loss/accuracy/win-rate sparklines. |
| `import_deck.py` | imports a LimitlessTCG decklist (`<count> <name> <SET>-<num>`) into an engine 60-card deck: basic-energy-by-type, exact-printing, then name fallback; reports substitutions/unmatched. |
| `eval_vs_decks.py` | runs our agent vs a pool of imported opponent decks (`decks/*.csv`), reporting per-matchup and overall win rates. |
| `kaggle_train_pokebot.ipynb` | end-to-end Kaggle (Linux) notebook: locate engine → tables → import decks → self-play data → train BC → eval → package. |
| `decks/*.txt` | LimitlessTCG decklists (POR-rotation top meta) used as the varied opponent pool. |
| `features.py` | `obs_dict` → numpy tensors for the network. |
| `model.py` | pointer/option-scoring policy+value net (PyTorch, ~0.74M params). |
| `inspect_cards.py` | profiles the whole card pool; writes `capability_table.json`, `attack_table.json`, `gotchas.csv`. |
| `run_game.py` | drives a battle with our agent (real engine or `--mock`), logs every decision, checks legality. |
| `mock_engine.py` | scripted stand-in implementing the battle protocol (for `run_game.py --mock`). |
| `build_submission.sh` | packs everything + the engine `cg/` into `submission.tar.gz`. |
| `deck.csv` | 60 card IDs, one per line. **Placeholder — replace with the real list.** |
| `selftest.py`, `test_behaviors.py` | tests; no engine needed. |

## Get it working (once you have the engine)
1. Download the starter materials from the Kaggle competition page; note the
   path to the engine's `cg/` folder. `pip install kaggle-environments==1.30.1`.
2. **Generate the knowledge tables** (one-time, needs the engine):
   ```
   CG_LIB_PATH=/path/to/cg python inspect_cards.py
   ```
   Writes `capability_table.json` + `attack_table.json` (the policy is much
   stronger with them) and `gotchas.csv` (skim with your TCG teammate).
3. **Put a real 60-card list in `deck.csv`** (teammate's call).
4. **Watch it play** a self-play game with a readable decision log:
   ```
   CG_LIB_PATH=/path/to/cg python run_game.py
   ```
5. **Package & submit:**
   ```
   CG_LIB_PATH=/path/to/cg ./build_submission.sh
   kaggle competitions submit pokemon-tcg-ai-battle -f submission.tar.gz -m "heuristic v1"
   ```

## Verify without the engine (works right now)
```
python evaluate.py          # evaluator + Tera-bench edge case
python test_behaviors.py    # move order, declining actions, targeting
python selftest.py          # harness contract + (if torch) NN act path
python run_game.py --mock   # full battle loop end-to-end on the mock engine
```

## Contract (do not break)
- Deck selection: `obs["select"] is None` → return 60 card IDs.
- Otherwise return option indices, length in `[minCount, maxCount]`.
- Never raise; respect the per-move time limit. `main.py` enforces all three.

## Phases
1. **Heuristic + engine oracle (done).** Tables + deck + submit. `POLICY=heuristic`
   (default) now includes `combat.py` (lethal, 2-ply attack-vs-pass with
   `deck_inference` opponent prediction, gust/damage targeting).
2. **Behavioral cloning (pipeline ready).** Generate data, train, ship the net:
   ```
   python gen_selfplay_data.py --games 500 --policy combat --out data/bc.pkl
   python train_bc.py --data data/bc.pkl --epochs 20 --out model.pt
   POLICY=nn python run_game.py --log logs/eval.jsonl     # A/B the net
   python stats_ui.py --watch                              # live curves
   ```
   Distills the strong `combat` agent now; swap in ladder replays when available.
3. **League self-play (RL).** Warm-start from BC; gate on ladder Elo.

## Training/eval at a glance
`stats.py` writes JSONL to `logs/`; `stats_ui.py` renders loss, top-1 match, and
win-rate as terminal sparklines. Everything runs locally in the bundled engine —
no GPU required (the net is ~0.74M params; MPS on Apple Silicon is plenty).

## Testing vs varied real decks (LimitlessTCG, TEF–POR)
The engine pool spans exactly the **TEF–POR Standard** format. Pull a top decklist
from `play.limitlesstcg.com` with the `set=POR` rotation filter, paste the card
lines into `decks/<name>.txt`, then:
```
python import_deck.py decks/dragapult.txt --csv EN_Card_Data.csv --out decks/dragapult.csv
python eval_vs_decks.py --decks decks/ --games 30
```
Measured (our agent vs heuristic-piloted imports): **92% vs Dragapult, 83% vs
Lucario Hariyama**. The importer handles printing differences (it maps Boss's
Orders → the legal PAL printing, etc.) and flags anything outside the pool
(e.g. post-POR CRI cards).

## Training on Kaggle (native Linux)
Open `kaggle_train_pokebot.ipynb` on Kaggle, *Add Input* → the competition (engine
+ card data), enable Internet, and run top to bottom: it generates tables, imports
the decks, runs self-play data-gen with the combat agent, trains the BC net, evals
vs the deck pool, and writes `model.pt` + `submission.tar.gz`. Faster than Docker
on a Mac since Kaggle is x86-64 Linux.

## Known upgrade hooks
`combat.py` now closes lethal detection, 2-ply attack-vs-pass (gated on a
confident `deck_inference` read), and gust/damage targeting. Remaining: grow the
`ArchetypeLibrary` from real replay decklists so opponent prediction is accurate
in non-mirror games, and move from BC distillation to self-play RL.
