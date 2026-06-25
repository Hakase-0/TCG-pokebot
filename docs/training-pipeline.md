# Training pipeline — how the agnostic agent actually learns

This is the **mechanics** doc: what each stage does, which file/function runs it,
the real hyperparameters the Kaggle notebook passes, and what gets logged. It is
deliberately descriptive, not aspirational —
[`docs/roadmap.md`](roadmap.md) is the measurement-gated list of what to build
*next*, and [`docs/search-design.md`](search-design.md) is the reasoning behind
the imperfect-information search. This doc explains the loop that exists today and
then maps every design choice to the modern game-playing-AI literature, with
sources, so a reader can see exactly which ideas we adopted, which we simplified,
and where we deliberately diverge.

The whole thing is **deck-agnostic**: one `PointerPolicyValueNet` is trained to
pilot the *entire* deck pool (both seats are sampled from it), so card identity
enters only through a shared embedding and the net never memorizes one list.

---

## 0. The substrate

- **Engine.** `cg/` is the native PTCG battle engine (Linux-only, gitignored). It
  is the ground-truth simulator: legal options, state transitions, win/loss. The
  whole pipeline is "model-based" in the trivial sense that we own a perfect
  forward model and call it directly.
- **Card tables.** `capability_table.json` + `attack_table.json` (gitignored) feed
  the heuristic teacher and the feature encoder.
- **Deck pool.** `decks/*.csv` is the meta field (each a legal 60); `decks/adversary/*.csv`
  is the off-meta bucket. Self-play and evaluation both sample from these.
- **The net (`model.py`).** A *pointer* policy-value net: board entities → tokens
  (static features ⊕ card embedding) → Transformer set-encoder → masked-mean board
  vector; a globals MLP; then **each legal option is scored** by an MLP over
  `[option feats ⊕ option-card embed ⊕ board ⊕ globals]` → one logit per option →
  softmax over the legal set (the policy). A separate value head puts a sigmoid over
  `[board ⊕ globals]` → win probability. The notebook trains it at `d=256, 4 heads,
  3 layers` (~3.0M params).
- **Feature encoder (`features.py`).** `encode_observation` turns an engine
  observation into the entity/option/global tensors the net consumes. The *same*
  encoding is used for BC and RL, so a sample is portable between them.

---

## 1. Behavioral-cloning bootstrap (first run only)

**Why:** self-play from a random net wastes most of a 12h kernel flailing. We seed
the net by cloning a competent teacher, then let RL take over.

- **Teacher / data** — `gen_selfplay_data.py --policy combat`. The `combat` teacher
  is the engine-oracle heuristic (`arena.make_heuristic_agent(..., use_combat=True)`):
  it does shallow combat simulation to pick moves. Both seats are sampled from the
  pool every game, so the recorded decisions span the field, not one deck. Each
  decision is saved as `(encoded observation, chosen option index/indices)`.
  Notebook: **150 games** → `data/bc.pkl`. Logs `selfplay {games, total_samples}`
  to `logs/sp.jsonl`.
- **Cloning** — `train_bc.py`. Masked cross-entropy over exactly the legal options
  present at each decision (imitate the teacher's choice). Notebook: **15 epochs**,
  `d=256/4/3`, Adam. Logs `bc_epoch {loss, top1_match}` per epoch to `logs/tr.jsonl`,
  and writes `model.pt` + `model_meta.json`.
- **Output:** `model.pt`, the warm-start for RL. (On a resumed session BC is
  skipped and RL warms from the recovered checkpoint instead.)

This is **distillation of a fixed teacher** — it gives the net a sane prior but
cannot exceed the teacher. RL is what pushes past it.

---

## 2. Self-play RL — Expert Iteration (the core loop)

`selfplay_rl.py`. This is **Expert Iteration (ExIt)**: a slow "expert" (tree
search) produces a better policy than the raw net at each state; the net is trained
to imitate that improved policy; the now-stronger net makes the *next* search
better; repeat. AlphaZero is the most famous instance of this pattern.

The notebook runs it as
`--search ismcts --leaf value --ismcts-worlds 3 --ismcts-sims 48 --games 60
--league-every 2 --replay-buffer 30000 --replay-preload valuedata
--time-budget-min 420 --dump-samples valuedata`.

### 2a. The expert: ISMCTS over determinized worlds

`expert()` → `ismcts.search()`. PTCG is **imperfect information** (opponent hand,
deck contents and order are hidden), so the search can't run on the raw state. For
each of `n_worlds` determinizations it calls `deck_inference.predict_opponent_zones`
to fill the hidden opponent zones with a *plausible* concrete deck/hand/prizes
(seeded by what's been revealed), forks the engine at that world, and runs MCTS
with PUCT: prior = the net's policy, value = the net's value head at the leaf
(`leaf="value"`), opponent nodes minimizing our win probability. Visit counts are
**aggregated across worlds** into one improved policy target, and the search also
returns a low-variance **search value** `sval` for that state.

- **Played vs. recorded move.** The clean aggregated target is what *trains* the
  net. The move actually *played* adds AlphaZero-style exploration: Dirichlet root
  noise + temperature, sampled for the first `greedy_after=8` of our moves, greedy
  thereafter (explore early, exploit late).
- **The opponent.** Each game pairs the learner against a **league snapshot**
  (§2d). By default the opponent plays fast (`opp-search=raw` → net argmax) while
  only the learner's seat runs ISMCTS; seats alternate. `opp-search=ismcts` makes
  it symmetric search-vs-search at ~2× cost.
- **Fallbacks.** If the engine/search is unavailable, `expert` degrades to a 1-ply
  `combat` rollout over the net's top-k, then to the raw net prior — the loop never
  hard-fails on a worker without the engine.

### 2b. Labeling

`play_game()` plays to terminal, then assigns the game outcome `z ∈ {1, 0.5, 0}`
(win/draw/loss) to **every** decision in that game. Each training sample is
`[encoded state, improved policy target, z, search value sval]`.

### 2c. The replay buffer

A single iteration's ~thousands of decisions are highly self-correlated; training
several epochs on them overfits and *degrades* general strength (we measured this:
a 47% vs. 58%-baseline field drop early on). So we keep **1 epoch** and train on a
**sliding-window replay buffer** (`--replay-buffer 30000`): each iter appends its
samples and drops the oldest, and the minibatch is drawn from the whole window.
`--replay-preload valuedata` warms the buffer from *past sessions'* harvested
samples, so the window spans runs, not just this kernel.

### 2d. Training step

`train_on()`. Adam, `lr=1e-4`. Loss = **policy cross-entropy** to the search target
+ `vcoef` × **value MSE** to `z`. Logs `rl_train {policy_loss, value_loss, samples}`
to `logs/rl.jsonl`. (Note: `rl_train` carries no `iter` field — the metrics report
attaches it to the most recent `rl_iter`.)

### 2e. Gateless output + league

- **Gateless** (AlphaZero/KataGo-style): the continuously trained net *is* the
  output. Every iteration it's saved to both `rl_model.pt` (published) and
  `rl_model.latest.pt` (resume), so a 12h kill never loses progress. There is **no
  promotion arena gate**; the replay buffer decorrelates updates and the field-eval
  early-stop (below) catches sustained regression.
- **League** (AlphaStar-style fictitious self-play): every `--league-every 2` iters
  the current net is deep-copied into a league of past snapshots (cap
  `--league-size 5`), and self-play draws opponents from it. This is a *cheap*
  league — a copy, no gating games.

### 2f. Evaluation, early-stop, time budget

- **Baseline.** Before any RL, `arena.field_eval_agnostic` measures how the
  warm-started net pilots the whole field vs. the heuristic reference → logged once
  as `rl_baseline`. This is the bar to beat.
- **Per-iter field check.** After each iter, the net is re-evaluated on the field
  pool (meta + off-meta adversaries) → `field_summary` + `rl_field
  {field_winrate, baseline}`. Field eval pilots the **raw** net (no search), so it
  measures the *learned* policy honestly.
- **Early stop.** If field win-rate stays below `baseline − 0.10` for 3 iters
  running → stop (`rl_early_stop`). **Time budget:** `--time-budget-min 420`,
  checked between iters, stops cleanly before Kaggle's 720-min kill
  (`rl_time_budget_stop`), leaving room for sharpening + persistence.

### 2g. Free value-data harvest

`--dump-samples valuedata` writes each iteration's `(enc, target, z, sval)` samples
to `valuedata/` as a byproduct. These feed both the next session's replay preload
(§2c) and the value-head sharpen (§3).

---

## 3. Value-head sharpening

`fit_value.py` (notebook §9). Because the value head *is* the ISMCTS leaf, a sharper
value head is a direct search-quality win. This step **freezes the entire body and
trains only the value head** on the harvested `valuedata/`, regressing toward a
blend of the game outcome `z` and the search value `sval` (`--target-blend 0.5`,
8 epochs). It temperature-calibrates on a holdout (the temperature is baked into the
last layer) and **asserts the policy logits don't move** (drift ~0) — so it cannot
change how the net plays, only how well it evaluates. The sharpened net is copied
back onto `rl_model.pt` + `rl_model.latest.pt` so next session's leaf is better.

`fit_value` prints `samples: N (M with search value...)`, `holdout value MSE:
before -> after (temperature T=...)`, and `policy logits drift (must be ~0): ...`.
The notebook's sharpen cell **captures that stdout and parses it** into a `SHARPEN`
dict for the metrics report.

---

## 4. Persistence & resume across sessions

A run is one 12h kernel; training spans many.

- **Resume (§6).** A robust `find_ckpt()` searches the named weights dataset, the
  working dirs, and *any* attached input for the newest `*.pt`, recovers
  `model_meta.json` (architecture dims), and recovers `valuedata/` for the replay
  preload. A net-size change auto-cold-starts (won't load mismatched dims).
- **Persist (§11).** Weights + harvested value-data are pushed to the Kaggle
  dataset `hakase0/tcg-pokebot-weights` so the next kernel resumes mid-training.

Net effect: **weights and games both carry forward**, so the replay window and the
value-data span sessions, not just one kernel.

---

## 5. Reading the run — the metrics report (§10)

Added in notebook v2.3: one consolidated scoreboard read from the JSONL logs, so a
single run prints one clear picture instead of scattered prints.

- **BC bootstrap** — epochs, loss curve, top-1 match vs. teacher, teacher-data size
  (from `logs/tr.jsonl` + `logs/sp.jsonl`).
- **RL self-play** — a per-iteration table: self-play win-rate, decisions harvested,
  policy loss, value loss, **field win-rate and its delta vs. baseline**, league
  size; then best/final field win-rate, the loss trajectory, total decisions, and
  any early-stop/time-budget note (from `logs/rl.jsonl`).
- **Value-head sharpen** — holdout MSE before→after, temperature, policy drift, and
  sample count (from the parsed `SHARPEN` dict).

The **field win-rate vs. baseline** line is the one to watch: it's the honest "is RL
making the deployable (search-free) net stronger?" signal.

---

## 6. How this maps to modern game-playing AI

The pipeline is a deliberate composition of published ideas, adapted to a
hidden-information card game on a tight compute budget.

### Perfect-information lineage (what the loop's *shape* comes from)

- **Expert Iteration** — Anthony, Tian & Barber, *Thinking Fast and Slow with Deep
  Learning and Tree Search*, NeurIPS 2017 (arXiv:1705.08439). The exact pattern we
  run: tree search as a policy-improvement operator, net trained to imitate it. Our
  `expert()` is the "slow" system; the net is the "fast" one.
- **AlphaGo Zero / AlphaZero** — Silver et al., *Mastering the game of Go without
  human knowledge*, Nature 2017; *A general RL algorithm…through self-play*, Science
  2018 (arXiv:1712.01815). Source of: policy target = MCTS **visit distribution**,
  value target = **game outcome**, PUCT with a net prior, and **Dirichlet root
  noise + temperature** for exploration. We copy all four.
- **KataGo** — Wu, *Accelerating Self-Play Learning in Go*, 2019
  (arXiv:1902.10565). The efficiency playbook for small budgets, including training
  from a **sliding replay window** rather than only fresh games. Our fixed-size
  30k window is the same idea (KataGo *grows* its window with data; we keep it flat
  — see §7).
- **MuZero** — Schrittwieser et al., *Mastering Atari, Go, chess and shogi by
  planning with a learned model*, Nature 2020 (arXiv:1911.08265). The contrast that
  defines us: MuZero *learns* a latent dynamics model because it lacks a simulator;
  **we own the engine**, so we plan in the true model and spend that saved capacity
  on the hidden-information problem instead.
- **AlphaStar** — Vinyals et al., *Grandmaster level in StarCraft II…*, Nature 2019.
  Source of the **league**: train against a population of past snapshots, not just
  the latest self, to avoid strategic cycling/forgetting. Ours is a stripped-down
  league (uniform snapshots, no prioritized fictitious self-play or
  main/exploiter roles — see §7).
- **OpenAI Five** — Berner et al., *Dota 2 with Large Scale Deep RL*, 2019
  (arXiv:1912.06680). Precedent for mixing current-vs-past opponents in self-play
  (their 80/20 latest/past split) to keep games stable but non-stationary.

### Imperfect-information lineage (the part that makes PTCG hard)

- **ISMCTS + determinization** — Cowling, Powley & Whitehouse, *Information Set
  Monte Carlo Tree Search*, IEEE TCIAIG 2012. The standard tool for large-hidden-
  state card games and the backbone of our `expert`: sample concrete worlds, search,
  aggregate visit counts over them.
- **Strategy fusion & non-locality** — Frank & Basin, *Search in games with
  incomplete information* (Bridge), Artificial Intelligence 1998; Long, Sturtevant,
  Buro & Furtak, *Understanding the Success of Perfect Information Monte Carlo
  Sampling*, AAAI 2010. These name the *known weakness* of determinized search: with
  perfect info inside each world the searcher implicitly assumes it will know which
  world it's in, producing overconfident lines. This is the pathology our
  belief-conditioning roadmap targets — see `docs/search-design.md`.
- **CFR family** — Zinkevich et al., *Regret Minimization in Games with Incomplete
  Information*, NeurIPS 2007; **DeepStack** (Moravčík et al., Science 2017),
  **Libratus** (Brown & Sandholm, Science 2018), **Pluribus** (Brown & Sandholm,
  Science 2019). The principled alternative to determinization — reason over belief
  distributions rather than sampled worlds — but it's poker-shaped (small, clean
  hidden state) and doesn't fit PTCG's huge structured hidden state off the shelf.
- **ReBeL / Player of Games / Student of Games** — Brown et al., *Combining Deep RL
  and Search for Imperfect-Information Games* (ReBeL), NeurIPS 2020
  (arXiv:2007.13544); Schmid et al., *Player of Games*, 2021 (arXiv:2112.03178);
  Schmid et al., *Student of Games*, Science Advances 2023. The modern unification
  of search + RL for imperfect-information games via **public belief states**. This
  is the "north star" lineage; we use cheap determinization now and treat
  belief-state search as a measured future step, not a starting point.
- **Perfect-information distillation / oracle guiding** — **Suphx** (Li et al.,
  *Mastering Mahjong with Deep RL*, 2020, arXiv:2003.13590) uses an oracle agent
  with hidden info to guide a deployable agent; **DouZero** (Zha et al., ICML 2021,
  arXiv:2106.06135) and **PerfectDou** (Yang et al., NeurIPS 2022,
  arXiv:2203.16406) distill perfect-information value into an imperfect-information
  policy for DouDizhu. We're a relative of this family: our **search value** `sval`
  (computed with determinized, effectively-revealed worlds) is a lower-variance
  target than the raw outcome, and the value-head sharpen (§3) is exactly a
  perfect-info→deployable-net distillation step. Our `deck_inference` predictor is
  the (soft, learned-from-reveals) stand-in for an oracle.

---

## 7. Where we deviate, and why

Honest accounting of the simplifications, so nobody mistakes the current loop for
the full method:

- **Gateless, not gated.** AlphaZero/KataGo publish the latest net without an arena
  promotion gate; so do we. We rely on the replay buffer + field-eval early-stop
  instead of per-iter gating games (those would burn ~80 arena games/iter). Risk:
  per-iter noise in the published net; mitigated by §2f.
- **Flat league, not PFSP.** AlphaStar's strength is *prioritized* fictitious
  self-play with main/exploiter agents. Ours is uniform snapshots — enough to fight
  forgetting on this budget, far short of AlphaStar's machinery.
- **Fixed replay window, not grown.** KataGo grows the window with dataset size; we
  hold 30k. Simpler, and our data volume per session is modest.
- **Determinization, not belief-state CFR.** We accept strategy fusion (§6) today
  for tractability. `deck_inference` softens it by sampling *plausible* worlds
  rather than uniform ones; principled belief-state search is a roadmap step gated
  on measurement (`docs/roadmap.md`, `docs/search-design.md`).
- **Asymmetric search by default.** Only the learner's seat runs ISMCTS; the league
  opponent plays raw net argmax (`opp-search=raw`) to keep self-play affordable.
  `opp-search=ismcts` makes it symmetric when the value leaf makes search cheap
  enough.
- **One agnostic net, not per-deck specialists.** A single net pilots the whole
  field by design; a deck-specialist net is a separate, later step.

---

## Sources

Recurrent-loop / perfect-information:
- Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree
  Search.* NeurIPS 2017. arXiv:1705.08439.
- Silver et al. *Mastering the game of Go without human knowledge.* Nature 2017.
- Silver et al. *A general reinforcement learning algorithm that masters chess,
  shogi and Go through self-play.* Science 2018. arXiv:1712.01815.
- Wu. *Accelerating Self-Play Learning in Go.* 2019. arXiv:1902.10565.
- Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning with a
  learned model* (MuZero). Nature 2020. arXiv:1911.08265.
- Vinyals et al. *Grandmaster level in StarCraft II using multi-agent reinforcement
  learning* (AlphaStar). Nature 2019.
- Berner et al. *Dota 2 with Large Scale Deep Reinforcement Learning* (OpenAI Five).
  2019. arXiv:1912.06680.

Imperfect-information:
- Cowling, Powley, Whitehouse. *Information Set Monte Carlo Tree Search.* IEEE
  Transactions on Computational Intelligence and AI in Games, 2012.
- Frank, Basin. *Search in games with incomplete information: a case study using
  Bridge card play.* Artificial Intelligence, 1998.
- Long, Sturtevant, Buro, Furtak. *Understanding the Success of Perfect Information
  Monte Carlo Sampling in Game Tree Search.* AAAI 2010.
- Zinkevich, Johanson, Bowling, Piccione. *Regret Minimization in Games with
  Incomplete Information* (CFR). NeurIPS 2007.
- Moravčík et al. *DeepStack: Expert-level artificial intelligence in heads-up
  no-limit poker.* Science 2017.
- Brown, Sandholm. *Superhuman AI for heads-up no-limit poker* (Libratus). Science
  2018. / *Superhuman AI for multiplayer poker* (Pluribus). Science 2019.
- Brown, Bakhtin, Lerer, Gong. *Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games* (ReBeL). NeurIPS 2020. arXiv:2007.13544.
- Schmid et al. *Player of Games.* 2021. arXiv:2112.03178.
- Schmid et al. *Student of Games: A unified learning algorithm for both perfect and
  imperfect information games.* Science Advances 2023.
- Li et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning.* 2020.
  arXiv:2003.13590.
- Zha et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement
  Learning.* ICML 2021. arXiv:2106.06135.
- Yang et al. *PerfectDou: Dominating DouDizhu with Perfect Information
  Distillation.* NeurIPS 2022. arXiv:2203.16406.
