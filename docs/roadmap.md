# Roadmap — gated by measurement

Nothing here is "optional" in the sense of "maybe skip it." Everything is
**conditional on the arena**: each step is built when its trigger fires and kept
only if it passes its check. Stacking unverified changes lowers the ceiling; this
list is how we add the highest-value thing next without flying blind.

Read: **Trigger** = build it when this is true. **Win check** = keep it only if
this holds. **Kill** = if this happens, revert and reconsider.

---

### Step 1 — Measured baseline  *(in progress)*
Behavioral cloning → self-play RL vs the 29-deck field → arena gate vs heuristic+combat.
- **Purpose:** does the self-play flywheel turn at all?
- **Win check:** gate Elo trends up across iterations; field-eval win-rate ≥ BC net.
- **Kill:** Elo regressing iteration over iteration → bug/instability; fix before anything else.
- **Reads:** `logs/rl.jsonl` (gate Elo), arena field-eval block.

### Step 2 — ISMCTS  *(next; biggest single lever)*
Replace the flat top-k determinized rollout with information-set MCTS over the engine,
guided by the net's policy prior + value head.
- **Trigger:** Step 1 shows the flywheel turns OR is flat (flat means the search/target
  is the bottleneck — that *is* the case for ISMCTS).
- **Win check:** MCTS-wrapped net beats heuristic+combat in the arena (the moment we
  have a submittable learned agent); RL flywheel improves faster than Step 1.
- **Kill:** slower with no Elo gain after tuning sim count → revisit determinization/value.
- **Cost:** more engine calls per move (compute). Most bug-prone piece — own tested pass.

### Step 3 — Belief-conditioned value + belief-weighted determinization
Feed the value head `deck_inference`'s archetype posterior; sample MCTS worlds from that
posterior instead of uniformly.
- **Trigger:** Step 2 shipped and stable.
- **Win check:** **hard-matchup** win-rates rise specifically (not just overall) — this
  is the strategy-fusion fix, so look at the worst matchups in field-eval and the
  ADVERSARY eval.
- **Kill:** no per-matchup improvement → belief signal too noisy; improve `deck_inference`
  coverage/calibration first.

### Step 4 — Root-only regret
Lightweight regret update over root actions across sampled worlds (committed move robust
to which world is real).
- **Trigger:** Step 3 done AND arena still shows fusion-type blunders (overcommitting on
  a read that was wrong).
- **Win check:** fewer such blunders; adversary/hard-matchup win-rates rise.
- **Kill:** no measurable change → drop it (it's targeted, not a broad lift).

### Step 5 — Two-net league (deck-specific agent + deck-agnostic sparring net)
A second, deck-agnostic net whose only job is to pilot the opponent pool well — never
submitted. Our agent trains against a self-improving, varied gauntlet.
- **Trigger:** field-eval / ladder shows the binding constraint is **opponent-piloting
  quality** (suspiciously high training win-rates that don't hold on the ladder; or we
  want a gauntlet stronger than the ladder before facing it).
- **Win check:** agent's ladder Elo rises after training vs the sparring net; robustness
  to off-meta (ADVERSARY eval) improves because the sparring net discovers degenerate
  lines for us.
- **Kill:** ~2× compute with no ladder gain → the constraint was elsewhere.
- **Note:** also the main defense against exploitation (the "always all-in" failure):
  a self-play sparring net invents degenerate strategies in training so the agent meets
  them prepared, not cold on the ladder.

---

## Always-on, free as we go
- **Ladder replays:** as games accumulate, fold winners' moves into the BC warm start
  (`ingest_replays.py --winners-only`) and use replay-driven opponents. The one input
  that improves just by playing.
- **Adversary eval every checkpoint:** `arena.py --adversary` reports off-meta matchups
  separately, so a "stronger" checkpoint that folds to a one-dimensional deck is caught
  *before* it ships. Self-play also trains vs these (`--adversary-frac`).
- **Deck choice stays with the human expert.** RL maximizes win-rate *within* the
  matchups our deck creates; it does not fix a deck with a fatal type/archetype hole.
  Field-eval per-matchup numbers tell the expert where the deck is exposed.

## Decision discipline
Add the next step only when its trigger fires; keep it only if its win check passes.
Novelty (Steps 3–5, belief/regret/league) is a multiplier on a measured, working
system — never a substitute for validating the simple thing first.
