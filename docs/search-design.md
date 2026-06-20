# Search & imperfect-information design notes

How we handle hidden information in search, why we chose what we chose, and the
risk-ordered roadmap. This is the reasoning behind `combat.py`, `deck_inference.py`,
and the planned upgrades — captured before building so we don't drift.

## The landscape (and the correct names)

PTCG is a large hidden-information game (opponent hand, deck contents, and deck
order are unknown). Options, roughly in increasing principled-ness and cost:

- **ISMCTS + determinization** — sample concrete hidden states (determinizations),
  run MCTS, aggregate. The distinction worth getting right: vanilla *determinized
  MCTS* builds a separate tree per sampled world and averages; **Information Set
  MCTS (ISMCTS)** builds ONE tree over information sets and threads a different
  determinization through it each iteration, sharing statistics across worlds.
  ISMCTS is the standard tool for large-hidden-state card games (Magic,
  Hearthstone research, bridge, skat). It scales to huge hidden state and
  sidesteps explicit belief machinery.
  - **Known weakness: strategy fusion.** Within each sampled world the searcher
    has perfect information, so it implicitly assumes it will *know* which world
    it's in at future decisions — effectively letting it choose a different plan
    per world. That produces overconfident lines that rely on information we
    won't actually have. This is the specific pathology to watch for in our
    agent's mistakes (and the thing the belief-aware ideas below target).

- **ReBeL / Player of Games (the principled lineage).** ReBeL (Brown 2020) does
  search+RL over *public belief states* with a CFR solver inside search, and
  provably converges to Nash in 2p zero-sum — but it's poker-shaped (small, clean
  hidden state). **Player of Games (DeepMind 2021)** generalizes the same
  search+RL+CFR ideas (Growing-Tree CFR) across perfect- and imperfect-info games
  (chess, Go, poker, Scotland Yard) and is the better reference point for a game
  like ours. The blocker for us is the same either way: representing/solving over
  the belief distribution of an unknown 60-card deck + hand + order is intractable
  to do fully.

- **POMDP / model-free deep RL with a recurrent (belief-approximating) policy.**
  Skip explicit search; let a recurrent net learn to summarize hidden state from
  observation history and train by self-play (AlphaStar, OpenAI Five). Proven to
  scale to messy hidden-info games. A real alternative axis, heavier on compute
  and self-play infrastructure, lighter on search engineering.

## What we're doing, and why

**Target: ISMCTS with determinization, guided by our policy/value net.** It's the
empirically dominant approach for card games, it's tractable, and it reuses
exactly what we already built — the engine forward model (`search_begin/step`)
and the determinizer (`combat.py` + `deck_inference.py`). The current `combat.py`
is a degenerate case (single determinization, flat one-ply look); ISMCTS is the
real version.

We are **not** building full ReBeL/PoG belief search: the belief state is
intractable here, and its payoff (unexploitability over many adversarial games)
isn't what a high-variance bot ladder rewards. We need to win more games than the
field, not compute an un-exploitable equilibrium.

## ReBeL-flavored ideas we *will* consider (cheap borrows, not the full machine)

The portable insight from the ReBeL/PoG lineage is: *position value depends on the
belief distribution, and search should reason over that distribution rather than a
single sampled world.* Tractable shadows of that, in order of value/risk:

1. **Belief-conditioned value head.** Feed the value net a summary of the belief —
   the opponent-archetype posterior `deck_inference` already produces — alongside
   the determinization it's scoring. The net learns "how good is this position
   *given this belief about the opponent*." Cheap, reuses existing parts, and is
   how `deck_inference` improvements actually reach the NET (not just the search).
2. **Belief-weighted determinization.** Sample opponent worlds from
   `deck_inference`'s posterior (not uniformly), and let the search aggregate
   re-weight which worlds matter. Directly attacks strategy fusion: reinforce the
   move that survives across the *believable* worlds.
3. **Root-only regret.** Full CFR-in-search is the expensive PoG piece. A bounded
   borrow: run a lightweight regret update over the handful of root actions across
   sampled worlds, so the committed move is robust to which world is real, with
   plain search below the root. Only if the arena shows fusion-type blunders that
   (1)–(2) didn't fix.

Framing honestly: this is "ISMCTS with belief-conditioned value and (maybe) root
regret," a sensible literature-adjacent combination — not a claimed breakthrough.
Anything here must be proven by `arena.py`, not assumed.

## Where deck_inference fits

`deck_inference` is the belief model. Its quality and search depth are
**multiplicative, not additive**: a good opponent model with shallow search is
wasted; deep search over a bad opponent model reasons confidently about the wrong
worlds (strategy fusion). They must improve together. Today most of combat's
*measured* value comes from `find_lethal` and 1-ply attack-vs-pass, which barely
use the opponent model — so improving `deck_inference` in isolation, against the
current shallow search, yields little. Its value is *unlocked* by ISMCTS (worlds
sampled from the belief) and by the belief-conditioned value head.

## Risk-ordered roadmap

Each step happens only if the arena says the previous one helped.

1. **Measured RL baseline** — does the self-play flywheel turn at all, vs
   heuristic+combat. (We do not yet know this.)
2. **ISMCTS** — the biggest proven lever: deeper lookahead, value-head leaf
   evaluation, and a far cleaner visit-count training target.
3. **Belief-conditioned value head + belief-weighted determinization** — the
   defensible ReBeL-flavored upgrade; also the path that makes `deck_inference`
   improvements pay off. Pair with expanding the archetype library to the real
   field (cheap, helps training/eval/inference at once).
4. **Root regret** — only if the arena shows residual strategy-fusion blunders.

Novelty is a multiplier on a working, measured system — never a substitute for
validating the simple thing first.
