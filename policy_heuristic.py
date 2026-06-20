"""
policy_heuristic.py — a card-aware policy that plays like a person thinks.

Key ideas (vs the v1 "do the first available action, then attack"):

  * MOVE ORDER. Within a turn the engine asks us for a MAIN action repeatedly.
    We answer in the order a player would: take free information/value first
    (draw/search abilities & supporters), then evolve, then develop the board,
    then attach Energy NEAR-LAST (it's an irreversible commitment), then decide
    whether to attack or pass.

  * WHETHER TO ACT AT ALL. Many options are optional. We decline the ones that
    hurt: don't bench a multi-prize Pokemon we don't need (it just becomes a
    gust target on the opponent's prize map); don't take optional discards of
    good cards; only gust when there's a worthwhile target; only blow up hands
    (Iono) when we're ahead on the prize race.

  * QUALITY when choosing cards. Discards dump the least valuable cards; search
    grabs what our plan lacks; damage targets the most valuable / closest-to-KO
    Pokemon; promotions pick the best attacker.

Decisions that genuinely need lookahead (is THIS attack a good prize trade?
which gust target sets up lethal?) are marked `# ENGINE:` — they're where
combat.py / search will replace the local heuristic. Everything is wrapped so
it always returns a legal selection and never raises.

Signature: select(obs_dict, db=None, attack_db=None) -> list[int]
`db` is an evaluate.CardDB; `attack_db` maps attackId -> {damage, energies}.
Both optional; the policy degrades to structural defaults without them.
"""

from __future__ import annotations

try:
    from evaluate import CardDB, can_afford, prizes_left
except Exception:                       # keep importable in isolation
    CardDB = None
    def can_afford(a, b): return True
    def prizes_left(ps): return len((ps or {}).get("prize", []) or [])

# OptionType ids
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = 0, 1, 2, 3, 4, 5, 6
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, SKILL, SPECIAL = 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
# SelectType ids
ST_MAIN, ST_CARD, ST_ATTACHED, ST_CARD_OR_ATTACHED, ST_ENERGY, ST_SKILL, ST_ATTACK, ST_EVOLVE, ST_COUNT, ST_YES_NO, ST_SPECIAL = range(11)
# AreaType ids
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH = 1, 2, 3, 4, 5
# SelectContext ids (subset we special-case)
CTX_MAIN = 0
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE = 1, 2, 3, 4
CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM = 8, 9, 10
CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY, CTX_DAMAGE = 13, 14, 15
CTX_REMOVE_COUNTER, CTX_HEAL = 16, 17
CTX_DRAW_COUNT = 38
CTX_IS_FIRST, CTX_MULLIGAN, CTX_ACTIVATE, CTX_FIRST_EFFECT, CTX_COIN_HEAD = 41, 42, 43, 44, 46


# ---------------------------------------------------------------- context ----
class Ctx:
    def __init__(self, obs, db, attack_db):
        self.sel = obs.get("select") or {}
        self.state = obs.get("current") or {}
        self.db = db
        self.attack_db = attack_db or {}
        self.options = self.sel.get("option", []) or []
        self.stype = self.sel.get("type")
        self.context = self.sel.get("context")
        self.lo = int(self.sel.get("minCount", 1) or 0)
        self.hi = max(int(self.sel.get("maxCount", 1) or 1), self.lo)
        self.yi = self.state.get("yourIndex", 0) or 0
        players = self.state.get("players") or [{}, {}]
        self.me = players[self.yi] if len(players) > self.yi else {}
        self.opp = players[1 - self.yi] if len(players) > 1 else {}

    # resolve which card id an option points at, so we can judge its value
    def card_id(self, opt):
        if opt.get("cardId"):
            return opt["cardId"]
        area, idx, pi = opt.get("area"), opt.get("index"), opt.get("playerIndex", self.yi)
        ps = self.me if pi == self.yi else self.opp
        try:
            if opt.get("type") == PLAY:
                return (self.me.get("hand") or [])[opt["index"]].get("id")
            if area == A_HAND:
                return (ps.get("hand") or [])[idx].get("id")
            if area == A_DECK:
                return (self.sel.get("deck") or [])[idx].get("id")
            if area == A_DISCARD:
                return (ps.get("discard") or [])[idx].get("id")
            if area == A_ACTIVE:
                a = ps.get("active") or []
                return a[0].get("id") if a else None
            if area == A_BENCH:
                return (ps.get("bench") or [])[idx].get("id")
        except Exception:
            return None
        return None

    def in_play_pokemon(self, opt):
        """For ATTACH/EVOLVE/targeting: the target Pokemon on the field."""
        area = opt.get("inPlayArea", opt.get("area"))
        idx = opt.get("inPlayIndex", opt.get("index"))
        pi = opt.get("playerIndex", self.yi)
        ps = self.me if pi == self.yi else self.opp
        try:
            if area == A_ACTIVE:
                a = ps.get("active") or []
                return a[0] if a else None
            if area == A_BENCH:
                return (ps.get("bench") or [])[idx]
        except Exception:
            return None
        return None


# ------------------------------------------------------------ card valuation -
def card_value(cid, db):
    """Rough keep-value for discard/search decisions (higher = keep)."""
    if db is None or cid is None:
        return 1.0
    if db.ex(cid):
        return 3.0
    for tag, val in (("draw_search", 2.5), ("gust_switch", 2.2),
                     ("energy_accel", 2.0), ("heal", 1.5)):
        if db.has_tag(cid, tag):
            return val
    return 1.0


def attack_damage(opt, attack_db):
    a = attack_db.get(opt.get("attackId"))
    if not a:
        return 0
    return a.get("damage", 0) if isinstance(a, dict) else getattr(a, "damage", 0)


# ----------------------------------------------------------------- handlers --
def _yes_no(ctx):
    yes = next((i for i, o in enumerate(ctx.options) if o.get("type") == YES), None)
    no = next((i for i, o in enumerate(ctx.options) if o.get("type") == NO), None)
    c = ctx.context
    if c == CTX_MULLIGAN:                      # keep a workable opening hand
        return [no if no is not None else 0]
    if c == CTX_IS_FIRST:                      # going first secures tempo/setup
        return [yes if yes is not None else 0]
    if c in (CTX_ACTIVATE, CTX_FIRST_EFFECT):
        # take the effect only if its source card is a value engine (draw/accel)
        src = ctx.sel.get("contextCard") or {}
        cid = src.get("id") if isinstance(src, dict) else None
        good = ctx.db and (ctx.db.has_tag(cid, "draw_search") or ctx.db.has_tag(cid, "energy_accel"))
        if good or ctx.db is None:
            return [yes if yes is not None else 0]
        return [no if no is not None else (yes if yes is not None else 0)]
    return [yes if yes is not None else 0]     # incl. COIN_HEAD and misc


def _choose_main(ctx):
    opts = ctx.options
    by = lambda t: [i for i, o in enumerate(opts) if o.get("type") == t]

    # 1) free value: draw/search/accel abilities BEFORE committing resources
    for i in by(ABILITY):
        cid = ctx.card_id(opts[i])
        if ctx.db is None or ctx.db.has_tag(cid, "draw_search") or ctx.db.has_tag(cid, "energy_accel"):
            return [i]

    # 2) evolve (advances the board, clears conditions on the evolving Pokemon)
    if by(EVOLVE):
        return [by(EVOLVE)[0]]

    # 3) play from hand — DECIDE per card, don't blindly dump the hand
    best_play = None
    for i in by(PLAY):
        cid = ctx.card_id(opts[i])
        if ctx.db and ctx.db.has_tag(cid, "draw_search"):
            return [i]                       # draw/search supporter: play early
        if ctx.db and ctx.db.has_tag(cid, "gust_switch"):
            if ctx.opp.get("bench"):
                return [i]                   # ENGINE: pick the gust target that enables lethal
            continue                         # no target -> hold it
        if ctx.db and ctx.db.has_tag(cid, "hand_disruption"):
            if prizes_left(ctx.me) < prizes_left(ctx.opp):
                return [i]                   # Iono only while ahead on prizes
            continue
        if ctx.db and ctx.db.ex(cid) and not ctx.db.tera(cid):
            n_inplay = (1 if (ctx.me.get("active") or []) else 0) + len(ctx.me.get("bench") or [])
            if n_inplay >= 2:
                continue                     # don't bench an unnecessary 2-prize liability
        if best_play is None:
            best_play = i
    if best_play is not None:
        return [best_play]

    # 4) attach Energy — near-last, onto the attacker we intend to use
    attach = by(ATTACH)
    if attach and not ctx.state.get("energyAttached"):
        active_pk = (ctx.me.get("active") or [None])[0]
        active_attach = None
        for i in attach:
            if active_pk is not None and ctx.in_play_pokemon(opts[i]) is active_pk:
                active_attach = i
                break
        return [active_attach if active_attach is not None else attach[0]]

    # 5) (retreat handled implicitly; ENGINE: compare matchup/cost vs bench)

    # 6) attack or pass
    attacks = by(ATTACK)
    if attacks:
        best = _pick_attack(ctx, attacks)
        if best is not None and _should_attack(ctx, opts[best]):
            return [best]
    end = by(END)
    if end:
        return [end[0]]
    return [0]


def _pick_attack(ctx, attack_idxs):
    opp_active = (ctx.opp.get("active") or [None])[0]
    opp_hp = (opp_active or {}).get("hp", 9999) if opp_active else 9999
    best, best_dmg = None, -1
    for i in attack_idxs:
        dmg = attack_damage(ctx.options[i], ctx.attack_db)
        if dmg >= opp_hp and opp_hp > 0:     # likely KO -> take it
            return i
        if dmg > best_dmg:
            best, best_dmg = i, dmg
    return best


def _should_attack(ctx, opt):
    """
    Baseline: if an attack is available, take it (it's the normal turn-ender,
    and lethal is already prioritized in _pick_attack). Declining to attack in
    favor of developing is a genuine prize-trade judgement that needs lookahead.
    ENGINE: replace this with one-ply search comparing evaluate_state after
    attacking vs after passing/developing; until then, attacking is the default.
    """
    return True


def _hp_of(ctx, opt):
    pk = ctx.in_play_pokemon(opt)
    return pk.get("hp", 9999) if isinstance(pk, dict) and pk else 9999


def _take_between(ctx, ranked):
    n = ctx.hi if ctx.lo == ctx.hi else max(ctx.lo, min(1, ctx.hi))
    n = max(ctx.lo, min(n, ctx.hi, len(ranked)))
    return sorted(ranked[:n]) if n > 0 else []


def _choose_cards(ctx):
    opts = ctx.options
    c = ctx.context

    if c == CTX_SETUP_ACTIVE:                 # low-liability starter that can act
        ranked = sorted(range(len(opts)),
                        key=lambda i: (ctx.db.ex(ctx.card_id(opts[i])) if ctx.db else False,
                                       -card_value(ctx.card_id(opts[i]), ctx.db)))
        return [ranked[0]]
    if c == CTX_SETUP_BENCH:
        ranked = sorted(range(len(opts)),
                        key=lambda i: -card_value(ctx.card_id(opts[i]), ctx.db))
        return _take_between(ctx, ranked)

    if c in (CTX_SWITCH, CTX_TO_ACTIVE):      # promote our best attacker
        ranked = sorted(range(len(opts)),
                        key=lambda i: -card_value(ctx.card_id(opts[i]), ctx.db))
        return [ranked[0]]

    if c in (CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM):   # dump least valuable
        n = ctx.lo                            # decline optional discards of good cards
        if n <= 0:
            return []
        ranked = sorted(range(len(opts)),
                        key=lambda i: card_value(ctx.card_id(opts[i]), ctx.db))
        return sorted(ranked[:n])

    if c in (CTX_DAMAGE, CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY):  # hit value/soft
        ranked = sorted(range(len(opts)),
                        key=lambda i: (-(ctx.db.prize_value(ctx.card_id(opts[i])) if ctx.db else 1),
                                       _hp_of(ctx, opts[i])))
        return _take_between(ctx, ranked)

    if c in (CTX_HEAL, CTX_REMOVE_COUNTER):   # protect our key attacker
        ranked = sorted(range(len(opts)),
                        key=lambda i: -(ctx.db.prize_value(ctx.card_id(opts[i])) if ctx.db else 1))
        return _take_between(ctx, ranked)

    # default search/select: grab the highest-value cards our plan wants
    ranked = sorted(range(len(opts)),
                    key=lambda i: -card_value(ctx.card_id(opts[i]), ctx.db))
    return _take_between(ctx, ranked)


def _choose_count(ctx):
    nums = [(i, o.get("number", 0) or 0) for i, o in enumerate(ctx.options)]
    if ctx.context == CTX_DRAW_COUNT:         # don't draw yourself out of deck
        deck = ctx.me.get("deckCount", 60) or 60
        safe = [p for p in nums if p[1] <= max(deck - 1, 0)]
        best = max(safe or nums, key=lambda p: p[1])
        return [best[0]]
    return [max(nums, key=lambda p: p[1])[0]]


# ------------------------------------------------------------------ dispatch -
def select(obs_dict, db=None, attack_db=None) -> list[int]:
    ctx = Ctx(obs_dict, db, attack_db)
    if not ctx.options:
        return []
    try:
        if ctx.stype == ST_YES_NO:
            out = _yes_no(ctx)
        elif ctx.stype == ST_MAIN or ctx.context == CTX_MAIN:
            out = _choose_main(ctx)
        elif ctx.stype == ST_COUNT:
            out = _choose_count(ctx)
        elif ctx.stype == ST_ATTACK:
            best = _pick_attack(ctx, list(range(len(ctx.options))))
            out = [best if best is not None else 0]
        else:
            out = _choose_cards(ctx)
        # guarantee count legality
        out = [i for i in out if 0 <= i < len(ctx.options)]
        if len(out) < ctx.lo:
            extra = [i for i in range(len(ctx.options)) if i not in out]
            out = sorted(out + extra[:ctx.lo - len(out)])
        return out[:ctx.hi]
    except Exception:
        return list(range(ctx.lo)) if ctx.lo > 0 else ([0] if ctx.options else [])
