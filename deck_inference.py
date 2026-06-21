"""
deck_inference.py — model the opponent's hidden cards.

The opponent's deck is hidden, but two things are not: the cards they reveal as
the game unfolds (discard, board, and ids that pass through visible zones), and
the rules of deckbuilding. v1 of this module matched the revealed cards against a
fixed library of catalogued archetypes. That recognizes only what we already
catalogued — it breaks on rogue decks, teched variants, and a shifting meta, and
a savvy opponent can bait a predictable matcher.

v2 replaces "recognize a known deck" with "*construct* a legal, plausible deck
that's consistent with what we've seen", from first principles:

  * CardIndex      — derives evolution lines, role buckets, and energy/type pools
                     straight from the engine's per-card data (no decklists).
  * DeckGenerator  — fills an archetype "recipe" of role slots outward from the
                     observed cards: completes evolution lines, matches energy to
                     the attacker's type, respects the 4-copy / 60-card / >=1-basic
                     rules, and never contradicts an observation. Deterministic for
                     the MAP guess; stochastic (per-rng) so each search world gets a
                     *different* plausible opponent.
  * ArchetypeLibrary (kept) — now an optional *prior*: when the revealed cards
                     clearly match a known list, it seeds the generator's template;
                     when they don't, the generator falls back to pure structure.

The public seam is unchanged: `predict_opponent_zones(...)`. combat.py calls it
for one MAP world; ISMCTS passes an `rng` per world for diverse determinizations.
The matcher and `library_from_pool` are preserved for backward compatibility.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import math, random

# log types that carry an opponent card id we can harvest
_LOG_CARD_TYPES = {6, 10, 11, 12}      # MOVE_CARD, PLAY, ATTACH, EVOLVE
POKEMON_CARDTYPE = 0
BASIC_ENERGY_CARDTYPE = 5
SPECIAL_ENERGY_CARDTYPE = 6
TRAINER_CARDTYPES = {1, 2, 3, 4}       # item / supporter / tool / stadium (subtype-agnostic here)
DECK_SIZE = 60
MAX_COPIES = 4                         # non-basic-energy deckbuilding cap


# ───────────────────────────── observation tracking ─────────────────────────

class OpponentTracker:
    """Accumulates the set of distinct opponent cards revealed during a game."""
    def __init__(self):
        self.seen = {}          # serial -> cardId
        self.last_turn = None

    def reset(self):
        self.seen = {}
        self.last_turn = None

    def _note(self, card):
        if not card:
            return
        s, cid = card.get("serial"), card.get("id")
        if s is not None and cid:
            self.seen[s] = cid

    def update(self, obs_dict):
        st = obs_dict.get("current") or {}
        turn = st.get("turn", 0) or 0
        if self.last_turn is not None and turn < self.last_turn:
            self.reset()                       # new game
        self.last_turn = turn

        yi = st.get("yourIndex", 0) or 0
        oi = 1 - yi
        players = st.get("players") or []
        if len(players) > oi:
            opp = players[oi]
            for c in (opp.get("discard") or []):
                self._note(c)
            for pk in ([(opp.get("active") or [None])[0]] + list(opp.get("bench") or [])):
                if not pk:
                    continue
                self._note(pk)
                for k in ("energyCards", "tools", "preEvolution"):
                    for c in (pk.get(k) or []):
                        self._note(c)
        for log in (obs_dict.get("logs") or []):
            if not isinstance(log, dict):
                continue
            if log.get("playerIndex") == oi and log.get("type") in _LOG_CARD_TYPES:
                self._note({"serial": log.get("serial"), "id": log.get("cardId")})
        return self

    def known(self) -> Counter:
        return Counter(self.seen.values())


# ───────────────────────────── card knowledge index ─────────────────────────

def _entries(card_db):
    """Yield (cid:int, entry:dict) over a CardDB-like object or a raw {id:entry} table."""
    t = getattr(card_db, "t", None)
    if t is None and isinstance(card_db, dict):
        t = card_db
    for k, v in (t or {}).items():
        try:
            yield int(k), v
        except (TypeError, ValueError):
            continue


class CardIndex:
    """
    Pre-computes everything the generator needs from per-card data — derived from
    the card rules, not from any decklist. Degrades gracefully if optional fields
    (stage flags, evolvesFrom, attacks) are absent from the table.
    """
    def __init__(self, card_db):
        self.db = card_db
        self.entry = {}
        self.name_to_id = {}
        self.pokemon, self.basic_pokemon = [], []
        self.basic_energy, self.special_energy, self.trainers = [], [], []
        self.attackers = []                       # pokemon that can attack
        self.energy_by_type = defaultdict(list)   # energyType -> [basic energy ids]
        self.pokemon_by_type = defaultdict(list)  # energyType -> [pokemon ids]
        self.role = defaultdict(list)             # role tag -> [card ids]
        self.evolves_from = {}                    # id -> id (lower stage), if resolvable
        self.evolves_to = defaultdict(list)       # id -> [higher stage ids]
        self.stage = {}                           # id -> 0/1/2

        for cid, e in _entries(card_db):
            self.entry[cid] = e
            nm = e.get("name")
            if nm and nm not in self.name_to_id:
                self.name_to_id[nm] = cid

        for cid, e in self.entry.items():
            ct = e.get("cardType")
            if ct == POKEMON_CARDTYPE:
                self.pokemon.append(cid)
                et = e.get("energyType")
                if et is not None:
                    self.pokemon_by_type[et].append(cid)
                if e.get("basic") or (not e.get("stage1") and not e.get("stage2")
                                      and not e.get("evolvesFrom")):
                    self.basic_pokemon.append(cid)
                    self.stage[cid] = 0
                elif e.get("stage1"):
                    self.stage[cid] = 1
                elif e.get("stage2"):
                    self.stage[cid] = 2
                if e.get("base_damages") or e.get("attacks"):
                    self.attackers.append(cid)
            elif ct == BASIC_ENERGY_CARDTYPE:
                self.basic_energy.append(cid)
                self.energy_by_type[e.get("energyType")].append(cid)
            elif ct == SPECIAL_ENERGY_CARDTYPE:
                self.special_energy.append(cid)
            elif ct in TRAINER_CARDTYPES:
                self.trainers.append(cid)
            # role buckets from capability tags (apply to whatever card carries them)
            for tag in (e.get("tags") or []):
                self.role[tag].append(cid)

        # resolve evolution lines by name (evolvesFrom is a name string)
        for cid, e in self.entry.items():
            pre = e.get("evolvesFrom")
            pid = e.get("evolvesFromId")
            if pid is None and pre:
                pid = self.name_to_id.get(pre)
            if pid is not None:
                self.evolves_from[cid] = pid
                self.evolves_to[pid].append(cid)

    # ---- small helpers the generator/zones logic use ----
    def is_pokemon(self, cid):       return self.entry.get(cid, {}).get("cardType") == POKEMON_CARDTYPE
    def is_basic_pokemon(self, cid): return cid in self.stage and self.stage[cid] == 0
    def is_basic_energy(self, cid):  return self.entry.get(cid, {}).get("cardType") == BASIC_ENERGY_CARDTYPE
    def energy_type(self, cid):      return self.entry.get(cid, {}).get("energyType")

    def evolution_chain_down(self, cid):
        """[cid, its pre, its pre-pre, ...] following evolvesFrom as far as known."""
        chain, cur, guard = [], cid, 0
        while cur is not None and guard < 4:
            chain.append(cur)
            cur = self.evolves_from.get(cur)
            guard += 1
        return chain

    def basic_energy_for(self, etype, default=None):
        opts = self.energy_by_type.get(etype) or self.basic_energy
        return opts[0] if opts else default


# ───────────────────────────── archetype matcher (prior) ────────────────────

class ArchetypeLibrary:
    """A small set of known 60-card archetypes — kept as an optional *prior*."""
    def __init__(self):
        self.decks = {}
        self._df = Counter()
        self._n = 0

    def fit(self, labeled_decklists):
        self.decks = {label: Counter(ids) for label, ids in labeled_decklists}
        self._n = len(self.decks)
        self._df = Counter()
        for deck in self.decks.values():
            for c in deck:
                self._df[c] += 1
        return self

    def _idf(self, c):
        return math.log(1 + self._n / (1 + self._df.get(c, 0)))

    def classify(self, known: Counter):
        if not self.decks or not known:
            return None, 0.0, {}
        raw = {}
        for label, deck in self.decks.items():
            raw[label] = sum(min(known[c], deck.get(c, 0)) * self._idf(c) for c in known)
        total = sum(raw.values())
        if total <= 0:
            return None, 0.0, {}
        posterior = {k: v / total for k, v in raw.items()}
        best = max(posterior, key=posterior.get)
        seen = sum(known.values())
        # confidence blends top-match dominance, its margin over 2nd, and how much we've seen
        ordered = sorted(posterior.values(), reverse=True)
        margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        conf = posterior[best] * (0.5 + 0.5 * margin) * min(1.0, seen / 8.0)
        return best, conf, posterior

    def predict_decklist(self, known: Counter):
        label, conf, _ = self.classify(known)
        if label is None:
            return None, 0.0
        base = Counter(self.decks[label])
        for c, n in known.items():
            base[c] = max(base.get(c, 0), n)
        cards = list(base.elements())
        return (cards[:DECK_SIZE] if len(cards) > DECK_SIZE else cards), conf


# ───────────────────────────── the deck generator (idea 1) ──────────────────

# A rough role recipe: target counts for the jobs every deck fills. The generator
# treats these as soft targets, adjusted by what we've already observed.
_RECIPE = [
    ("pokemon", 14),     # attacker lines + support pokemon (evolution-completed)
    ("draw_search", 12), # draw + ball/search engine
    ("gust_switch", 3),  # boss/gust + switching
    ("energy_accel", 2),
    ("trainer_other", 9),# stadiums, tools, tech supporters, recovery
    ("energy", 10),      # energy base (typed to the attacker)
]


class DeckGenerator:
    """
    Builds a legal, plausible 60 that is consistent with the observed cards.
    Deterministic when rng is None (MAP guess); stochastic per-rng otherwise so
    each search world gets a distinct determinization.
    """
    def __init__(self, index: CardIndex):
        self.ix = index

    def _pick(self, pool, rng, exclude_full, want):
        """Pick up to `want` ids from pool, respecting per-card copy caps."""
        out = []
        cand = [c for c in pool if exclude_full(c)]
        if not cand:
            return out
        if rng is not None:
            rng.shuffle(cand)
        i = 0
        while len(out) < want and cand:
            c = cand[i % len(cand)]
            if exclude_full(c):
                out.append(c)
            else:
                cand.remove(c)
                if not cand:
                    break
                continue
            i += 1
        return out

    def complete(self, known: Counter, rng=None, template: Counter | None = None):
        """Return Counter(id -> count) summing to DECK_SIZE, consistent with `known`."""
        ix = self.ix
        deck = Counter()

        def cap(cid):
            return DECK_SIZE if ix.is_basic_energy(cid) else MAX_COPIES

        def room(cid):
            return deck[cid] < cap(cid)

        # 1) seed with everything we've actually seen (never contradict observations)
        for cid, n in known.items():
            deck[cid] = min(max(n, deck[cid]), cap(cid))

        # 2) close evolution lines for any evolved pokemon we've committed to
        for cid in list(deck):
            if ix.stage.get(cid, 0) > 0:
                for lower in ix.evolution_chain_down(cid)[1:]:
                    if deck[lower] < deck[cid]:
                        deck[lower] = min(deck[cid], cap(lower))

        # 3) optional matcher template: pull in its cards as a soft prior
        if template:
            for cid, n in template.items():
                if sum(deck.values()) >= DECK_SIZE:
                    break
                if room(cid):
                    add = min(n - deck[cid], cap(cid) - deck[cid])
                    if add > 0:
                        deck[cid] += add

        # 4) infer the attacker's energy type(s) and ensure a matching energy base
        atk_types = Counter()
        for cid in deck:
            if ix.is_pokemon(cid) and (cid in ix.attackers):
                et = ix.energy_type(cid)
                if et is not None:
                    atk_types[et] += deck[cid]
        if not atk_types and known:                     # nothing committed yet
            atk_types[None] += 1

        # 5) fill the remaining role budget toward the recipe
        for role, target in _RECIPE:
            if sum(deck.values()) >= DECK_SIZE:
                break
            have = self._have_for_role(deck, role)
            want = max(0, target - have)
            if want <= 0:
                continue
            pool = self._pool_for_role(role, atk_types)
            budget = min(want, DECK_SIZE - sum(deck.values()))
            for c in self._pick(pool, rng, room, budget):
                if sum(deck.values()) >= DECK_SIZE:
                    break
                if room(c):
                    deck[c] += 1
                    if ix.stage.get(c, 0) > 0:          # keep lines legal as we add
                        for lower in ix.evolution_chain_down(c)[1:]:
                            if room(lower) and sum(deck.values()) < DECK_SIZE and deck[lower] < deck[c]:
                                deck[lower] += 1

        # 6) guarantee >=1 basic pokemon, then pad to exactly 60
        if not any(ix.is_basic_pokemon(c) for c in deck) and ix.basic_pokemon:
            bp = (rng.choice(ix.basic_pokemon) if rng else ix.basic_pokemon[0])
            deck[bp] += 1
        self._pad_to_size(deck, atk_types, rng, room)
        return deck

    def _have_for_role(self, deck, role):
        ix = self.ix
        if role == "pokemon":
            return sum(n for c, n in deck.items() if ix.is_pokemon(c))
        if role == "energy":
            return sum(n for c, n in deck.items() if ix.is_basic_energy(c) or c in ix.special_energy)
        if role in ("draw_search", "gust_switch", "energy_accel"):
            return sum(n for c, n in deck.items() if c in set(ix.role.get(role, [])))
        if role == "trainer_other":
            tagged = set(ix.role.get("draw_search", [])) | set(ix.role.get("gust_switch", [])) \
                     | set(ix.role.get("energy_accel", []))
            return sum(n for c, n in deck.items()
                       if c in set(ix.trainers) and c not in tagged)
        return 0

    def _pool_for_role(self, role, atk_types):
        ix = self.ix
        if role == "pokemon":
            # prefer pokemon matching the committed attacker type, else any
            pool = []
            for et in atk_types:
                pool += ix.pokemon_by_type.get(et, [])
            return pool or ix.basic_pokemon or ix.pokemon
        if role == "energy":
            pool = []
            for et in atk_types:
                pool += ix.energy_by_type.get(et, [])
            return pool or ix.basic_energy
        if role in ("draw_search", "gust_switch", "energy_accel"):
            return ix.role.get(role, []) or ix.trainers
        if role == "trainer_other":
            tagged = set(ix.role.get("draw_search", [])) | set(ix.role.get("gust_switch", [])) \
                     | set(ix.role.get("energy_accel", []))
            return [c for c in ix.trainers if c not in tagged] or ix.trainers
        return []

    def _pad_to_size(self, deck, atk_types, rng, room):
        ix = self.ix
        fillers = (self._pool_for_role("energy", atk_types)
                   + ix.trainers + ix.basic_pokemon)
        fillers = [c for c in fillers if c]
        guard = 0
        while sum(deck.values()) < DECK_SIZE and guard < 100000:
            guard += 1
            cand = [c for c in fillers if room(c)]
            if not cand:
                # last resort: any basic energy can exceed 4
                be = ix.basic_energy[0] if ix.basic_energy else None
                if be is None:
                    break
                deck[be] += 1
                continue
            deck[(rng.choice(cand) if rng else cand[0])] += 1
        # trim if evolution-closure overshot 60
        while sum(deck.values()) > DECK_SIZE:
            # drop a non-observed, non-basic-pokemon filler
            drop = next((c for c in deck if deck[c] > 0 and not ix.is_basic_pokemon(c)), None)
            if drop is None:
                break
            deck[drop] -= 1
            if deck[drop] == 0:
                del deck[drop]


def sample_decklist(known: Counter, index: CardIndex, rng=None,
                    template: Counter | None = None):
    """One plausible legal 60 (list of ids) consistent with `known`. Pass an rng
    to draw a *different* plausible deck per call (one per search world)."""
    deck = DeckGenerator(index).complete(known, rng=rng, template=template)
    return list(deck.elements())


# ───────────────────────────── search_begin zones ───────────────────────────

def predict_opponent_zones(obs_dict, tracker: OpponentTracker, library: ArchetypeLibrary,
                           card_db=None, min_conf=0.0, rng=None, index: CardIndex | None = None):
    """
    Produce counts-matched (opponent_deck, opponent_prize, opponent_hand,
    opponent_active) id lists for search_begin.

    v2: always returns a legal, observation-consistent guess (no None on the
    "unknown archetype" path). The matcher is used only as a soft template when
    it's confident; otherwise the generator builds from structure alone. Pass an
    `rng` to get a distinct determinization per search world.
    """
    known = tracker.known()
    if index is None:
        index = _index_for(card_db)
    if index is None:
        # no card knowledge at all -> fall back to v1 matcher-only behavior
        predicted, conf = (library.predict_decklist(known) if library else (None, 0.0))
        if predicted is None or conf < min_conf:
            return None
        full = Counter(predicted)
    else:
        template = None
        if library is not None:
            label, conf, _ = library.classify(known)
            if label is not None and conf >= max(min_conf, 0.0):
                template = library.decks.get(label)
        full = DeckGenerator(index).complete(known, rng=rng, template=template)

    st = obs_dict.get("current") or {}
    yi = st.get("yourIndex", 0) or 0
    opp = (st.get("players") or [{}, {}])[1 - yi]

    visible = Counter()
    for c in (opp.get("discard") or []):
        if c:
            visible[c["id"]] += 1
    for pk in ([(opp.get("active") or [None])[0]] + list(opp.get("bench") or [])):
        if not pk:
            continue
        visible[pk["id"]] += 1
        for k in ("energyCards", "tools", "preEvolution"):
            for c in (pk.get(k) or []):
                if c:
                    visible[c["id"]] += 1
    hidden = list((Counter(full) - visible).elements())

    dc = opp.get("deckCount", 0) or 0
    hc = opp.get("handCount", 0) or 0
    pc = len(opp.get("prize") or [])
    need = dc + hc + pc
    be = _first_basic_energy(index, card_db)
    while len(hidden) < need:
        hidden.append(be)
    if rng is not None:
        rng.shuffle(hidden)                          # which hidden cards are deck vs hand vs prize
    opp_deck = hidden[:dc]
    opp_hand = hidden[dc:dc + hc]
    opp_prize = hidden[dc + hc:dc + hc + pc]

    if opp_deck and not any(_is_basic_pokemon(index, card_db, c) for c in opp_deck):
        bp = _first_basic_pokemon(index, card_db)
        if bp is not None:
            opp_deck[0] = bp

    oa = opp.get("active") or []
    opp_active = []
    if oa and oa[0] is None:
        opp_active = [next((c for c in opp_deck if _is_basic_pokemon(index, card_db, c)),
                           opp_deck[0] if opp_deck else be)]
    return opp_deck, opp_prize, opp_hand, opp_active


# ───────────────────────────── library builder (kept) ───────────────────────

def library_from_pool(our_deck=None, decks_dir="decks", pattern="*.csv"):
    """Fit an ArchetypeLibrary from every imported deck in decks/ (the real field)."""
    import glob, os
    lists = []
    if our_deck:
        lists.append(("our", list(our_deck)))
    for f in sorted(glob.glob(os.path.join(decks_dir, pattern))):
        try:
            ids = [int(x) for x in open(f).read().split()][:DECK_SIZE]
            if len(ids) == DECK_SIZE:
                lists.append((os.path.basename(f)[:-4], ids))
        except Exception:
            pass
    return ArchetypeLibrary().fit(lists)


# ───────────────────────────── internal fallbacks ───────────────────────────

_INDEX_CACHE = {}

def _index_for(card_db):
    if card_db is None:
        return None
    key = id(card_db)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = CardIndex(card_db)
    return _INDEX_CACHE[key]

def _first_basic_energy(index, card_db, default=1):
    if index is not None and index.basic_energy:
        return index.basic_energy[0]
    for cid, d in _entries(card_db):
        if d.get("cardType") == BASIC_ENERGY_CARDTYPE:
            return cid
    return default

def _first_basic_pokemon(index, card_db):
    if index is not None and index.basic_pokemon:
        return index.basic_pokemon[0]
    for cid, d in _entries(card_db):
        if d.get("cardType") == POKEMON_CARDTYPE and d.get("basic"):
            return cid
    return None

def _is_basic_pokemon(index, card_db, cid):
    if index is not None:
        return index.is_basic_pokemon(cid)
    d = (getattr(card_db, "get", lambda x: None)(cid)) if card_db is not None else None
    return bool(d and d.get("cardType") == POKEMON_CARDTYPE and d.get("basic"))


# ───────────────────────────── self-test (no engine) ────────────────────────

if __name__ == "__main__":
    import random as _r

    # A tiny synthetic card universe with the same schema the engine emits.
    # Types: 0=pokemon 3=trainer 5=basic-energy.
    FIRE, WATER = 1, 2
    table = {
        # fire attacker line: Charmander(0) -> Charmeleon(1) -> Charizard ex(2)
        100: {"name": "Charmander", "cardType": 0, "energyType": FIRE, "basic": True,
              "attacks": [1], "base_damages": [20]},
        101: {"name": "Charmeleon", "cardType": 0, "energyType": FIRE, "stage1": True,
              "evolvesFrom": "Charmander", "attacks": [1], "base_damages": [60]},
        102: {"name": "Charizard ex", "cardType": 0, "energyType": FIRE, "stage2": True,
              "evolvesFrom": "Charmeleon", "ex": True, "attacks": [2], "base_damages": [180]},
        # water attacker (basic): Squirtle-less single-stage for contrast
        110: {"name": "Lapras", "cardType": 0, "energyType": WATER, "basic": True,
              "attacks": [1], "base_damages": [90]},
        120: {"name": "Pidgey", "cardType": 0, "energyType": 0, "basic": True,
              "attacks": [], "base_damages": []},   # support basic, no attack
        # trainers, tagged by role
        200: {"name": "Professor's Research", "cardType": 3, "energyType": 0, "tags": ["draw_search"]},
        201: {"name": "Iono", "cardType": 3, "energyType": 0, "tags": ["draw_search", "hand_disruption"]},
        202: {"name": "Nest Ball", "cardType": 3, "energyType": 0, "tags": ["draw_search"]},
        203: {"name": "Boss's Orders", "cardType": 3, "energyType": 0, "tags": ["gust_switch"]},
        204: {"name": "Switch", "cardType": 3, "energyType": 0, "tags": ["gust_switch"]},
        205: {"name": "Energy Search Plus", "cardType": 3, "energyType": 0, "tags": ["energy_accel"]},
        206: {"name": "Rare Candy", "cardType": 3, "energyType": 0, "tags": []},
        207: {"name": "Forest Stadium", "cardType": 3, "energyType": 0, "tags": []},
        208: {"name": "Tech Supporter", "cardType": 3, "energyType": 0, "tags": []},
        # energy
        300: {"name": "Fire Energy", "cardType": 5, "energyType": FIRE},
        301: {"name": "Water Energy", "cardType": 5, "energyType": WATER},
    }

    ix = CardIndex(table)
    print("index: pokemon=%d basic=%d trainers=%d energy=%d attackers=%d"
          % (len(ix.pokemon), len(ix.basic_pokemon), len(ix.trainers),
             len(ix.basic_energy), len(ix.attackers)))
    assert ix.stage[102] == 2 and ix.evolves_from[102] == 101 and ix.evolves_from[101] == 100
    assert 102 in ix.attackers and 110 in ix.attackers and 120 not in ix.attackers

    def check_legal(deck: Counter, observed: Counter, tag):
        assert sum(deck.values()) == DECK_SIZE, (tag, "size", sum(deck.values()))
        for c, n in deck.items():
            cap = DECK_SIZE if ix.is_basic_energy(c) else MAX_COPIES
            assert n <= cap, (tag, "copies", c, n)
        assert any(ix.is_basic_pokemon(c) for c in deck), (tag, "no basic pokemon")
        for c, n in observed.items():                  # never contradict what we saw
            assert deck[c] >= n, (tag, "dropped observed", c, deck[c], n)
        # evolution closure: a Stage-2 implies its Stage-1 and Basic present
        for c in deck:
            if ix.stage.get(c, 0) == 2:
                for lower in ix.evolution_chain_down(c)[1:]:
                    assert deck[lower] >= 1, (tag, "open evo line", c, lower)
        return True

    gen = DeckGenerator(ix)

    # 1) saw an evolved Charizard ex -> must back-fill the line + fire energy
    seen = Counter({102: 1, 200: 2})
    d = gen.complete(seen, rng=None)
    check_legal(d, seen, "charizard-MAP")
    assert d[101] >= 1 and d[100] >= 1, "evo line not closed"
    assert d[300] >= 1, "no fire energy for a fire attacker"
    print("MAP guess (Charizard seen): 60 cards, line closed, fire energy present  OK")

    # 2) per-world diversity: K rng worlds -> distinct legal decks
    worlds = []
    for s in range(6):
        dk = gen.complete(seen, rng=_r.Random(s))
        check_legal(dk, seen, f"world-{s}")
        worlds.append(tuple(sorted(dk.items())))
    distinct = len(set(worlds))
    print(f"6 sampled worlds -> {distinct} distinct legal decks  "
          f"{'OK' if distinct >= 4 else 'TOO FEW'}")
    assert distinct >= 4, "sampler not diverse enough"

    # 3) unknown / no observation -> still a legal 60 (no crash, no None)
    d0 = gen.complete(Counter(), rng=_r.Random(0))
    check_legal(d0, Counter(), "empty")
    print("empty-knowledge guess: legal 60 built from structure alone  OK")

    # 4) matcher template as a soft prior (library stays optional)
    lib = ArchetypeLibrary().fit([("zard", list((Counter({102: 2, 101: 2, 100: 3,
                                   200: 4, 203: 2, 300: 8}) ).elements()) + [206]*39)])
    label, conf, _ = lib.classify(Counter({102: 1, 200: 2}))
    d2 = gen.complete(Counter({102: 1}), rng=None, template=lib.decks.get(label))
    check_legal(d2, Counter({102: 1}), "templated")
    print(f"matcher prior available (label={label}, conf={conf:.2f}); template-seeded build  OK")

    # 5) sample_decklist convenience + zones smoke (no engine, minimal obs)
    cards = sample_decklist(seen, ix, rng=_r.Random(1))
    assert len(cards) == DECK_SIZE
    print("deck_inference.py v2: OK")
