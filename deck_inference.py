"""
deck_inference.py — model the opponent's hidden cards.

With a small, enumerable card pool, predicting the opponent's deck is a
*matcher*, not a generative model: accumulate the cards we've seen them reveal,
match against a small library of known archetypes, and read off the closest
60-card list. This feeds two things:

  * combat.py 2-ply: realistic `search_begin` opponent zones turn "will I
    survive their reply" from a guess into a grounded estimate.
  * gust targeting / threat assessment.

What's observable (cabt): the opponent's discard is fully visible; everything
in play is visible (with attached energy/tools); and `logs` reveal card ids as
cards pass through visible zones (PLAY/ATTACH/EVOLVE/MOVE_CARD). We track by
`serial` (unique per physical card) so re-seeing a card never double-counts.

Early game we've seen little, so confidence is low and callers should fall back
to placeholders. The archetype library is built from replay decklists via
`ArchetypeLibrary.fit(...)`; until then prediction returns None gracefully.
"""

from __future__ import annotations
from collections import Counter
import math

# log types that carry an opponent card id we can harvest
_LOG_CARD_TYPES = {6, 10, 11, 12}      # MOVE_CARD, PLAY, ATTACH, EVOLVE
BASIC_ENERGY_CARDTYPE = 5
POKEMON_CARDTYPE = 0


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
        # harvest ids that passed through visible zones since last selection
        for log in (obs_dict.get("logs") or []):
            if not isinstance(log, dict):
                continue
            if log.get("playerIndex") == oi and log.get("type") in _LOG_CARD_TYPES:
                self._note({"serial": log.get("serial"), "id": log.get("cardId")})
        return self

    def known(self) -> Counter:
        return Counter(self.seen.values())


class ArchetypeLibrary:
    """A small set of known 60-card archetypes to match the opponent against."""
    def __init__(self):
        self.decks = {}                        # label -> Counter(id -> count)
        self._df = Counter()                   # how many archetypes contain each card
        self._n = 0

    def fit(self, labeled_decklists):
        """labeled_decklists: iterable of (label, [card ids])."""
        self.decks = {label: Counter(ids) for label, ids in labeled_decklists}
        self._n = len(self.decks)
        self._df = Counter()
        for deck in self.decks.values():
            for c in deck:
                self._df[c] += 1
        return self

    def _idf(self, c):
        # signature cards (in few archetypes) weigh more
        return math.log(1 + self._n / (1 + self._df.get(c, 0)))

    def classify(self, known: Counter):
        """Return (best_label, confidence in [0,1], posterior dict)."""
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
        # confidence blends "how dominant is the top match" with "how much have we seen"
        seen = sum(known.values())
        conf = posterior[best] * min(1.0, seen / 8.0)
        return best, conf, posterior

    def predict_decklist(self, known: Counter):
        """Closest library list, made consistent with what we've actually seen."""
        label, conf, _ = self.classify(known)
        if label is None:
            return None, 0.0
        base = Counter(self.decks[label])
        for c, n in known.items():             # never contradict observed cards
            base[c] = max(base.get(c, 0), n)
        cards = list(base.elements())
        if len(cards) > 60:
            cards = cards[:60]
        return cards, conf


def predict_opponent_zones(obs_dict, tracker: OpponentTracker, library: ArchetypeLibrary,
                           card_db=None, min_conf=0.0):
    """
    Produce counts-matched (opponent_deck, opponent_prize, opponent_hand,
    opponent_active) id lists for search_begin, from the predicted decklist
    minus what's already visible. Returns None if we have no usable prediction
    (caller should fall back to placeholders).
    """
    predicted, conf = library.predict_decklist(tracker.known())
    if predicted is None or conf <= 0.0 or conf < min_conf:
        return None
    st = obs_dict.get("current") or {}
    yi = st.get("yourIndex", 0) or 0
    opp = (st.get("players") or [{}, {}])[1 - yi]

    # remove what we can already see in the opponent's visible zones
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
    hidden = list((Counter(predicted) - visible).elements())

    dc = opp.get("deckCount", 0) or 0
    hc = opp.get("handCount", 0) or 0
    pc = len(opp.get("prize") or [])
    need = dc + hc + pc
    # pad with a basic energy if our prediction came up short
    be = _first_cardtype(card_db, BASIC_ENERGY_CARDTYPE, default=1)
    while len(hidden) < need:
        hidden.append(be)
    opp_deck = hidden[:dc]
    opp_hand = hidden[dc:dc + hc]
    opp_prize = hidden[dc + hc:dc + hc + pc]

    # the engine requires >=1 basic Pokemon in the predicted opponent deck
    if card_db is not None and not any(_is_basic_pokemon(card_db, c) for c in opp_deck) and opp_deck:
        bp = _first_cardtype(card_db, POKEMON_CARDTYPE, default=None, basic=True)
        if bp is not None:
            opp_deck[0] = bp

    oa = opp.get("active") or []
    opp_active = []
    if oa and oa[0] is None:                   # face-down active -> predict a basic
        opp_active = [next((c for c in opp_deck if _is_basic_pokemon(card_db, c)), opp_deck[0] if opp_deck else be)]
    return opp_deck, opp_prize, opp_hand, opp_active


def _first_cardtype(card_db, ctype, default=None, basic=False):
    if card_db is None:
        return default
    for cid, d in getattr(card_db, "t", {}).items():
        if d.get("cardType") == ctype and (not basic or d.get("basic")):
            return cid
    return default


def _is_basic_pokemon(card_db, cid):
    if card_db is None:
        return False
    d = card_db.get(cid)
    return bool(d and d.get("cardType") == POKEMON_CARDTYPE and d.get("basic"))


if __name__ == "__main__":
    # synthetic test of the matcher logic (no engine needed)
    lib = ArchetypeLibrary().fit([
        ("Lightning",  [278]*3 + [4]*12 + [600]*4 + [50]*41),
        ("Psychic",    [50]*3 + [5]*12 + [600]*4 + [900]*41),
    ])
    seen = Counter({278: 2, 4: 3})             # revealed Bellibolt + Lightning energy
    label, conf, post = lib.classify(seen)
    print("classify:", label, round(conf, 2), {k: round(v, 2) for k, v in post.items()})
    assert label == "Lightning", post
    cards, c = lib.predict_decklist(seen)
    assert cards is not None and len(cards) == 60
    assert cards.count(278) >= 2               # consistent with what we saw
    print("predicted 60-card list length:", len(cards), "| confidence:", round(c, 2))
    print("deck_inference.py: OK")
