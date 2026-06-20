"""
mock_engine.py — a tiny stand-in for the cabt `game` module.

It is NOT a rules engine. It replays a scripted sequence of schema-correct
observations so we can verify, without the real engine, that our agent:
  * handles every SelectType it will meet (SETUP, MAIN, YES_NO, CARD, ATTACK),
  * always returns a legal selection (indices within [minCount, maxCount]),
  * drives the battle_start -> battle_select -> result loop to termination.

Exposes the same surface as `game`: battle_start, battle_select, battle_finish.
run_game.py uses this when the real engine isn't importable (`--mock`).
"""

from __future__ import annotations


def _board(your_index=0, turn=1, energy_attached=False, hand=None):
    return {
        "turn": turn, "turnActionCount": 0, "yourIndex": your_index,
        "firstPlayer": 0, "supporterPlayed": False, "stadiumPlayed": False,
        "energyAttached": energy_attached, "retreated": False, "result": -1,
        "stadium": [], "looking": None,
        "players": [
            {"active": [{"id": 278, "hp": 280, "maxHp": 280, "energies": [4, 4], "tools": []}],
             "bench": [{"id": 7, "hp": 70, "maxHp": 70, "energies": [], "tools": []}],
             "prize": [None] * 6, "deckCount": 47, "discard": [],
             "handCount": len(hand or []), "hand": hand or [],
             "poisoned": False, "burned": False, "asleep": False,
             "paralyzed": False, "confused": False},
            {"active": [{"id": 99, "hp": 90, "maxHp": 130, "energies": [3], "tools": []}],
             "bench": [], "prize": [None] * 6, "deckCount": 50, "discard": [],
             "handCount": 5, "hand": None,
             "poisoned": False, "burned": False, "asleep": False,
             "paralyzed": False, "confused": False},
        ],
    }


class MockEngine:
    def __init__(self):
        self.i = 0
        self.script = []

    # ---- protocol surface (mirrors the `game` module) ----------------------
    def battle_start(self, deck0, deck1):
        if len(deck0) != 60 or len(deck1) != 60:
            raise ValueError("each deck must contain exactly 60 cards")
        self.i = 0
        self.script = self._build_script()
        return self.script[0], {"error": 0}

    def battle_select(self, select_list):
        # validate the agent's selection against the options it was shown
        cur = self.script[self.i]
        sel = cur.get("select")
        if sel is not None:
            n = len(sel.get("option", []))
            lo, hi = sel.get("minCount", 1), max(sel.get("maxCount", 1), sel.get("minCount", 1))
            assert isinstance(select_list, list), "agent must return a list"
            assert all(isinstance(x, int) and 0 <= x < n for x in select_list), \
                f"illegal index in {select_list} for {n} options"
            assert lo <= len(select_list) <= hi, \
                f"selected {len(select_list)} not in [{lo},{hi}]"
        self.i += 1
        return self.script[self.i]

    def battle_finish(self):
        self.i = 0
        self.script = []

    # ---- the scripted scenario --------------------------------------------
    def _build_script(self):
        draw_supporter_in_hand = [{"id": 600}, {"id": 7}]  # idx0 supporter, idx1 basic
        return [
            # 1) setup: choose an Active from basics in hand
            {"logs": [], "current": _board(hand=draw_supporter_in_hand),
             "select": {"type": 1, "context": 1, "minCount": 1, "maxCount": 1,
                        "deck": None, "contextCard": None,
                        "option": [{"type": 3, "area": 2, "index": 1, "cardId": 7}]}},
            # 2) MAIN: play / attach / attack / end all available (move-order test)
            {"logs": [], "current": _board(turn=1, hand=draw_supporter_in_hand),
             "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                        "option": [{"type": 8, "inPlayArea": 4, "inPlayIndex": 0},  # ATTACH
                                   {"type": 7, "index": 0},                          # PLAY supporter
                                   {"type": 13, "attackId": 1},                      # ATTACK
                                   {"type": 14}]}},                                  # END
            # 3) MAIN again (fewer options)
            {"logs": [], "current": _board(turn=1, energy_attached=True),
             "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                        "option": [{"type": 13, "attackId": 1}, {"type": 14}]}},
            # 4) YES/NO: activate an effect
            {"logs": [], "current": _board(turn=3),
             "select": {"type": 9, "context": 43, "minCount": 1, "maxCount": 1,
                        "contextCard": {"id": 278},
                        "option": [{"type": 1}, {"type": 2}]}},
            # 5) CARD: discard 1 of 2 (quality test)
            {"logs": [], "current": _board(turn=3),
             "select": {"type": 1, "context": 8, "minCount": 1, "maxCount": 1,
                        "option": [{"type": 3, "area": 2, "index": 0, "cardId": 278},
                                   {"type": 3, "area": 2, "index": 1, "cardId": 700}]}},
            # 6) COUNT: choose how many to draw
            {"logs": [], "current": _board(turn=5),
             "select": {"type": 8, "context": 38, "minCount": 1, "maxCount": 1,
                        "option": [{"type": 0, "number": 1}, {"type": 0, "number": 2},
                                   {"type": 0, "number": 3}]}},
            # 7) terminal: player 0 wins
            {"logs": [{"type": 23}],
             "current": {**_board(turn=7), "result": 0}, "select": None},
        ]


# module-level singleton so callers can `import mock_engine as game`
_ENGINE = MockEngine()
def battle_start(d0, d1): return _ENGINE.battle_start(d0, d1)
def battle_select(s):     return _ENGINE.battle_select(s)
def battle_finish():      return _ENGINE.battle_finish()
