"""
evaluate.py — position evaluation from your perspective.

The agent does NOT maximize damage-this-turn. It scores resulting *positions*
on the things a competitive player actually weighs:

  * the prize race (favorable trades win games, even with fewer KOs)
  * prize liabilities: an ex/megaEx is 2 prizes; a benched multi-prize Pokemon
    is a gust target on the opponent's prize map (Tera is immune on the Bench)
  * board development / setup (powered attackers, evolution progress, energy)
  * deck-out pressure (a real win condition, not "low output")
  * board health / threat

evaluate_state() returns a scalar in roughly [-1, 1] (higher = better for you),
suitable both as the heuristic's decision signal and as the value-head target.

Card identities (ex/tera flags, attack damage) come from the capability table
produced by inspect_cards.py. Everything degrades gracefully without it.
"""

from __future__ import annotations
from collections import Counter
import json

COLORLESS, RAINBOW = 0, 10

WEIGHTS = {
    "prize_diff":      0.34,   # opp_prizes_left - your_prizes_left, normalized
    "bench_liability": 0.16,   # net exposed multi-prize bench risk (theirs - yours)
    "board_dev":       0.12,   # development / setup differential
    "active_threat":   0.10,   # can our active threaten vs theirs
    "energy":          0.08,   # energy in play differential
    "deckout":         0.12,   # opponent running out of deck
    "hp":              0.08,   # total board HP differential (tiebreaker)
}


class CardDB:
    """Thin wrapper over inspect_cards.py's capability_table.json (keys are strings)."""
    def __init__(self, table: dict | None = None):
        self.t = {}
        if table:
            for k, v in table.items():
                self.t[int(k)] = v

    @classmethod
    def load(cls, path):
        try:
            with open(path) as f:
                return cls(json.load(f))
        except Exception:
            return cls(None)

    def get(self, cid): return self.t.get(int(cid)) if cid is not None else None
    def ex(self, cid):  d = self.get(cid); return bool(d and (d.get("ex") or d.get("megaEx")))
    def tera(self, cid): d = self.get(cid); return bool(d and d.get("tera"))
    def has_tag(self, cid, tag):
        d = self.get(cid); return bool(d and tag in (d.get("tags") or []))
    def base_damages(self, cid):
        d = self.get(cid); return (d.get("base_damages") or []) if d else []
    def prize_value(self, cid):
        """Prizes given up if this Pokemon is KO'd."""
        return 2 if self.ex(cid) else 1


# ---- small game-logic helpers ----------------------------------------------
def can_afford(attached_energies, required_energies):
    """Can a Pokemon with `attached_energies` pay `required_energies`?"""
    avail = Counter(e for e in (attached_energies or []))
    req = list(required_energies or [])
    colorless = sum(1 for e in req if e == COLORLESS)
    wild = avail.get(RAINBOW, 0)
    for e in req:
        if e == COLORLESS:
            continue
        if avail.get(e, 0) > 0:
            avail[e] -= 1
        elif wild > 0:
            wild -= 1
        else:
            return False
    remaining = sum(v for v in avail.values()) + wild - 0
    return remaining >= colorless


def _active(ps):
    a = (ps or {}).get("active") or []
    return a[0] if a else None


def prizes_left(ps):
    return len((ps or {}).get("prize", []) or [])


# ---- evaluation components (also reusable as net features) ------------------
def bench_liability(ps, db: CardDB):
    """Exposed multi-prize risk on a player's bench (Tera = immune, excluded)."""
    risk = 0.0
    for b in (ps.get("bench") or []):
        if b is None:
            continue
        cid = b.get("id")
        if db.tera(cid):           # Tera Pokemon take no damage on the Bench
            continue
        pv = db.prize_value(cid)
        if pv < 2:                 # single-prizers are low-value gust targets
            continue
        hp, mx = b.get("hp", 0) or 0, b.get("maxHp", 1) or 1
        damaged = 1.0 - (hp / mx if mx else 1.0)
        risk += pv * (0.5 + 0.5 * damaged)   # already-damaged liabilities are worse
    return risk


def board_development(ps, db: CardDB):
    n_pkmn = (1 if _active(ps) else 0) + len(ps.get("bench") or [])
    energy = 0
    evo_bonus = 0.0
    for p in ([_active(ps)] + list(ps.get("bench") or [])):
        if not p:
            continue
        energy += len(p.get("energies") or [])
        d = db.get(p.get("id"))
        if d:
            evo_bonus += 1.0 if d.get("base_damages") else 0.0
    return 0.5 * n_pkmn + 0.3 * energy + 0.2 * evo_bonus


def active_offense(ps, db: CardDB):
    """Rough: can the active threaten? energy on board + its best base damage."""
    a = _active(ps)
    if not a:
        return 0.0
    energy = len(a.get("energies") or [])
    dmg = max(db.base_damages(a.get("id")) or [0]) / 100.0
    return 0.5 * energy + dmg


def energy_in_play(ps):
    tot = 0
    for p in ([_active(ps)] + list(ps.get("bench") or [])):
        if p:
            tot += len(p.get("energies") or [])
    return tot


def total_hp(ps):
    tot = 0
    for p in ([_active(ps)] + list(ps.get("bench") or [])):
        if p:
            tot += p.get("hp", 0) or 0
    return tot


def deckout_pressure(you, opp):
    """Positive when the opponent is close to decking out."""
    od = opp.get("deckCount", 60) or 0
    return max(0.0, (8 - od) / 8.0) if od <= 8 else 0.0


def evaluate_components(state, your_index, db: CardDB | None = None):
    db = db or CardDB(None)
    players = (state or {}).get("players") or [{}, {}]
    you = players[your_index] if len(players) > your_index else {}
    opp = players[1 - your_index] if len(players) > 1 else {}

    comp = {
        "prize_diff":      (prizes_left(opp) - prizes_left(you)) / 6.0,
        "bench_liability": (bench_liability(opp, db) - bench_liability(you, db)) / 4.0,
        "board_dev":       (board_development(you, db) - board_development(opp, db)) / 6.0,
        "active_threat":   (active_offense(you, db) - active_offense(opp, db)) / 3.0,
        "energy":          (energy_in_play(you) - energy_in_play(opp)) / 8.0,
        "deckout":         (deckout_pressure(you, opp) - deckout_pressure(opp, you)),
        "hp":              (total_hp(you) - total_hp(opp)) / 600.0,
    }
    return comp


def evaluate_state(state, your_index, db: CardDB | None = None) -> float:
    # terminal result short-circuit
    res = (state or {}).get("result", -1)
    if res is not None and res >= 0:
        if res == your_index:
            return 1.0
        if res == 2:                # draw
            return 0.0
        return -1.0
    comp = evaluate_components(state, your_index, db)
    v = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)
    return max(-1.0, min(1.0, v))


if __name__ == "__main__":
    db = CardDB({
        "278": {"ex": True, "tera": False, "base_damages": [230], "tags": []},
        "900": {"ex": True, "tera": True, "base_damages": [100], "tags": ["immunity"]},
        "7":   {"ex": False, "tera": False, "base_damages": [60], "tags": []},
    })
    state = {
        "yourIndex": 0, "result": -1,
        "players": [
            {"active": [{"id": 278, "hp": 280, "maxHp": 280, "energies": [4, 4, 0]}],
             "bench": [{"id": 7, "hp": 70, "maxHp": 70, "energies": []}],
             "prize": [None] * 4, "deckCount": 40},   # you: ahead (4 prizes left)
            {"active": [{"id": 7, "hp": 60, "maxHp": 70, "energies": [4]}],
             "bench": [{"id": 900, "hp": 230, "maxHp": 230, "energies": []}],  # their benched Tera ex (immune)
             "prize": [None] * 6, "deckCount": 38},
        ],
    }
    comp = evaluate_components(state, 0, db)
    print("components:", {k: round(v, 3) for k, v in comp.items()})
    print("value:", round(evaluate_state(state, 0, db), 3))
    # their benched Tera ex must NOT count as a liability for them
    assert bench_liability(state["players"][1], db) == 0.0, "Tera bench should be immune"
    # we're ahead on prizes -> positive value
    assert evaluate_state(state, 0, db) > 0
    print("evaluate.py: OK")
