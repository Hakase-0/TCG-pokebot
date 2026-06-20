"""
inspect_cards.py — read the ENTIRE hackathon card pool and surface edge cases.

The exact cardId->card table lives inside the cabt engine. Point this at the
engine and it will (1) profile the pool, (2) tag every card with derived
capability booleans (so our heuristic can reason without invoking the engine
each time), and (3) flag "gotcha" cards whose text breaks naive assumptions
(variable damage, immunity, prize manipulation, turn-skips, self-KO, locks).

Run against the real engine:
    CG_LIB_PATH=/path/to/cg python inspect_cards.py
Verify the logic without the engine:
    python inspect_cards.py --mock

Outputs: capability_table.json (per-card tags + base stats) and gotchas.csv.
"""

from __future__ import annotations
import csv, json, os, re, sys
from collections import Counter


# ---- load the pool from the engine, or a small mock that exercises the logic
def load_pool():
    if "--mock" not in sys.argv:
        for path in (os.environ.get("CG_LIB_PATH"), ".", "..", "./sample_submission"):
            if path and path not in sys.path:
                sys.path.insert(0, path)
        api = None
        try:
            from cg import api as _api      # engine ships as a `cg` package
            api = _api
        except Exception:
            try:
                import api as _api           # or `cg/` itself on the path
                api = _api
            except Exception as e:
                print(f"[!] engine import failed ({e}); falling back to --mock", file=sys.stderr)
        if api is not None:
            cards = [c.__dict__ if hasattr(c, "__dict__") else c for c in api.all_card_data()]
            attacks = {(a.attackId if hasattr(a, "attackId") else a["attackId"]):
                       (a.__dict__ if hasattr(a, "__dict__") else a)
                       for a in api.all_attack()}
            return cards, attacks
    return _mock_pool()


def _mock_pool():
    """A handful of cards chosen to exercise every tagger branch."""
    attacks = {
        1: {"attackId": 1, "name": "Tackle", "text": "", "damage": 30, "energies": [0]},
        2: {"attackId": 2, "name": "Powerful Hand",
            "text": "This attack does 20 damage for each card in your hand.",
            "damage": 0, "energies": [5, 5]},
        3: {"attackId": 3, "name": "Thunderous Bolt",
            "text": "Discard 2 Energy from this Pokemon.", "damage": 230, "energies": [4, 4, 0]},
        4: {"attackId": 4, "name": "Gust Strike",
            "text": "Flip a coin. If heads, this attack does 60 more damage.",
            "damage": 40, "energies": [0]},
    }
    cards = [
        {"cardId": 278, "name": "Bellibolt ex", "cardType": 0, "energyType": 4,
         "hp": 280, "retreatCost": 2, "weakness": 6, "resistance": None,
         "basic": False, "stage1": True, "stage2": False,
         "ex": True, "megaEx": False, "tera": False, "aceSpec": False,
         "evolvesFrom": "Tadbulb", "attacks": [3],
         "skills": [{"name": "Electromagnetic Circuit",
                     "text": "Attach 2 Lightning Energy from your hand to your Pokemon."}]},
        {"cardId": 50, "name": "Alakazam", "cardType": 0, "energyType": 5,
         "hp": 150, "retreatCost": 2, "weakness": 7, "resistance": None,
         "basic": False, "stage1": False, "stage2": True, "ex": False,
         "megaEx": False, "tera": False, "aceSpec": False,
         "evolvesFrom": "Kadabra", "attacks": [2], "skills": []},
        {"cardId": 900, "name": "Some Tera ex", "cardType": 0, "energyType": 3,
         "hp": 230, "retreatCost": 1, "weakness": 4, "resistance": None,
         "basic": True, "stage1": False, "stage2": False, "ex": True,
         "megaEx": False, "tera": True, "aceSpec": False,
         "evolvesFrom": None, "attacks": [4],
         "skills": [{"name": "Tera",
                     "text": "As long as this Pokemon is on your Bench, "
                             "prevent all damage done to it by attacks."}]},
        {"cardId": 410, "name": "Iono", "cardType": 3, "energyType": 0, "hp": 0,
         "retreatCost": 0, "weakness": None, "resistance": None, "basic": False,
         "stage1": False, "stage2": False, "ex": False, "megaEx": False,
         "tera": False, "aceSpec": False, "evolvesFrom": None, "attacks": [],
         "skills": [{"name": "Iono",
                     "text": "Each player shuffles their hand into their deck, "
                             "then draws a card for each of their remaining Prize cards."}]},
        {"cardId": 411, "name": "Boss's Orders", "cardType": 3, "energyType": 0,
         "hp": 0, "retreatCost": 0, "weakness": None, "resistance": None,
         "basic": False, "stage1": False, "stage2": False, "ex": False,
         "megaEx": False, "tera": False, "aceSpec": False, "evolvesFrom": None,
         "attacks": [], "skills": [{"name": "Boss's Orders",
                     "text": "Switch in 1 of your opponent's Benched Pokemon to the Active Spot."}]},
        {"cardId": 999, "name": "Some ACE SPEC", "cardType": 1, "energyType": 0,
         "hp": 0, "retreatCost": 0, "weakness": None, "resistance": None,
         "basic": False, "stage1": False, "stage2": False, "ex": False,
         "megaEx": False, "tera": False, "aceSpec": True, "evolvesFrom": None,
         "attacks": [], "skills": [{"name": "?", "text": "Heal all damage from 1 of your Pokemon."}]},
    ]
    return cards, attacks


# ---- keyword taggers: where edge cases get caught --------------------------
PATTERNS = {
    "variable_damage":  r"\bfor each\b|\b\d+\s*[x×]\b|\btimes\b|plus \d+ more|more damage",
    "coin_flip":        r"\bflip\b.*\bcoin",
    "bench_damage":     r"benched? pok[eé]mon|to that pok[eé]mon|don't apply weakness",
    "self_damage":      r"to (itself|this pok[eé]mon)|to your own",
    "gust_switch":      r"switch in.*opponent|opponent.*active spot|gust",
    "self_switch":      r"switch (this|your active)",
    "draw_search":      r"\bdraw\b|search your deck|look at the top",
    "energy_accel":     r"attach .* energy",
    "energy_denial":    r"discard .* energy from your opponent|opponent.*discard.*energy",
    "hand_disruption":  r"shuffle .* hand into|each player shuffles their hand",
    "prize_dependent":  r"remaining prize|prize cards? (you|your opponent) (have|has)",
    "prize_manip":      r"take \d+ (more )?prize|take an? .*prize card",
    "heal":             r"\bheal\b|remove .* damage counter",
    "immunity":         r"prevent all damage|no damage|takes no damage|can't be|cannot be|isn't affected",
    "lock":             r"can't .* attack|cannot attack|can't play|can't use|no.*item cards|no.*supporter",
    "turn_skip":        r"end your turn|skip your|during your next turn.*can't|this pok[eé]mon can't attack during your next",
    "status_inflict":   r"\bpoisoned\b|\bburned\b|\basleep\b|\bparalyzed\b|\bconfused\b",
    "win_loss_effect":  r"\byou win\b|\byou lose\b|wins the game|takes? all .* prize",
    "conditional_dmg":  r"does nothing|if .* this attack does|if .*,? this pok[eé]mon",
}
COMPILED = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}


def card_text(card, attacks):
    parts = []
    for s in card.get("skills") or []:
        parts.append(s.get("text", "") if isinstance(s, dict) else getattr(s, "text", ""))
    for aid in card.get("attacks") or []:
        a = attacks.get(aid)
        if a:
            parts.append(a.get("text", "") if isinstance(a, dict) else getattr(a, "text", ""))
    return " \n ".join(p for p in parts if p)


def tag_card(card, attacks):
    text = card_text(card, attacks)
    tags = {name: bool(rx.search(text)) for name, rx in COMPILED.items()}
    base_dmgs = []
    for aid in card.get("attacks") or []:
        a = attacks.get(aid)
        if a:
            base_dmgs.append(a.get("damage", 0) if isinstance(a, dict) else getattr(a, "damage", 0))
    # an attack with text but 0 printed damage is an "effect attack" (damage is text-driven)
    has_effect_attack = any(
        (not (attacks.get(aid, {}).get("damage") if isinstance(attacks.get(aid), dict)
              else getattr(attacks.get(aid), "damage", 0)))
        and card_text({"skills": [], "attacks": [aid]}, attacks).strip()
        for aid in (card.get("attacks") or [])
    )
    return tags, base_dmgs, has_effect_attack, text


def main():
    cards, attacks = load_pool()
    print(f"Loaded {len(cards)} cards, {len(attacks)} attacks "
          f"{'(MOCK)' if '--mock' in sys.argv else '(real engine)'}\n")

    flag_counts = Counter()
    type_counts = Counter()
    tag_counts = Counter()
    table = {}
    gotchas = []

    for c in cards:
        cid = c.get("cardId")
        type_counts[c.get("cardType")] += 1
        for f in ("ex", "megaEx", "tera", "aceSpec", "basic", "stage1", "stage2"):
            if c.get(f):
                flag_counts[f] += 1
        tags, base_dmgs, eff_attack, text = tag_card(c, attacks)
        for k, v in tags.items():
            if v:
                tag_counts[k] += 1
        table[cid] = {
            "name": c.get("name"), "cardType": c.get("cardType"),
            "hp": c.get("hp"), "energyType": c.get("energyType"),
            "weakness": c.get("weakness"), "retreatCost": c.get("retreatCost"),
            "ex": c.get("ex"), "megaEx": c.get("megaEx"), "tera": c.get("tera"),
            "aceSpec": c.get("aceSpec"),
            "base_damages": base_dmgs, "effect_attack": eff_attack,
            "tags": [k for k, v in tags.items() if v],
        }
        # a card is a "gotcha" if it does anything that breaks static reasoning
        risky = ("variable_damage", "immunity", "prize_dependent", "prize_manip",
                 "turn_skip", "win_loss_effect", "lock", "conditional_dmg",
                 "hand_disruption", "energy_denial")
        hits = [k for k in risky if tags.get(k)]
        if hits or eff_attack or c.get("tera") or c.get("megaEx"):
            gotchas.append({"cardId": cid, "name": c.get("name"),
                            "reasons": ",".join(hits + (["effect_attack"] if eff_attack else [])
                                                + (["tera_bench_immunity"] if c.get("tera") else [])
                                                + (["mega_evolution"] if c.get("megaEx") else []))})

    # ---- report ----
    print("Card types:", dict(type_counts))
    print("Special flags:", dict(flag_counts))
    print("\nCapability tag frequencies (high counts = must handle well):")
    for k, n in tag_counts.most_common():
        print(f"  {k:18s} {n}")
    print(f"\nGotcha cards (break naive static reasoning): {len(gotchas)}")
    for g in gotchas[:20]:
        print(f"  [{g['cardId']}] {g['name']}: {g['reasons']}")

    with open("capability_table.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    with open("gotchas.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cardId", "name", "reasons"])
        w.writeheader(); w.writerows(gotchas)
    # attack_table.json: what the policy/combat code needs for damage & costs
    atk_out = {}
    for aid, a in attacks.items():
        get = (lambda k, d=None: a.get(k, d)) if isinstance(a, dict) else (lambda k, d=None: getattr(a, k, d))
        atk_out[str(aid)] = {"name": get("name", ""), "damage": get("damage", 0) or 0,
                             "energies": list(get("energies", []) or []), "text": get("text", "")}
    with open("attack_table.json", "w") as f:
        json.dump(atk_out, f, indent=2, ensure_ascii=False)
    print("\nWrote capability_table.json, attack_table.json, and gotchas.csv")


if __name__ == "__main__":
    main()
