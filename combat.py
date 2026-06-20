"""
combat.py — the engine is our combat oracle (now 2-ply + target-aware).

We never reimplement card text. We ask the real engine "what happens if I do
this" via search_begin/search_step, read the resulting board, and score it with
evaluate.evaluate_state.

Upgrades over v1:
  * opponent determinization can come from deck_inference (an `opp_predictor`
    callable) instead of placeholders — needed for a meaningful opponent reply.
  * `plies=2` rolls the opponent's reply out too, so should_attack can see
    "they KO me back" — the real reason to sometimes pass instead of attack.
  * `best_target` optimizes which opponent Pokemon to hit (gust / damage
    targeting) by simulating each choice.

Public API (all degrade to None / base if the engine isn't importable):
  find_lethal(obs, deck, db, atk, opp_predictor=None)
  should_attack(obs, atk_i, end_i, deck, db, atk, opp_predictor=None, plies=1)
  best_target(obs, deck, db, atk, opp_predictor=None)
  refine_selection(obs, deck, db, atk, base, opp_predictor=None, plies=1)
"""

from __future__ import annotations
from collections import Counter

try:
    from cg import api as _api
except Exception:
    try:
        import api as _api
    except Exception:
        _api = None

import policy_heuristic as _H
from evaluate import evaluate_state

ATTACK, END, MAIN, CARD = 13, 14, 0, 1
# SelectContexts where the agent targets an opponent Pokemon
_TARGET_CTX = {3, 4, 13, 14, 15}      # SWITCH, TO_ACTIVE(gust), DAMAGE_COUNTER(_ANY), DAMAGE
_FILLERS = None


def available() -> bool:
    return _api is not None


def _fillers():
    global _FILLERS
    if _FILLERS is None and _api is not None:
        cards = _api.all_card_data()
        bp = next((c.cardId for c in cards if c.cardType == 0 and c.basic), 1)
        be = next((c.cardId for c in cards if c.cardType == 5), 1)
        _FILLERS = (bp, be)
    return _FILLERS or (1, 1)


def _obs_to_dict(observation):
    if isinstance(observation, dict):
        return observation
    import dataclasses
    try:
        return dataclasses.asdict(observation)
    except Exception:
        def rec(x):
            if hasattr(x, "__dict__"):
                return {k: rec(v) for k, v in vars(x).items()}
            if isinstance(x, list):
                return [rec(v) for v in x]
            return x
        return rec(observation)


def _visible_my_ids(state, yi):
    p = state["players"][yi]
    ids = []
    for z in ("hand", "discard"):
        for c in (p.get(z) or []):
            if c:
                ids.append(c["id"])
    for pk in ([(p.get("active") or [None])[0]] + list(p.get("bench") or [])):
        if not pk:
            continue
        ids.append(pk["id"])
        for k in ("energyCards", "tools", "preEvolution"):
            for c in (pk.get(k) or []):
                if c:
                    ids.append(c["id"])
    return ids


def _determinize(obs_dict, deck60, opp_predictor=None):
    bp, be = _fillers()
    st = obs_dict["current"]
    yi = st["yourIndex"]
    me, opp = st["players"][yi], st["players"][1 - yi]
    rem = list((Counter(deck60) - Counter(_visible_my_ids(st, yi))).elements())
    dc, pc = me["deckCount"], len(me.get("prize") or [])
    while len(rem) < dc + pc:
        rem.append(be)
    your_deck, your_prize = rem[:dc], rem[dc:dc + pc]

    zones = None
    if opp_predictor is not None:
        try:
            zones = opp_predictor(obs_dict)
        except Exception:
            zones = None
    if zones:
        opp_deck, opp_prize, opp_hand, opp_active = zones
    else:
        odc = opp["deckCount"]
        opp_deck = ([bp] + [be] * max(odc - 1, 0))[:max(odc, 1)]
        opp_prize = [be] * len(opp.get("prize") or [])
        opp_hand = [be] * (opp.get("handCount", 0) or 0)
        oa = opp.get("active") or []
        opp_active = [bp] if (oa and oa[0] is None) else []
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand, opp_active


def simulate_line(obs_dict, first_select, deck60, db, attack_db,
                  opp_predictor=None, plies=1, max_steps=120):
    """
    Play `first_select`, then roll out with the heuristic. plies=1 stops at the
    opponent's turn (value = our position after our turn); plies=2 also plays
    the opponent's reply and evaluates at the start of our next turn.
    Returns (result, value) or (None, None) on failure.
    """
    if _api is None:
        return None, None
    yi = obs_dict["current"]["yourIndex"]
    try:
        oc = _api.to_observation_class(obs_dict)
        root = _api.search_begin(oc, *_determinize(obs_dict, deck60, opp_predictor))
        state = _api.search_step(root.searchId, list(first_select))
    except Exception:
        try:
            _api.search_end()
        except Exception:
            pass
        return None, None
    try:
        phase = "ours"
        steps = 0
        while steps < max_steps:
            obs = _obs_to_dict(state.observation)
            cur = obs.get("current") or {}
            res = cur.get("result", -1)
            if res is not None and res >= 0:
                return res, (1.0 if res == yi else (0.0 if res == 2 else -1.0))
            sel = obs.get("select")
            if sel is None:
                break
            cur_yi = cur.get("yourIndex", yi)
            if phase == "ours" and cur_yi != yi:
                if plies < 2:
                    return -1, evaluate_state(cur, yi, db)
                phase = "opp"
            if phase == "opp" and cur_yi == yi:
                return -1, evaluate_state(cur, yi, db)   # our next turn begins
            choice = _H.select(obs, db=db, attack_db=attack_db)
            state = _api.search_step(state.searchId, choice)
            steps += 1
        cur = (_obs_to_dict(state.observation).get("current") or {})
        return cur.get("result", -1), evaluate_state(cur, yi, db)
    except Exception:
        return None, None
    finally:
        try:
            _api.search_end()
        except Exception:
            pass


def find_lethal(obs_dict, deck60, db, attack_db, opp_predictor=None):
    if _api is None:
        return None
    sel = obs_dict.get("select") or {}
    if sel.get("type") != MAIN:
        return None
    yi = obs_dict["current"]["yourIndex"]
    for i, o in enumerate(sel.get("option", [])):
        if o.get("type") != ATTACK:
            continue
        res, _ = simulate_line(obs_dict, [i], deck60, db, attack_db, opp_predictor, plies=1)
        if res == yi:
            return i
    return None


def should_attack(obs_dict, attack_idx, end_idx, deck60, db, attack_db,
                  opp_predictor=None, plies=1):
    _, a_val = simulate_line(obs_dict, [attack_idx], deck60, db, attack_db, opp_predictor, plies)
    _, e_val = simulate_line(obs_dict, [end_idx], deck60, db, attack_db, opp_predictor, plies)
    if a_val is None or e_val is None:
        return attack_idx
    return attack_idx if a_val >= e_val else end_idx


def best_target(obs_dict, deck60, db, attack_db, opp_predictor=None):
    """For a single-pick opponent-target selection, the option with best value."""
    sel = obs_dict.get("select") or {}
    opts = sel.get("option", [])
    if not opts or _api is None:
        return None
    best_i, best_v = None, None
    for i in range(len(opts)):
        _, v = simulate_line(obs_dict, [i], deck60, db, attack_db, opp_predictor, plies=1)
        if v is not None and (best_v is None or v > best_v):
            best_i, best_v = i, v
    return best_i


def _is_opponent_target(sel, yi):
    if sel.get("type") != CARD or sel.get("context") not in _TARGET_CTX:
        return False
    opts = sel.get("option", [])
    return any(o.get("playerIndex") not in (None, yi) for o in opts)


def refine_selection(obs_dict, deck60, db, attack_db, base_choice,
                     opp_predictor=None, plies=1):
    if _api is None or deck60 is None:
        return base_choice
    sel = obs_dict.get("select") or {}
    yi = (obs_dict.get("current") or {}).get("yourIndex", 0)
    opts = sel.get("option", [])
    try:
        if sel.get("type") == MAIN:
            base_i = base_choice[0] if base_choice else None
            base_type = opts[base_i].get("type") if (base_i is not None and base_i < len(opts)) else None
            if base_type not in (ATTACK, END):
                return base_choice
            lethal = find_lethal(obs_dict, deck60, db, attack_db, opp_predictor)
            if lethal is not None:
                return [lethal]
            attacks = [i for i, o in enumerate(opts) if o.get("type") == ATTACK]
            end_i = next((i for i, o in enumerate(opts) if o.get("type") == END), None)
            if not attacks or end_i is None:
                return base_choice
            best_atk, best_val = attacks[0], None
            for i in attacks:
                _, v = simulate_line(obs_dict, [i], deck60, db, attack_db, opp_predictor, plies=1)
                if v is not None and (best_val is None or v > best_val):
                    best_atk, best_val = i, v
            return [should_attack(obs_dict, best_atk, end_i, deck60, db, attack_db, opp_predictor, plies)]

        # gust / damage targeting: pick the opponent target that scores best
        if _is_opponent_target(sel, yi) and len(opts) >= 2 and sel.get("maxCount", 1) == 1:
            bt = best_target(obs_dict, deck60, db, attack_db, opp_predictor)
            if bt is not None:
                return [bt]
        return base_choice
    except Exception:
        return base_choice
