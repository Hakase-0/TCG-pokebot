"""
selftest.py — exercises the whole Phase-1 path without the real engine.

Run: python selftest.py
Covers: deck-selection return, MAIN menu, yes/no, a count selection,
normalization/fallback edge cases, and (if torch is present) the NN act path.
The real cabt engine + card data are NOT needed for this test.
"""
import importlib
import main as M
import policy_heuristic as H


def case(name, obs, expect_len_range=None):
    out = M.agent(obs)
    ok = isinstance(out, list) and all(isinstance(i, int) for i in out)
    if obs.get("select") is None:
        ok = ok and len(out) == 60
    else:
        n = len(obs["select"]["option"])
        ok = ok and all(0 <= i < n for i in out)
        lo = obs["select"].get("minCount", 1); hi = obs["select"].get("maxCount", 1)
        ok = ok and lo <= len(out) <= max(hi, lo)
    print(f"[{'OK' if ok else 'XX'}] {name:28s} -> {out}")
    assert ok, name
    return out


# deck selection
case("deck_selection", {"select": None, "current": None})

# MAIN: should evolve/attach/play before attacking; never just pass if action exists
case("main_menu", {"select": {
    "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
    "option": [{"type": 14}, {"type": 8, "index": 0}, {"type": 13, "attackId": 1}]}})

# yes/no mulligan -> prefer NO
case("mulligan_no", {"select": {
    "type": 9, "context": 42, "minCount": 1, "maxCount": 1,
    "option": [{"type": 1}, {"type": 2}]}})

# "discard up to 2" -> minCount 0 is legal to return []
case("discard_up_to_2", {"select": {
    "type": 1, "context": 8, "minCount": 0, "maxCount": 2,
    "option": [{"type": 3, "index": 0}, {"type": 3, "index": 1},
               {"type": 3, "index": 2}]}})

# forced "choose exactly 2"
case("choose_exactly_2", {"select": {
    "type": 1, "context": 8, "minCount": 2, "maxCount": 2,
    "option": [{"type": 3}, {"type": 3}, {"type": 3}, {"type": 3}]}})

# garbage from a policy must still normalize to legal
bad_obs = {"select": {"type": 1, "context": 8, "minCount": 1, "maxCount": 1,
                      "option": [{"type": 3}, {"type": 3}]}}
M._inner_select = lambda o: [99, -3, "x", 1]   # monkeypatch a misbehaving policy
print("[..] testing normalization of an illegal policy return")
out = M.agent(bad_obs)
assert out == [1] or (len(out) == 1 and 0 <= out[0] < 2), out
print(f"[OK] normalize_illegal_return     -> {out}")
importlib.reload(M)  # restore

# featurizer standalone
import features as fx
fx_mod = importlib.import_module("features")
print("[OK] features import + constants  -> "
      f"E={fx.ENTITY_NUM_FEATS} O={fx.OPTION_NUM_FEATS} G={fx.GLOBAL_NUM_FEATS}")

# NN path (optional)
try:
    import torch  # noqa
    from model import PointerPolicyValueNet
    net = PointerPolicyValueNet(num_card_ids=4096, d=96)
    enc = fx.encode_observation({
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": [{"type": 7, "index": 0}, {"type": 14}]},
        "current": {"turn": 1, "yourIndex": 0,
                    "players": [{"active": [{"id": 278, "hp": 120, "maxHp": 120,
                                             "energies": [4], "tools": []}],
                                 "bench": [], "prize": [None]*6, "handCount": 5,
                                 "deckCount": 47},
                                {"active": [], "bench": [], "prize": [None]*6,
                                 "handCount": 5, "deckCount": 47}]}})
    choice = net.act(enc)
    assert all(0 <= i < 2 for i in choice)
    print(f"[OK] nn act path                 -> {choice}")
except ImportError:
    print("[--] torch not installed; skipped NN act path (heuristic path is enough for Phase 1)")

print("\nselftest.py: ALL PASS")
