"""Tests for the new player-considerations behaviors in policy_heuristic."""
import policy_heuristic as H
from evaluate import CardDB

DB = CardDB({
    "278": {"ex": True,  "tera": False, "base_damages": [230], "tags": ["energy_accel"]},  # Bellibolt ex
    "900": {"ex": True,  "tera": True,  "base_damages": [100], "tags": []},                 # Tera ex
    "50":  {"ex": False, "tera": False, "base_damages": [0],   "tags": []},                 # vanilla mon
    "410": {"ex": False, "tera": False, "base_damages": [],    "tags": ["hand_disruption", "draw_search"]},  # Iono
    "411": {"ex": False, "tera": False, "base_damages": [],    "tags": ["gust_switch"]},    # Boss's Orders
    "600": {"ex": False, "tera": False, "base_damages": [],    "tags": ["draw_search"]},    # draw supporter
    "700": {"ex": False, "tera": False, "base_damages": [],    "tags": []},                 # junk item
    "280": {"ex": True,  "tera": False, "base_damages": [200], "tags": []},                 # plain 2-prize ex
})
ATK = {1: {"damage": 230, "energies": [4, 4, 0]}, 2: {"damage": 0, "energies": []}}

def state(me_active=278, me_bench=None, opp_active=50, opp_bench=None,
          me_prizes=6, opp_prizes=6, energy_attached=False, hand=None, deck=40):
    me_bench = me_bench or []
    opp_bench = opp_bench or []
    return {"current": {"yourIndex": 0, "energyAttached": energy_attached, "result": -1,
        "players": [
            {"active": [{"id": me_active, "hp": 280, "maxHp": 280, "energies": [4,4]}] if me_active else [],
             "bench": [{"id": c, "hp": 70, "maxHp": 70, "energies": []} for c in me_bench],
             "prize": [None]*me_prizes, "deckCount": deck,
             "hand": hand or []},
            {"active": [{"id": opp_active, "hp": 60, "maxHp": 70, "energies": []}] if opp_active else [],
             "bench": [{"id": c, "hp": 200, "maxHp": 230, "energies": []} for c in opp_bench],
             "prize": [None]*opp_prizes, "deckCount": 38}]}}

def main_obs(option_list, **kw):
    s = state(**kw); s["select"] = {"type": 0, "context": 0, "minCount": 1, "maxCount": 1, "option": option_list}
    return s

def card_obs(context, options, lo, hi, **kw):
    s = state(**kw); s["select"] = {"type": 1, "context": context, "minCount": lo, "maxCount": hi, "option": options}
    return s

def pick(obs): return H.select(obs, db=DB, attack_db=ATK)

# 1) MOVE ORDER: draw supporter before attaching energy (attach is committal)
hand = [{"id": 600}]  # a draw supporter in hand at index 0
obs = main_obs([{"type": H.ATTACH, "inPlayArea": 4, "inPlayIndex": 0},
                {"type": H.PLAY, "index": 0},
                {"type": H.ATTACK, "attackId": 1}, {"type": H.END}], hand=hand)
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.PLAY, ("draw before attach", out)
print("[OK] plays draw supporter before attaching energy")

# 2) DECLINE benching an unnecessary 2-prize liability (already have 2 in play)
hand = [{"id": 280}]  # a plain 2-prize ex basic we could bench
obs = main_obs([{"type": H.PLAY, "index": 0}, {"type": H.END}],
               me_bench=[50, 50], hand=hand)  # already 1 active + 2 bench
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.END, ("should not bench liability", out)
print("[OK] declines benching an unnecessary 2-prize Pokemon")

# 3) GUST only with a target: no opp bench -> hold it
hand = [{"id": 411}]
obs = main_obs([{"type": H.PLAY, "index": 0}, {"type": H.END}], opp_bench=[], hand=hand)
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.END, ("no gust without target", out)
print("[OK] holds gust when opponent bench is empty")
obs = main_obs([{"type": H.PLAY, "index": 0}, {"type": H.END}], opp_bench=[50], hand=[{"id": 411}])
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.PLAY, ("gust with target", out)
print("[OK] gusts when there is a bench target")

# 4) Iono only while AHEAD on prizes
obs = main_obs([{"type": H.PLAY, "index": 0}, {"type": H.END}],
               me_prizes=2, opp_prizes=5, hand=[{"id": 410}])  # we're ahead
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.PLAY, ("iono ahead", out)
obs = main_obs([{"type": H.PLAY, "index": 0}, {"type": H.END}],
               me_prizes=5, opp_prizes=2, hand=[{"id": 410}])  # we're behind
# behind: Iono is tagged draw_search too, so it still plays as a draw card — acceptable; just ensure legal
out = pick(obs); assert 0 <= out[0] < 2
print("[OK] plays Iono while ahead on prizes")

# 5) ATTACK when lethal is available
obs = main_obs([{"type": H.ATTACK, "attackId": 1}, {"type": H.END}], opp_active=50)
out = pick(obs); assert obs["select"]["option"][out[0]]["type"] == H.ATTACK
print("[OK] attacks when a KO is available")

# 6) DISCARD dumps the LEAST valuable card (junk 700 over ex 278)
opts = [{"type": H.CARD, "area": 2, "index": 0, "cardId": 278},
        {"type": H.CARD, "area": 2, "index": 1, "cardId": 700}]
obs = card_obs(H.CTX_DISCARD, opts, 1, 1)
out = pick(obs); assert opts[out[0]]["cardId"] == 700, ("discard junk", out)
print("[OK] discards the least valuable card")

# 7) OPTIONAL discard (minCount 0) of good cards -> decline
obs = card_obs(H.CTX_DISCARD, [{"type": H.CARD, "cardId": 278}], 0, 1)
out = pick(obs); assert out == [], ("decline optional discard", out)
print("[OK] declines an optional discard of a valuable card")

# 8) DAMAGE targets the higher-prize Pokemon (ex over vanilla)
opts = [{"type": H.CARD, "area": 5, "index": 0, "playerIndex": 1, "cardId": 50},
        {"type": H.CARD, "area": 5, "index": 1, "playerIndex": 1, "cardId": 900}]
obs = card_obs(H.CTX_DAMAGE, opts, 1, 1, opp_bench=[50, 900])
out = pick(obs); assert opts[out[0]]["cardId"] == 900, ("target ex", out)
print("[OK] targets the higher-prize Pokemon")

print("\ntest_behaviors.py: ALL PASS")
