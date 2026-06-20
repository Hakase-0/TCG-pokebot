"""
features.py — turn a raw cabt `obs_dict` into numpy arrays for the network.

Everything here maps to the official cabt API
(https://matsuoinstitute.github.io/cabt/api.html). It is intentionally
defensive: the live ladder must never crash, so every field access uses
.get(...) with sane defaults and tolerates None (face-down cards, opponent
hand, empty active spot, the deck-selection phase where current/select are None).

The featurizer is deck-agnostic. It reads structure (option types, board
entities, energies, statuses), not specific card IDs — card identity enters
only through the learned embedding table (see model.py).
"""

from __future__ import annotations
import numpy as np

# ---- enum sizes (from the cabt api spec) -------------------------------------
NUM_ENERGY_TYPES = 12        # EnergyType 0..11 (incl. RAINBOW, TEAM_ROCKET)
NUM_OPTION_TYPES = 17        # OptionType 0..16
NUM_SELECT_TYPES = 11        # SelectType 0..10
NUM_SELECT_CTX   = 49        # SelectContext 0..48
NUM_AREA_TYPES   = 13        # AreaType 1..12 (we keep index 0 as "none")
MAX_ENTITIES     = 12        # 1 active + 5 bench, per player, both players

# Feature widths (kept stable so model.py can hard-code input dims)
ENTITY_NUM_FEATS = (
    6                      # hp, maxHp, hp_frac, n_energy, n_tools, appearThisTurn
    + NUM_ENERGY_TYPES     # per-type energy counts
    + 5                    # status flags (only meaningful for active)
    + 2                    # is_active, is_mine
)
OPTION_NUM_FEATS = (
    NUM_OPTION_TYPES       # option type one-hot
    + NUM_AREA_TYPES       # area one-hot
    + NUM_AREA_TYPES       # inPlayArea one-hot
    + 6                    # index, inPlayIndex, count, number, attack dmg, attack n-energy (normalized)
    + 1                    # has_card_id flag
)
GLOBAL_NUM_FEATS = (
    8                      # turn, turnActionCount, prizes_me, prizes_opp, hand_me, hand_opp, deck_me, deck_opp (normalized)
    + 5                    # supporterPlayed, stadiumPlayed, energyAttached, retreated, is_first
    + NUM_SELECT_TYPES     # select type one-hot
    + NUM_SELECT_CTX       # select context one-hot
    + 3                    # minCount, maxCount, remainEnergyCost (normalized)
)


def _onehot(idx, n):
    v = np.zeros(n, dtype=np.float32)
    if idx is not None and 0 <= int(idx) < n:
        v[int(idx)] = 1.0
    return v


def _energy_counts(energies):
    """list[EnergyType] -> 12-dim count vector."""
    v = np.zeros(NUM_ENERGY_TYPES, dtype=np.float32)
    for e in (energies or []):
        if e is not None and 0 <= int(e) < NUM_ENERGY_TYPES:
            v[int(e)] += 1.0
    return v


def _pokemon_feats(pkmn, is_active, is_mine, status_flags):
    """One board Pokemon -> (feature_vector, card_id)."""
    if pkmn is None:
        return np.zeros(ENTITY_NUM_FEATS, dtype=np.float32), 0
    hp     = float(pkmn.get("hp", 0) or 0)
    max_hp = float(pkmn.get("maxHp", 0) or 0)
    energies = pkmn.get("energies", []) or []
    tools    = pkmn.get("tools", []) or []
    base = np.array([
        hp / 300.0,
        max_hp / 300.0,
        (hp / max_hp) if max_hp > 0 else 0.0,
        len(energies) / 6.0,
        len(tools) / 2.0,
        1.0 if pkmn.get("appearThisTurn") else 0.0,
    ], dtype=np.float32)
    en = _energy_counts(energies)
    status = np.array(status_flags if is_active else [0, 0, 0, 0, 0], dtype=np.float32)
    tail = np.array([1.0 if is_active else 0.0, 1.0 if is_mine else 0.0], dtype=np.float32)
    feats = np.concatenate([base, en, status, tail]).astype(np.float32)
    return feats, int(pkmn.get("id", 0) or 0)


def _player_status(ps):
    return [
        1.0 if ps.get("poisoned")  else 0.0,
        1.0 if ps.get("burned")    else 0.0,
        1.0 if ps.get("asleep")    else 0.0,
        1.0 if ps.get("paralyzed") else 0.0,
        1.0 if ps.get("confused")  else 0.0,
    ]


def encode_entities(state, your_index):
    """Build the (active+bench) entity set for both players."""
    feats, ids = [], []
    players = (state or {}).get("players", []) or []
    for pi, ps in enumerate(players):
        if ps is None:
            continue
        is_mine = (pi == your_index)
        status = _player_status(ps)
        active_list = ps.get("active", []) or []
        active = active_list[0] if active_list else None
        f, cid = _pokemon_feats(active, True, is_mine, status)
        feats.append(f); ids.append(cid)
        for b in (ps.get("bench", []) or []):
            f, cid = _pokemon_feats(b, False, is_mine, status)
            feats.append(f); ids.append(cid)
    if not feats:  # deck-selection phase or empty board
        feats = [np.zeros(ENTITY_NUM_FEATS, dtype=np.float32)]
        ids = [0]
    E = len(feats)
    pad = MAX_ENTITIES * 2 - E
    if pad > 0:
        feats += [np.zeros(ENTITY_NUM_FEATS, dtype=np.float32)] * pad
        ids += [0] * pad
        mask = np.array([1.0] * E + [0.0] * pad, dtype=np.float32)
    else:
        feats, ids = feats[:MAX_ENTITIES * 2], ids[:MAX_ENTITIES * 2]
        mask = np.ones(MAX_ENTITIES * 2, dtype=np.float32)
    return np.stack(feats), np.array(ids, dtype=np.int64), mask


def encode_option(opt, attack_lookup):
    """One Option -> (feature_vector, referenced_card_id)."""
    otype = opt.get("type")
    feats = [_onehot(otype, NUM_OPTION_TYPES),
             _onehot(opt.get("area"), NUM_AREA_TYPES),
             _onehot(opt.get("inPlayArea"), NUM_AREA_TYPES)]
    atk = attack_lookup.get(opt.get("attackId")) if attack_lookup else None
    scalars = np.array([
        (opt.get("index") or 0) / 10.0,
        (opt.get("inPlayIndex") or 0) / 6.0,
        (opt.get("count") or 0) / 6.0,
        (opt.get("number") or 0) / 10.0,
        (atk.get("damage", 0) / 300.0) if atk else 0.0,
        (len(atk.get("energies", [])) / 6.0) if atk else 0.0,
    ], dtype=np.float32)
    has_card = np.array([1.0 if opt.get("cardId") else 0.0], dtype=np.float32)
    vec = np.concatenate(feats + [scalars, has_card]).astype(np.float32)
    return vec, int(opt.get("cardId") or 0)


def encode_globals(state, select):
    state = state or {}
    select = select or {}
    yi = state.get("yourIndex", 0) or 0
    players = state.get("players", []) or [{}, {}]
    me = players[yi] if len(players) > yi else {}
    opp = players[1 - yi] if len(players) > 1 else {}

    def prizes_left(ps): return len(ps.get("prize", []) or [])
    g = np.array([
        (state.get("turn", 0) or 0) / 40.0,
        (state.get("turnActionCount", 0) or 0) / 20.0,
        prizes_left(me) / 6.0,
        prizes_left(opp) / 6.0,
        (me.get("handCount", 0) or 0) / 20.0,
        (opp.get("handCount", 0) or 0) / 20.0,
        (me.get("deckCount", 0) or 0) / 60.0,
        (opp.get("deckCount", 0) or 0) / 60.0,
    ], dtype=np.float32)
    flags = np.array([
        1.0 if state.get("supporterPlayed") else 0.0,
        1.0 if state.get("stadiumPlayed")   else 0.0,
        1.0 if state.get("energyAttached")  else 0.0,
        1.0 if state.get("retreated")       else 0.0,
        1.0 if state.get("firstPlayer") == yi else 0.0,
    ], dtype=np.float32)
    st_type = _onehot(select.get("type"), NUM_SELECT_TYPES)
    st_ctx  = _onehot(select.get("context"), NUM_SELECT_CTX)
    counts = np.array([
        (select.get("minCount", 0) or 0) / 6.0,
        (select.get("maxCount", 0) or 0) / 6.0,
        (select.get("remainEnergyCost", 0) or 0) / 6.0,
    ], dtype=np.float32)
    return np.concatenate([g, flags, st_type, st_ctx, counts]).astype(np.float32)


def encode_observation(obs_dict, attack_lookup=None):
    """
    Full featurization. Returns a dict of numpy arrays, or None during the
    deck-selection phase (select is None) where the agent just returns a deck.
    """
    select = obs_dict.get("select")
    if select is None:
        return None
    state = obs_dict.get("current") or {}
    your_index = state.get("yourIndex", 0) or 0

    ent_feats, ent_ids, ent_mask = encode_entities(state, your_index)
    options = select.get("option", []) or []
    opt_feats, opt_ids = [], []
    for o in options:
        f, cid = encode_option(o, attack_lookup or {})
        opt_feats.append(f); opt_ids.append(cid)
    if not opt_feats:  # should not happen, but stay safe
        opt_feats = [np.zeros(OPTION_NUM_FEATS, dtype=np.float32)]
        opt_ids = [0]
    return {
        "entity_feats": ent_feats,                       # (2*MAX_ENTITIES, ENTITY_NUM_FEATS)
        "entity_ids":   ent_ids,                         # (2*MAX_ENTITIES,)
        "entity_mask":  ent_mask,                        # (2*MAX_ENTITIES,)
        "option_feats": np.stack(opt_feats),             # (O, OPTION_NUM_FEATS)
        "option_ids":   np.array(opt_ids, dtype=np.int64),  # (O,)
        "globals":      encode_globals(state, select),   # (GLOBAL_NUM_FEATS,)
        "min_count":    int(select.get("minCount", 1) or 0),
        "max_count":    int(select.get("maxCount", 1) or 1),
    }


if __name__ == "__main__":
    # tiny smoke test on a hand-built obs that matches the schema
    obs = {
        "select": {
            "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
            "remainEnergyCost": 0,
            "option": [
                {"type": 7, "index": 0},                 # PLAY hand[0]
                {"type": 13, "attackId": 5},             # ATTACK
                {"type": 14},                            # END
            ],
        },
        "current": {
            "turn": 3, "turnActionCount": 1, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "energyAttached": False,
            "players": [
                {"active": [{"id": 278, "hp": 120, "maxHp": 120,
                             "energies": [4, 4], "tools": []}],
                 "bench": [{"id": 7, "hp": 60, "maxHp": 70, "energies": [], "tools": []}],
                 "prize": [None] * 6, "handCount": 5, "deckCount": 47,
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
                {"active": [{"id": 99, "hp": 90, "maxHp": 130,
                             "energies": [3], "tools": []}],
                 "bench": [], "prize": [None] * 5, "handCount": 4, "deckCount": 50,
                 "poisoned": False, "burned": False, "asleep": False,
                 "paralyzed": False, "confused": False},
            ],
        },
    }
    enc = encode_observation(obs, attack_lookup={5: {"damage": 230, "energies": [4, 4, 0]}})
    for k, v in enc.items():
        shape = getattr(v, "shape", v)
        print(f"{k:14s} {shape}")
    assert enc["option_feats"].shape == (3, OPTION_NUM_FEATS)
    assert enc["entity_feats"].shape == (2 * MAX_ENTITIES, ENTITY_NUM_FEATS)
    assert enc["globals"].shape == (GLOBAL_NUM_FEATS,)
    print("features.py: OK")
