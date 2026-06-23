"""
main.py — the submission entrypoint. THIS is the file the ladder calls.

The contract (cabt): define `agent(obs_dict) -> list[int]`.
  * During deck selection, obs["select"] is None  -> return our 60 card IDs.
  * Otherwise -> return chosen option indices, length within [minCount, maxCount].
  * It must NEVER raise and must respect the per-move time limit.

Design: a thin, paranoid shell. Whatever the inner policy does, the shell
guarantees a legal, on-time return. The policy is swappable by env var:
  POLICY=heuristic   (default; no torch needed)
  POLICY=nn          (loads model.pt; falls back to heuristic if anything fails)
"""

from __future__ import annotations
import os
import signal

# ---- load deck ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

def _load_deck():
    for path in (os.path.join(_HERE, "deck.csv"),
                 "/kaggle_simulations/agent/deck.csv", "deck.csv"):
        try:
            with open(path) as f:
                # accept whitespace- OR newline-separated IDs (matches how
                # selfplay_rl.py / import_deck.py read the same file)
                deck = [int(tok) for tok in f.read().split()]
            if len(deck) == 60:
                return deck
        except Exception:
            continue
    # last-resort legal-ish placeholder so we never crash at deck selection;
    # replace deck.csv with a real 60-card list before submitting.
    return [1] * 60

DECK = _load_deck()

# ---- option types we treat specially in the fallback -------------------------
END_OPTION = 14  # OptionType.END

# ---- policy selection --------------------------------------------------------
_POLICY = os.environ.get("POLICY", "heuristic").lower()
_nn = None  # lazy-loaded network

_CARD_DB = None          # lazily loaded evaluate.CardDB (optional)
_ATTACK_DB = None        # lazily loaded attackId -> {damage, energies} (optional)

def _load_dbs():
    global _CARD_DB, _ATTACK_DB
    if _CARD_DB is None:
        try:
            from evaluate import CardDB
            _CARD_DB = CardDB.load(os.path.join(_HERE, "capability_table.json"))
        except Exception:
            _CARD_DB = False
    if _ATTACK_DB is None:
        try:
            with open(os.path.join(_HERE, "attack_table.json")) as f:
                import json
                _ATTACK_DB = {int(k): v for k, v in json.load(f).items()}
        except Exception:
            _ATTACK_DB = {}
    return (_CARD_DB or None), _ATTACK_DB

_TRACKER = None          # deck_inference.OpponentTracker (persists across a game)
_LIBRARY = None          # deck_inference.ArchetypeLibrary

def _ensure_inference():
    global _TRACKER, _LIBRARY
    if _TRACKER is None:
        import deck_inference as DI
        _TRACKER = DI.OpponentTracker()
        # seed with our own deck as one archetype; add real archetypes from
        # replay decklists for accurate opponent prediction in non-mirror games.
        _LIBRARY = DI.ArchetypeLibrary().fit([("self", DECK)])
    return _TRACKER, _LIBRARY

def _heuristic_select(obs_dict):
    import policy_heuristic
    db, attack_db = _load_dbs()
    base = policy_heuristic.select(obs_dict, db=db, attack_db=attack_db)
    try:
        import combat
        if not combat.available():
            return base
        import deck_inference as DI
        trk, lib = _ensure_inference()
        if obs_dict.get("current"):
            trk.update(obs_dict)                      # update once per real decision
        if obs_dict.get("select") is None:
            return base
        def predictor(o):
            try:
                return DI.predict_opponent_zones(o, trk, lib, card_db=db, min_conf=0.3)
            except Exception:
                return None
        zones = predictor(obs_dict)
        plies = 2 if zones else 1                     # 2-ply only with a confident read
        return combat.refine_selection(obs_dict, DECK, db, attack_db, base,
                                       opp_predictor=predictor, plies=plies)
    except Exception:
        return base

def _nn_select(obs_dict):
    global _nn
    import features as fx
    if _nn is None:
        import torch, json
        from model import PointerPolicyValueNet
        meta = {}
        meta_path = os.path.join(_HERE, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        net = PointerPolicyValueNet(
            num_card_ids=meta.get("num_card_ids", 4096),
            d=meta.get("d", 96),
            n_heads=meta.get("n_heads", 4),
            n_layers=meta.get("n_layers", 2))
        net.load_state_dict(torch.load(os.path.join(_HERE, "model.pt"),
                                       map_location="cpu"))
        net.eval()
        _nn = net
    enc = fx.encode_observation(obs_dict)  # attack_lookup optional; add if shipped
    return _nn.act(enc)

def _inner_select(obs_dict):
    if _POLICY == "nn":
        try:
            return _nn_select(obs_dict)
        except Exception:
            return _heuristic_select(obs_dict)   # NN never takes the ladder down
    return _heuristic_select(obs_dict)

# ---- legality normalization --------------------------------------------------
def _normalize(result, select):
    options = select.get("option", []) or []
    n_opt = len(options)
    lo = int(select.get("minCount", 1) or 0)
    hi = max(int(select.get("maxCount", 1) or 1), lo)
    # keep only valid, unique indices, preserve order
    seen, clean = set(), []
    for i in (result or []):
        try:
            i = int(i)
        except Exception:
            continue
        if 0 <= i < n_opt and i not in seen:
            seen.add(i); clean.append(i)
    if len(clean) > hi:
        clean = clean[:hi]
    if len(clean) < lo:                      # pad with lowest unused indices
        for i in range(n_opt):
            if len(clean) >= lo:
                break
            if i not in seen:
                seen.add(i); clean.append(i)
    return clean

def _legal_fallback(select):
    options = select.get("option", []) or []
    lo = int(select.get("minCount", 1) or 0)
    # prefer ending the turn when that is an option (avoids action loops)
    for i, o in enumerate(options):
        if o.get("type") == END_OPTION:
            return _normalize([i], select)
    base = list(range(max(lo, 0))) if lo > 0 else ([0] if options else [])
    return _normalize(base, select)

# ---- optional hard timeout (Unix main thread) --------------------------------
_TIME_LIMIT_S = float(os.environ.get("MOVE_TIME_LIMIT", "0"))  # 0 = disabled

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

# ---- the entrypoint ----------------------------------------------------------
def agent(obs_dict) -> list[int]:
    # deck-selection phase
    select = obs_dict.get("select")
    if select is None:
        return DECK
    try:
        if _TIME_LIMIT_S > 0 and hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, _TIME_LIMIT_S)
        result = _inner_select(obs_dict)
        return _normalize(result, select) or _legal_fallback(select)
    except Exception:
        return _legal_fallback(select)
    finally:
        if _TIME_LIMIT_S > 0 and hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
