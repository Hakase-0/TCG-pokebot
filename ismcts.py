"""
ismcts.py — information-set / PIMC search over the cabt engine (Stage 1 core).

PTCG is imperfect-information, and the engine's search API is per-determinization
and non-caching (re-stepping an action makes a fresh state), so we use
Perfect-Information Monte Carlo with MCTS: sample K opponent worlds, build one
net-guided MCTS tree per world (each node stores its engine searchId and is only
expanded once per edge), and aggregate the root visit counts across worlds into
one improved policy.

Net guidance (AlphaZero-style):
  * policy head -> prior P(s,a) at each node (PUCT)
  * value head  -> leaf evaluation (no full rollouts)
All values are kept in OUR seat's perspective in [0,1]; at opponent nodes the
value head reports the opponent's win-prob, so we use 1 - v.

This file is search only — no training. `search()` returns (visit_policy over the
root's options, root value estimate, chosen option index).
"""
from __future__ import annotations
import math, random
from collections import Counter

import numpy as np
import torch

try:
    from cg import api as _api
except Exception:
    try:
        import api as _api
    except Exception:
        _api = None

import combat                      # reuse _obs_to_dict, _visible_my_ids, _fillers
import policy_heuristic as _H
from evaluate import evaluate_state
from train_bc import collate

ATTACK, END, MAIN = 13, 14, 0


def available():
    return _api is not None


# ---------- net inference ----------
def _infer(net, obs, attack_db, dev):
    """Return (policy over options [O], value scalar v, O). v is player-to-move win-prob."""
    import features as fx
    enc = fx.encode_observation(obs, attack_lookup=attack_db)
    if enc is None:
        sel = obs.get("select") or {}
        O = len(sel.get("option", []))
        return (np.ones(O) / max(O, 1) if O else np.array([])), 0.5, O
    X, _ = collate([{**enc, "target": [0]}])
    X = {k: v.to(dev) for k, v in X.items()}
    with torch.no_grad():
        logits, value = net(X)
    m = X["option_mask"][0]
    p = torch.softmax(logits[0].masked_fill(m < 0.5, -1e9), 0).cpu().numpy()
    O = int(m.sum().item())
    return p[:O], float(value[0]), O


# ---------- determinization (randomized for world diversity) ----------
def _determinize(obs, deck60, predictor, rng):
    bp, be = combat._fillers()
    st = obs["current"]; yi = st["yourIndex"]
    me, opp = st["players"][yi], st["players"][1 - yi]
    rem = list((Counter(deck60) - Counter(combat._visible_my_ids(st, yi))).elements())
    rng.shuffle(rem)                                   # diversity across worlds
    dc, pc = me["deckCount"], len(me.get("prize") or [])
    while len(rem) < dc + pc:
        rem.append(be)
    your_deck, your_prize = rem[:dc], rem[dc:dc + pc]
    zones = None
    if predictor is not None:
        try:
            zones = predictor(obs)
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


# ---------- tree ----------
class _Node:
    __slots__ = ("sid", "obs", "to_move", "actions", "terminal", "tv",
                 "P", "N", "W", "children", "expanded")

    def __init__(self, sid, obs, our_seat):
        self.sid = sid
        self.obs = obs
        cur = obs.get("current") or {}
        self.to_move = cur.get("yourIndex", our_seat)
        res = cur.get("result", -1)
        self.terminal = res is not None and res >= 0
        self.tv = (1.0 if res == our_seat else (0.5 if res == 2 else 0.0)) if self.terminal else None
        self.expanded = False
        self.actions = []           # list of engine selections (each a list[int])
        self.P = None               # prior over actions
        self.N = None
        self.W = None
        self.children = {}          # action_index -> _Node


def _legal_actions(obs, db, attack_db):
    """Single-select -> one action per option; multi-select -> a single forced (heuristic) action."""
    sel = obs.get("select") or {}
    O = len(sel.get("option", []))
    if O == 0:
        return []
    if (sel.get("minCount", 1) or 0) <= 1:
        return [[i] for i in range(O)]
    return [list(_H.select(obs, db=db, attack_db=attack_db))]   # don't branch combinations


class _Tree:
    def __init__(self, net, dev, db, attack_db, our_seat, c_puct, leaf_eval="value", rollout_steps=16):
        self.net, self.dev, self.db, self.atk = net, dev, db, attack_db
        self.our_seat, self.c = our_seat, c_puct
        self.leaf_eval, self.rollout_steps = leaf_eval, rollout_steps

    def _our_value(self, node):
        if node.terminal:
            return node.tv
        _, v, _ = _infer(self.net, node.obs, self.atk, self.dev)
        return v if node.to_move == self.our_seat else (1.0 - v)

    def _rollout_value(self, node):
        """Engine-oracle leaf eval: step the heuristic forward from this node's
        searchId to a horizon, then score with evaluate_state. More reliable than
        a BC-only value head because the engine actually resolves damage/KOs/prizes."""
        if node.terminal:
            return node.tv
        sid, obs, steps = node.sid, node.obs, 0
        while steps < self.rollout_steps:
            cur = obs.get("current") or {}
            res = cur.get("result", -1)
            if res is not None and res >= 0:
                return 1.0 if res == self.our_seat else (0.5 if res == 2 else 0.0)
            sel = obs.get("select")
            if sel is None:
                break
            try:
                state = _api.search_step(sid, _H.select(obs, db=self.db, attack_db=self.atk))
            except Exception:
                break
            sid = state.searchId
            obs = combat._obs_to_dict(state.observation)
            steps += 1
        cur = obs.get("current") or {}
        return (evaluate_state(cur, self.our_seat, self.db) + 1.0) / 2.0   # [-1,1] -> [0,1]

    def _leaf(self, node):
        return self._rollout_value(node) if self.leaf_eval == "rollout" else self._our_value(node)

    def _expand(self, node):
        node.actions = _legal_actions(node.obs, self.db, self.atk)
        n = len(node.actions)
        p, _, O = _infer(self.net, node.obs, self.atk, self.dev)
        if n == O and O > 0:
            node.P = np.asarray(p, dtype=np.float64)
        else:                                          # multi-select / mismatch -> uniform
            node.P = np.ones(n) / max(n, 1)
        node.N = np.zeros(n); node.W = np.zeros(n)
        node.expanded = True

    def _select(self, node):
        tot = node.N.sum()
        best, best_a = -1e18, 0
        for a in range(len(node.actions)):
            q = (node.W[a] / node.N[a]) if node.N[a] > 0 else 0.5
            exploit = q if node.to_move == self.our_seat else (1.0 - q)   # opp minimizes our win
            u = self.c * node.P[a] * math.sqrt(tot + 1e-8) / (1 + node.N[a])
            s = exploit + u
            if s > best:
                best, best_a = s, a
        return best_a

    def simulate(self, root):
        node, path = root, []
        while node.expanded and not node.terminal:
            a = self._select(node)
            path.append((node, a))
            if a in node.children:
                node = node.children[a]
            else:
                try:
                    child_state = _api.search_step(node.sid, node.actions[a])
                    cobs = combat._obs_to_dict(child_state.observation)
                    child = _Node(child_state.searchId, cobs, self.our_seat)
                except Exception:
                    child = node                       # step failed -> treat as leaf at node
                node.children[a] = child
                node = child
                break
        if not node.terminal and not node.expanded:
            self._expand(node)
        v = self._leaf(node)
        for (nd, a) in path:
            nd.N[a] += 1; nd.W[a] += v
        return v


def search(obs, deck60, db, attack_db, net, dev, predictor=None,
           n_worlds=8, n_sims=48, c_puct=1.5, seed=None, leaf_eval="value", rollout_steps=16):
    """
    Returns (visit_policy[O] over the root's options, root_value, chosen_index).
    leaf_eval="value": net value head at leaves. leaf_eval="rollout": engine-oracle
    heuristic rollout + evaluate_state (slower, but a stronger compass when the
    value head is weak). Degrades to (None, None, None) if the engine is unavailable.
    """
    if _api is None:
        return None, None, None
    sel = obs.get("select") or {}
    O = len(sel.get("option", []))
    our_seat = (obs.get("current") or {}).get("yourIndex", 0)
    if O == 0:
        return None, None, None
    rng = random.Random(seed)
    agg_visits = np.zeros(O)
    agg_value, worlds_done = 0.0, 0
    multi = (sel.get("minCount", 1) or 0) > 1

    for _ in range(n_worlds):
        try:
            oc = _api.to_observation_class(obs)
            root_state = _api.search_begin(oc, *_determinize(obs, deck60, predictor, rng))
        except Exception:
            try: _api.search_end()
            except Exception: pass
            continue
        try:
            root = _Node(root_state.searchId, combat._obs_to_dict(root_state.observation), our_seat)
            tree = _Tree(net, dev, db, attack_db, our_seat, c_puct, leaf_eval, rollout_steps)
            tree._expand(root)
            for _s in range(n_sims):
                tree.simulate(root)
            # map this world's root action-visits back onto option indices
            if not multi:
                for a, actsel in enumerate(root.actions):
                    if actsel:
                        agg_visits[actsel[0]] += root.N[a]
            else:
                best_a = int(np.argmax(root.N)) if len(root.N) else 0
                if root.actions:
                    for i in root.actions[best_a]:
                        if i < O:
                            agg_visits[i] += 1
            agg_value += (root.W.sum() / max(root.N.sum(), 1))
            worlds_done += 1
        finally:
            try: _api.search_end()
            except Exception: pass

    if worlds_done == 0 or agg_visits.sum() == 0:
        return None, None, None
    policy = agg_visits / agg_visits.sum()
    chosen = int(np.argmax(agg_visits))
    return policy, agg_value / worlds_done, chosen
