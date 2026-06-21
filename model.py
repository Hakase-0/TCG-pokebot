"""
model.py — the policy/value network.

Architecture: a *candidate-scoring (pointer) net*. The engine hands us a list
of legal options each turn; we score each one given the board, then softmax
over exactly the options present. This sidesteps the variable / huge action
space entirely and generalizes across decks, because options reference cards
through a shared embedding table rather than through fixed action indices.

  card id ──► Embedding ─┐
                         ├─► entity tokens ─► set-encoder ─► board vector ─┐
  board feats ───────────┘                                                ├─► option scorer ─► logit per option ─► policy
  option feats + option-card embedding ──────────────────────────────────┘
  board vector + globals ─────────────────────────────────────────────────► value head ─► win prob

Kept deliberately small (d=96, 2 attention layers) so CPU inference at
submission time stays well inside the per-move limit.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

import features as fx


class PointerPolicyValueNet(nn.Module):
    def __init__(self, num_card_ids: int = 4096, d: int = 96, n_heads: int = 4,
                 n_layers: int = 2, card_feat_dim: int = 0):
        super().__init__()
        self.d = d
        # id 0 is reserved for pad / unknown / empty slot
        self.card_embed = nn.Embedding(num_card_ids, d, padding_idx=0)

        # optional static-card-feature MLP (hp, type, stage, ex, retreat, ...)
        # pass a precomputed (num_card_ids, card_feat_dim) matrix via set_card_feats()
        self.card_feat_dim = card_feat_dim
        if card_feat_dim > 0:
            self.card_feat_mlp = nn.Sequential(
                nn.Linear(card_feat_dim, d), nn.GELU(), nn.Linear(d, d))
            self.register_buffer("card_feat_table",
                                 torch.zeros(num_card_ids, card_feat_dim))

        self.entity_proj = nn.Linear(fx.ENTITY_NUM_FEATS + d, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            batch_first=True, activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.global_proj = nn.Sequential(
            nn.Linear(fx.GLOBAL_NUM_FEATS, d), nn.GELU())

        # option scorer: [option_feats ⊕ option_card_embed ⊕ board ⊕ globals] -> logit
        self.option_scorer = nn.Sequential(
            nn.Linear(fx.OPTION_NUM_FEATS + d + d + d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, d), nn.GELU(),
            nn.Linear(d, 1))

        self.value_head = nn.Sequential(
            nn.Linear(d + d, d), nn.GELU(), nn.Linear(d, 1))

    def set_card_feats(self, table: torch.Tensor):
        assert self.card_feat_dim > 0
        with torch.no_grad():
            self.card_feat_table.copy_(table)

    def _card_vec(self, ids: torch.Tensor) -> torch.Tensor:
        v = self.card_embed(ids)
        if self.card_feat_dim > 0:
            v = v + self.card_feat_mlp(self.card_feat_table[ids])
        return v

    def encode_board(self, entity_feats, entity_ids, entity_mask, globals_):
        """Returns (board_vec [B,d], global_vec [B,d])."""
        ent_card = self._card_vec(entity_ids)                    # [B,E,d]
        tok = torch.cat([entity_feats, ent_card], dim=-1)        # [B,E,F+d]
        tok = self.entity_proj(tok)                              # [B,E,d]
        key_pad = entity_mask < 0.5                              # True = pad
        enc = self.encoder(tok, src_key_padding_mask=key_pad)    # [B,E,d]
        m = entity_mask.unsqueeze(-1)
        board = (enc * m).sum(1) / m.sum(1).clamp_min(1.0)       # masked mean
        gvec = self.global_proj(globals_)                        # [B,d]
        return board, gvec

    def forward(self, batch):
        """
        batch keys (tensors, batch-first):
          entity_feats [B,E,Fe], entity_ids [B,E], entity_mask [B,E],
          globals [B,Fg], option_feats [B,O,Fo], option_ids [B,O],
          option_mask [B,O]  (1 = real option, 0 = pad)
        Returns: option_logits [B,O] (pad positions set to -inf), value [B]
        """
        board, gvec = self.encode_board(
            batch["entity_feats"], batch["entity_ids"],
            batch["entity_mask"], batch["globals"])

        opt_card = self._card_vec(batch["option_ids"])           # [B,O,d]
        O = batch["option_feats"].shape[1]
        ctx = torch.cat([board, gvec], dim=-1).unsqueeze(1).expand(-1, O, -1)
        scorer_in = torch.cat([batch["option_feats"], opt_card,
                               ctx[..., :self.d], ctx[..., self.d:]], dim=-1)
        logits = self.option_scorer(scorer_in).squeeze(-1)       # [B,O]
        if "option_mask" in batch:
            logits = logits.masked_fill(batch["option_mask"] < 0.5, float("-inf"))
        value = torch.sigmoid(self.value_head(torch.cat([board, gvec], -1))).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, enc: dict, greedy: bool = True) -> list[int]:
        """
        enc: output of features.encode_observation (single, unbatched).
        Returns chosen option indices respecting min/max count.
        """
        self.eval()
        def t(x, dt=torch.float32): return torch.as_tensor(x, dtype=dt).unsqueeze(0)
        batch = {
            "entity_feats": t(enc["entity_feats"]),
            "entity_ids":   t(enc["entity_ids"], torch.long),
            "entity_mask":  t(enc["entity_mask"]),
            "globals":      t(enc["globals"]),
            "option_feats": t(enc["option_feats"]),
            "option_ids":   t(enc["option_ids"], torch.long),
        }
        logits, _ = self.forward(batch)
        logits = logits[0]                                       # [O]
        order = torch.argsort(logits, descending=True).tolist()
        lo, hi = enc["min_count"], max(enc["max_count"], enc["min_count"])
        if lo == hi:                       # exact-count selection
            return sorted(order[:hi])
        # "choose between lo and hi": take the confident ones (prob>0.5),
        # clamped into [lo, hi]; always legal.
        probs = torch.sigmoid(logits)
        chosen = [i for i in order if probs[i] > 0.5][:hi]
        if len(chosen) < lo:
            chosen = order[:lo]
        return sorted(chosen[:hi])


def from_meta(dev="cpu", meta_path="model_meta.json", warm=None, num_ids_default=1268):
    """Single source of truth for net dims: build a net from model_meta.json
    (num_card_ids, d, n_heads, n_layers) so every script agrees on architecture,
    and a larger agnostic net loads everywhere without shape mismatches."""
    import json, os, torch
    m = {}
    if os.path.exists(meta_path):
        try:
            m = json.load(open(meta_path))
        except Exception:
            m = {}
    net = PointerPolicyValueNet(
        num_card_ids=m.get("num_card_ids", num_ids_default),
        d=m.get("d", 96), n_heads=m.get("n_heads", 4), n_layers=m.get("n_layers", 2)).to(dev)
    if warm and os.path.exists(warm):
        net.load_state_dict(torch.load(warm, map_location=dev))
    return net


if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(0)
    net = PointerPolicyValueNet(num_card_ids=4096, d=96)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # build a fake encoded obs via the real featurizer
    obs = {
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "remainEnergyCost": 0,
                   "option": [{"type": 7, "index": 0},
                              {"type": 13, "attackId": 5},
                              {"type": 14}]},
        "current": {"turn": 3, "turnActionCount": 1, "yourIndex": 0,
                    "firstPlayer": 0,
                    "players": [
                        {"active": [{"id": 278, "hp": 120, "maxHp": 120,
                                     "energies": [4, 4], "tools": []}],
                         "bench": [], "prize": [None]*6, "handCount": 5,
                         "deckCount": 47},
                        {"active": [{"id": 99, "hp": 90, "maxHp": 130,
                                     "energies": [3], "tools": []}],
                         "bench": [], "prize": [None]*5, "handCount": 4,
                         "deckCount": 50}]},
    }
    enc = fx.encode_observation(obs, attack_lookup={5: {"damage": 230, "energies": [4,4,0]}})
    choice = net.act(enc)
    print("chosen option indices:", choice)
    assert all(0 <= i < len(obs["select"]["option"]) for i in choice)
    print("model.py: OK")
