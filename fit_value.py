"""
fit_value.py — sharpen the value head without retraining the whole net.

Premise (see selfplay_rl): the value head is ALREADY trained on game outcomes
(z in {0, .5, 1}) at equal weight with the policy, one epoch per RL iteration.
That single-Bernoulli outcome is a high-variance target. This script improves the
head cheaply, in minutes, on data you already produced:

  * targets        — blend the outcome z with the SEARCH-backed root value (sval),
                     a much lower-variance estimate captured for free during self-play.
  * frozen body    — only the value-head MLP is trained; the encoder, embeddings, and
                     policy head are frozen, so the policy is provably unchanged.
  * calibration    — a 1-parameter temperature is fit on a held-out split and BAKED
                     into the value head's last layer (no runtime change needed).

It is NOT a from-scratch distillation run: no new rollouts, no hours of training.
Gate the result with the existing arena A/B (ISMCTS --leaf value vs --leaf rollout).

Usage:
    python fit_value.py --data valuedata/ --warm rl_model.pt --out rl_model.value.pt \
        --epochs 8 --target-blend 0.5
"""
from __future__ import annotations
import argparse, glob, os, pickle
import numpy as np
import torch

import model as M
from train_bc import collate, device


def load_samples(data_dir):
    samples = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        try:
            with open(f, "rb") as fh:
                samples += pickle.load(fh)
        except Exception as e:
            print(f"  skip {f}: {e}")
    return samples


def value_targets(samples, blend):
    """Per-sample (enc, target) where target blends outcome z with search value sval."""
    out = []
    used_sval = 0
    for s in samples:
        enc, z = s[0], s[2]
        if z is None:
            continue
        sval = s[3] if len(s) > 3 else None
        if sval is not None:
            t = (1.0 - blend) * float(z) + blend * float(sval); used_sval += 1
        else:
            t = float(z)
        out.append((enc, float(np.clip(t, 0.0, 1.0))))
    return out, used_sval


def value_mse(net, data, dev, bs=256):
    net.eval(); tot = n = 0.0
    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            X, _ = collate([{**enc, "target": [0]} for enc, _ in batch])
            X = {k: v.to(dev) for k, v in X.items()}
            _, value = net(X)
            t = torch.tensor([y for _, y in batch], dtype=torch.float32, device=dev)
            tot += ((value - t) ** 2).sum().item(); n += len(batch)
    return tot / max(n, 1)


def holdout_logits(net, data, dev, bs=256):
    """Return (pre-sigmoid value logits, targets) over a dataset, for calibration."""
    net.eval(); logits = []; tgts = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            X, _ = collate([{**enc, "target": [0]} for enc, _ in batch])
            X = {k: v.to(dev) for k, v in X.items()}
            board, gvec = net.encode_board(X["entity_feats"], X["entity_ids"],
                                           X["entity_mask"], X["globals"])
            logit = net.value_head(torch.cat([board, gvec], -1)).squeeze(-1)
            logits.append(logit.cpu()); tgts += [y for _, y in batch]
    return torch.cat(logits).numpy(), np.asarray(tgts, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of dumped self-play samples (selfplay_rl --dump-samples)")
    ap.add_argument("--warm", default="rl_model.pt", help="net whose value head we sharpen")
    ap.add_argument("--out", default="", help="output net (default: <warm>.value.pt)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-blend", type=float, default=0.5,
                    help="weight on the search value vs the outcome (0=outcome only, 1=search only)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--num-ids", type=int, default=1268)
    a = ap.parse_args()
    out = a.out or (os.path.splitext(a.warm)[0] + ".value.pt")
    dev = device()

    samples = load_samples(a.data)
    data, used = value_targets(samples, a.target_blend)
    if not data:
        raise SystemExit(f"no usable samples in {a.data}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(data))
    n_val = max(1, int(len(data) * a.val_frac))
    val = [data[i] for i in idx[:n_val]]; trn = [data[i] for i in idx[n_val:]]
    print(f"samples: {len(data)} ({used} with search value, blend={a.target_blend}) | "
          f"train {len(trn)} / holdout {len(val)}")

    net = M.from_meta(dev, warm=a.warm, num_ids_default=a.num_ids)
    if os.path.exists(a.warm):
        print(f"warm: {a.warm}")

    # freeze everything except the value head
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith("value_head")
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f"trainable params (value head only): {sum(p.numel() for p in trainable)}")

    # snapshot policy logits on a few holdout positions to prove the policy is untouched
    probe = [enc for enc, _ in val[:32]]
    def policy_logits(encs):
        net.eval()
        with torch.no_grad():
            X, _ = collate([{**e, "target": [0]} for e in encs]); X = {k: v.to(dev) for k, v in X.items()}
            lg, _ = net(X); return lg.cpu().numpy()
    pol_before = policy_logits(probe)

    before = value_mse(net, val, dev)
    opt = torch.optim.Adam(trainable, lr=a.lr)
    import random as _r
    for ep in range(a.epochs):
        net.train(); _r.shuffle(trn); tot = n = 0.0
        for i in range(0, len(trn), a.bs):
            batch = trn[i:i + a.bs]
            X, _ = collate([{**enc, "target": [0]} for enc, _ in batch]); X = {k: v.to(dev) for k, v in X.items()}
            _, value = net(X)
            t = torch.tensor([y for _, y in batch], dtype=torch.float32, device=dev)
            loss = ((value - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(batch); n += len(batch)
        print(f"  epoch {ep+1}: train value MSE {tot/max(n,1):.4f}  | holdout {value_mse(net, val, dev):.4f}")

    # temperature calibration on the holdout, baked into the value head's last layer
    logit, tgt = holdout_logits(net, val, dev)
    def mse_T(T):
        p = 1.0 / (1.0 + np.exp(-logit / T)); return float(np.mean((p - tgt) ** 2))
    Ts = np.linspace(0.5, 3.0, 51)
    bestT = float(min(Ts, key=mse_T))
    last = net.value_head[-1]                       # Linear(d,1) -> the pre-sigmoid logit
    with torch.no_grad():
        last.weight.div_(bestT); last.bias.div_(bestT)
    after = value_mse(net, val, dev)

    pol_after = policy_logits(probe)
    pol_drift = float(np.nanmax(np.abs(pol_before - pol_after)))
    print(f"\nholdout value MSE: {before:.4f} -> {after:.4f}  (temperature T={bestT:.2f})")
    print(f"policy logits drift (must be ~0): {pol_drift:.2e}")
    assert pol_drift < 1e-5, "policy changed — body/policy were not properly frozen"

    torch.save(net.state_dict(), out)
    print(f"saved sharpened net -> {out}")
    print("NEXT: gate it — python arena.py --candidate %s --anchor %s --ismcts --leaf value "
          "--games 100  (and compare to --leaf rollout)" % (out, a.warm))


if __name__ == "__main__":
    main()
