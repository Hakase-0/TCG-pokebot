"""
train_bc.py — behavioral cloning of the pointer net on a self-play/replay dataset.

Trains model.PointerPolicyValueNet to imitate the recorded choices: masked
cross-entropy over exactly the legal options present at each decision. Logs
loss + top-1 match each epoch to JSONL (for the terminal stats UI) and writes
model.pt + model_meta.json so `POLICY=nn` main.py can load it.

Runs on Apple Silicon (MPS) or CPU automatically.

Usage:
  python train_bc.py --data data/bc.pkl --epochs 10 --out model.pt
"""
from __future__ import annotations
import argparse, json, os, pickle, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import features as fx
from model import PointerPolicyValueNet
import stats


def collate(batch):
    B = len(batch)
    maxO = max(s["option_feats"].shape[0] for s in batch)
    ef = torch.tensor(np.stack([s["entity_feats"] for s in batch]), dtype=torch.float32)
    ei = torch.tensor(np.stack([s["entity_ids"] for s in batch]), dtype=torch.long)
    em = torch.tensor(np.stack([s["entity_mask"] for s in batch]), dtype=torch.float32)
    gl = torch.tensor(np.stack([s["globals"] for s in batch]), dtype=torch.float32)
    of = torch.zeros(B, maxO, fx.OPTION_NUM_FEATS)
    oi = torch.zeros(B, maxO, dtype=torch.long)
    om = torch.zeros(B, maxO)
    tgt = torch.zeros(B, maxO)
    for b, s in enumerate(batch):
        O = s["option_feats"].shape[0]
        of[b, :O] = torch.tensor(s["option_feats"])
        oi[b, :O] = torch.tensor(s["option_ids"])
        om[b, :O] = 1.0
        for t in s["target"]:
            if 0 <= t < O:
                tgt[b, t] = 1.0
        if tgt[b].sum() > 0:
            tgt[b] /= tgt[b].sum()
    return {"entity_feats": ef, "entity_ids": ei, "entity_mask": em, "globals": gl,
            "option_feats": of, "option_ids": oi, "option_mask": om}, tgt


def is_xla(dev):
    # dev may be a string ("cpu"/"cuda"/"mps") or a torch.device. The XLA device
    # stringifies to "xla:0", so a substring check covers both forms.
    return "xla" in str(dev).lower()


def xla_mark_step():
    # flush the lazy XLA graph -> materialize pending ops onto the TPU. Lazy import
    # so nothing pulls in torch_xla unless we actually committed to the TPU path.
    import torch_xla.core.xla_model as xm
    xm.mark_step()


def device():
    # opt-in TPU path: Kaggle TPU VMs ship PyTorch/XLA preinstalled (PJRT runtime).
    # Gate behind TCG_DEVICE=tpu so ordinary CPU/GPU/MPS runs never import torch_xla,
    # and so self-play spawn-workers (which don't set the env) stay off XLA entirely.
    # Any failure to reach the TPU falls through to the normal probe below, so a
    # "tpu" run on a machine without one degrades to CPU instead of crashing.
    if os.environ.get("TCG_DEVICE", "").lower() == "tpu":
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
            print(f"[device] TCG_DEVICE=tpu -> using XLA device: {dev}", flush=True)
            return dev
        except Exception as e:
            print(f"[device] TCG_DEVICE=tpu but XLA init failed ({type(e).__name__}: "
                  f"{str(e).splitlines()[0][:80]}); falling back", flush=True)
    # availability != usability: some Kaggle/Colab images ship a torch whose
    # compiled archs don't match the assigned GPU, so torch.cuda.is_available()
    # is True but every kernel raises cudaErrorNoKernelImageForDevice. Probe a
    # real kernel launch (sync forces async errors to surface) before committing.
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda").add_(1.0); torch.cuda.synchronize()
            return "cuda"
        except Exception as e:
            print(f"[device] CUDA present but unusable ({type(e).__name__}: "
                  f"{str(e).splitlines()[0][:80]}); falling back to CPU", flush=True)
    if torch.backends.mps.is_available():    # Apple Silicon
        return "mps"
    return "cpu"


def train(data, epochs, out, lr=1e-3, bs=64, log="logs/train.jsonl",
          d=192, n_heads=6, n_layers=3):
    samples = pickle.load(open(data, "rb"))
    if not samples:
        print("no samples"); return
    maxid = 1268
    for s in samples:
        if len(s["option_ids"]):
            maxid = max(maxid, int(s["option_ids"].max()) + 1)
        maxid = max(maxid, int(s["entity_ids"].max()) + 1)
    dev = device()
    net = PointerPolicyValueNet(num_card_ids=maxid, d=d, n_heads=n_heads, n_layers=n_layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(samples)
    nparams = sum(p.numel() for p in net.parameters())
    print(f"training on {n} samples, {epochs} epochs, device={dev}, vocab={maxid} | "
          f"net d={d} heads={n_heads} layers={n_layers} ({nparams/1e6:.2f}M params)", flush=True)
    for ep in range(epochs):
        np.random.shuffle(samples)
        lsum = correct = seen = 0
        net.train()
        for i in range(0, n, bs):
            batch = samples[i:i + bs]
            X, tgt = collate(batch)
            X = {k: v.to(dev) for k, v in X.items()}; tgt = tgt.to(dev)
            logits, _ = net(X)
            masked = logits.masked_fill(X["option_mask"] < 0.5, -1e9)
            logp = torch.log_softmax(masked, dim=1)
            loss = -(tgt * logp).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            lsum += loss.item() * len(batch); seen += len(batch)
            pred = masked.argmax(1)
            correct += sum(1 for b in range(len(batch)) if tgt[b, pred[b]] > 0)
        avg, acc = lsum / seen, correct / seen
        stats.log(log, event="bc_epoch", epoch=ep + 1, loss=round(avg, 4), top1_match=round(acc, 3))
        print(f"epoch {ep+1:>3}: loss {avg:.4f}  top1-match {acc:.3f}")
    torch.save(net.state_dict(), out)
    json.dump({"num_card_ids": maxid, "d": d, "n_heads": n_heads, "n_layers": n_layers},
              open("model_meta.json", "w"))
    print(f"saved {out} and model_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bc.pkl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--log", default="logs/train.jsonl")
    ap.add_argument("--d", type=int, default=192, help="model dim (agnostic net is bigger than the d=96 specialist)")
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=3)
    a = ap.parse_args()
    train(a.data, a.epochs, a.out, a.lr, a.bs, a.log, d=a.d, n_heads=a.n_heads, n_layers=a.n_layers)
