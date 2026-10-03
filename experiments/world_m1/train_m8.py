"""World M8 milestone: parameter-efficiency curve (the "smallest world model
that works" claim, PLAN §M8 / LeWorldModel target).

Instead of a single config, this traces the M1 objective's probe gain as the
liquid encoder widens (d = 64 / 128 / 256) — evidence that the industrial
world-model inductive bias holds at small scale ("2-loss, ~15M-class" spirit;
here we test the low end of the curve on the physics suite).

Success (pre-registered): the mean probe gain stays > +70% at EVERY width
(spring, 3 seeds) — i.e. the gain is not a small-model artefact and the
smallest configs already work. Report params alongside gains.

Run:
    python -m experiments.world_m1.train_m8 --steps 8000 --seeds 3
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.objectives import masked_latent_prediction_loss, mask_spans, sigreg

from .data import make_dataset
from .encoder import LiquidEncoder
from .train import LatentPredictor, latent_std


def train_one(kind: str, seed: int, steps: int, d: int, T: int = 64,
              batch: int = 64, sigreg_weight: float = 0.01) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = LiquidEncoder(2 if kind == "spring" else 4, d, n_layers=2).to(dev)
    pred = LatentPredictor(d).to(dev)
    params = sum(p.numel() for p in enc.parameters()) + \
        sum(p.numel() for p in pred.parameters())
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        traj, _ = make_dataset(kind, batch, T, seed=seed * 10000 + step)
        traj = traj.to(dev)
        z = enc(traj)
        keep = mask_spans(T, 0.5, 6, g)
        loss = masked_latent_prediction_loss(z, keep, pred)
        loss = loss + sigreg_weight * sigreg(z)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step()
        if (step + 1) % 4000 == 0:
            print(f"    d={d} seed {seed} step {step+1} loss {loss.item():.4f} "
                  f"std {latent_std(z.detach()):.3f}", flush=True)

    enc.eval()
    X, Y = make_dataset(kind, 4096, T + 1, seed=777 + seed)
    X, Y = X.to(dev), Y.to(dev)
    with torch.no_grad():
        Z = enc(X)
    base = float(((Y[:, :-1] - X[:, :-1]) ** 2).mean())
    probe = nn.Sequential(nn.Linear(d, 128), nn.ReLU(),
                          nn.Linear(128, Y.shape[-1])).to(dev)
    optp = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(1500):
        idx = torch.randperm(Z.shape[0])[:1024]
        l = F.mse_loss(probe(Z[idx][:, :-1]), Y[idx][:, :-1])
        optp.zero_grad()
        l.backward()
        optp.step()
    with torch.no_grad():
        pm = float(F.mse_loss(probe(Z[:, :-1]), Y[:, :-1]))
    return {"seed": seed, "d": d, "params": params, "rel_gain": (base - pm) / base,
            "latent_std": latent_std(Z.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring", choices=("spring", "orbit"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--widths", default="64,128,256")
    ap.add_argument("--sigreg-weight", type=float, default=0.01,
                    help="0.01 有 seed 离群（10-04 权重扫）；M8 重试建议 0.001")
    args = ap.parse_args()
    widths = [int(w) for w in args.widths.split(",")]

    print(f"=== World M8: parameter-efficiency curve ({args.kind}) widths "
          f"{widths} ===")
    rows = []
    for d in widths:
        gains = []
        for seed in range(args.seeds):
            r = train_one(args.kind, seed, args.steps, d,
                          sigreg_weight=args.sigreg_weight)
            rows.append(r)
            gains.append(r["rel_gain"])
            print(f"  d={d} seed {seed}: params {r['params']:,}  "
                  f"gain {r['rel_gain']*100:.1f}%  std {r['latent_std']:.3f}")
        mean = sum(gains) / len(gains)
        print(f"  d={d}: mean gain {mean*100:.1f}%  "
              f"(params {rows[-1]['params']:,})")

    means = []
    for d in widths:
        g = [r["rel_gain"] for r in rows if r["d"] == d]
        means.append(sum(g) / len(g))
    ok = all(m > 0.70 for m in means)
    print(f"\nmean gains: {[f'{m*100:.1f}%' for m in means]}")
    print("VERDICT:", "PASS" if ok else "FAIL", "(all widths > +70%)")


if __name__ == "__main__":
    main()
