"""Diffusion-like state evolution pilot (liquid-research-skill 明示选项).

Skill: "Pilot diffusion in state evolution — prototype a gradual-correction,
diffusion-like mechanism in the liquid state-evolution step (small-scale
validation only)."

Design (small-scale pilot, honest scope):
  current:  h_t = decay*h_{t-1} + (1-decay)*(tanh(Wx) + tanh(U h_{t-1}))
  pilot:    z^(0) = tanh(Wx + U h_{t-1} + b)
            z^(k) = z^(k-1) + alpha * (tanh(Wx + U_ref z^(k-1) + b) - z^(k-1))
            h_t   = decay*h_{t-1} + (1-decay)*z^(K-1)
  NOTE: the pilot cell itself is a variant (sum-inside-tanh); K=1 is the
  variant's single-step limit, NOT bit-identical to the original cell. The
  comparison K=1 vs K=3/5 therefore isolates the REFINEMENT effect on a
  common base cell — that is the cleanest available pilot, and no
  zero-regression claim is made here (the pilot is off the default path).

Validation (small-scale): the M1 masked-latent task (spring) — probe gain at
K=1 (baseline) vs K=3 and K=5. Success: K>1 does not hurt (within noise) and
any improvement is reported honestly; the mechanism's cost (K x recurrent
work) is logged.

Run:  python -m experiments.world_m1.world_diff
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.objectives import masked_latent_prediction_loss, mask_spans

from .data import make_dataset
from .train import LatentPredictor, latent_std


class DiffCell(nn.Module):
    """Liquid cell with K-step gradual correction of the state proposal."""

    def __init__(self, in_dim: int, d: int, refine: int = 1,
                 alpha: float = 0.5, tau_min: float = 0.1):
        super().__init__()
        self.d = d
        self.refine = refine
        self.alpha = alpha
        self.W = nn.Linear(in_dim, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        self.U_ref = nn.Linear(d, d, bias=False)   # refinement coupling
        nn.init.normal_(self.U.weight, std=0.02)
        nn.init.normal_(self.U_ref.weight, std=0.02)
        self.b = nn.Parameter(torch.zeros(d))
        lo, hi = 0.5, 90.0
        taus = lo * (hi / lo) ** (torch.arange(d) / max(d - 1, 1))
        self.log_tau = nn.Parameter(torch.log(torch.clamp(taus - tau_min, min=1e-3)))
        self.tau_min = tau_min

    def forward(self, x, h_prev=None):
        B, T, _ = x.shape
        tau = F.softplus(self.log_tau) + self.tau_min
        decay = torch.exp(-1.0 / tau).view(1, self.d)
        h = torch.zeros(B, self.d, device=x.device)
        if h_prev is not None:
            h = h + h_prev
        hs = []
        for t in range(T):
            z = torch.tanh(self.W(x[:, t]) + self.U(h) + self.b)
            for _ in range(self.refine - 1):
                z = z + self.alpha * (torch.tanh(
                    self.W(x[:, t]) + self.U_ref(z) + self.b) - z)
            h = decay * h + (1.0 - decay) * z
            hs.append(h)
        return torch.stack(hs, dim=1), h


class DiffEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 64, n_layers: int = 2,
                 refine: int = 1):
        super().__init__()
        self.cells = nn.ModuleList(
            DiffCell(in_dim if i == 0 else d, d, refine=refine)
            for i in range(n_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = x
        for cell in self.cells:
            h, _ = cell(h)
        return self.norm(h)


def run_one(kind: str, refine: int, seed: int, steps: int = 6000,
            T: int = 64, batch: int = 64) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DiffEncoder(2 if kind == "spring" else 4, d=64, n_layers=2,
                      refine=refine).to(dev)
    pred = LatentPredictor(64).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        traj, _ = make_dataset(kind, batch, T, seed=seed * 10000 + step)
        traj = traj.to(dev)
        z = enc(traj)
        keep = mask_spans(T, 0.5, 6, g)
        loss = masked_latent_prediction_loss(z, keep, pred)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step()

    # probe eval (same protocol as train.py)
    enc.eval()
    X, Y = make_dataset(kind, 2048, T + 1, seed=777 + seed)
    X, Y = X.to(dev), Y.to(dev)
    with torch.no_grad():
        Z = enc(X)
    base_mse = float(((Y[:, :-1] - X[:, :-1]) ** 2).mean())
    probe = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2)).to(dev)
    optp = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(1500):
        idx = torch.randperm(Z.shape[0])[:1024]
        loss = nn.functional.mse_loss(probe(Z[idx][:, :-1]), Y[idx][:, :-1])
        optp.zero_grad()
        loss.backward()
        optp.step()
    with torch.no_grad():
        pm = float(nn.functional.mse_loss(probe(Z[:, :-1]), Y[:, :-1]))
    return {"refine": refine, "seed": seed, "base": base_mse, "probe": pm,
            "gain": (base_mse - pm) / base_mse, "std": latent_std(Z.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refines", default="1,3,5")
    args = ap.parse_args()
    refines = [int(r) for r in args.refines.split(",")]

    print(f"=== Diffusion-like state evolution pilot ({args.kind}) ===")
    for K in refines:
        gains = []
        for seed in range(args.seeds):
            r = run_one(args.kind, K, seed, steps=args.steps)
            gains.append(r["gain"])
            print(f"  K={K} seed {seed}: gain {r['gain']*100:.1f}%  "
                  f"std {r['std']:.3f}", flush=True)
        print(f"  K={K}: mean gain {sum(gains)/len(gains)*100:.1f}%  "
              f"(x{K} recurrent work)")


if __name__ == "__main__":
    main()
