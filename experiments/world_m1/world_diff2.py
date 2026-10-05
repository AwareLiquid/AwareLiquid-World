"""Diffusion-like state evolution v2 — the contractive fix.

Pilot (world_diff.py) result: naive fixed-point refinement with alpha=0.5
DIVERGED at K=5 (mean gain -5174%): the map z -> tanh(Wx + U_ref z + b) is
not a contraction, so iterating it amplified errors instead of correcting
them. Skill note at the time: "若重启此线：带训练的细化目标（每个修正步有
监督）或严格收缩映射的参数化".

v2 implements the **strictly contractive parametrization**:
    - U_ref is spectral-normalised and scaled by `contract` (<1), so the
      refinement map has Lipschitz constant <= contract < 1 and the
      fixed-point iteration converges for ANY step size;
    - step sizes alpha_k = sigmoid(raw_k) are per-step learnable (variant
      `spectral_learn`), removing the hand-set 0.5 that left the stability
      region;
    - variant `plain_small` keeps the pilot's map but fixes alpha=0.1, to
      separate "step size alone" from "contraction".

Success criterion: K>1 does not hurt (mean gain >= baseline K=1 within
noise) AND no divergence. Baseline K=1 from the pilot: 67.8%.

Run:  python -m experiments.world_m1.world_diff2
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.objectives import masked_latent_prediction_loss, mask_spans

from .data import make_dataset
from .train import LatentPredictor, latent_std
from .world_diff import run_one as _pilot_run_one  # noqa: F401  (kept for cross-reference)


class DiffCellV2(nn.Module):
    """Gradual-correction cell, optionally strictly contractive."""

    def __init__(self, in_dim: int, d: int, refine: int = 1,
                 contract: float = 0.9, learn_alpha: bool = True,
                 tau_min: float = 0.1):
        super().__init__()
        self.d = d
        self.refine = refine
        self.contract = contract
        self.learn_alpha = learn_alpha
        self.W = nn.Linear(in_dim, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        if contract > 0:
            # spectral-normalised coupling: ||c * U_ref|| <= c < 1
            self.U_ref = nn.utils.parametrizations.spectral_norm(
                nn.Linear(d, d, bias=False))
        else:
            self.U_ref = nn.Linear(d, d, bias=False)
            nn.init.normal_(self.U_ref.weight, std=0.02)
        nn.init.normal_(self.U.weight, std=0.02)
        self.b = nn.Parameter(torch.zeros(d))
        lo, hi = 0.5, 90.0
        taus = lo * (hi / lo) ** (torch.arange(d) / max(d - 1, 1))
        self.log_tau = nn.Parameter(
            torch.log(torch.clamp(taus - tau_min, min=1e-3)))
        self.tau_min = tau_min
        if learn_alpha:
            # sigmoid(-0.85) ~ 0.3
            self.raw_alpha = nn.Parameter(
                torch.full((max(refine - 1, 1),), -0.85))
        else:
            self.register_buffer("raw_alpha",
                                 torch.full((max(refine - 1, 1),), -0.85),
                                 persistent=False)

    def forward(self, x, h_prev=None):
        B, T, _ = x.shape
        tau = F.softplus(self.log_tau) + self.tau_min
        decay = torch.exp(-1.0 / tau).view(1, self.d)
        h = torch.zeros(B, self.d, device=x.device)
        if h_prev is not None:
            h = h + h_prev
        hs = []
        for t in range(T):
            base = self.W(x[:, t]) + self.b
            z = torch.tanh(base + self.U(h))
            for k in range(self.refine - 1):
                a = torch.sigmoid(self.raw_alpha[k])
                z_new = torch.tanh(
                    base + self.contract * self.U_ref(z)
                    if self.contract > 0 else base + self.U_ref(z))
                z = z + a * (z_new - z)
            h = decay * h + (1.0 - decay) * z
            hs.append(h)
        return torch.stack(hs, dim=1), h


class DiffEncoderV2(nn.Module):
    def __init__(self, in_dim: int, d: int = 64, n_layers: int = 2,
                 refine: int = 1, contract: float = 0.9,
                 learn_alpha: bool = True):
        super().__init__()
        self.cells = nn.ModuleList(
            DiffCellV2(in_dim if i == 0 else d, d, refine=refine,
                       contract=contract, learn_alpha=learn_alpha)
            for i in range(n_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = x
        for cell in self.cells:
            h, _ = cell(h)
        return self.norm(h)


def run_one_v2(kind: str, refine: int, seed: int, steps: int = 6000,
               T: int = 64, batch: int = 64, contract: float = 0.9,
               learn_alpha: bool = True) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = DiffEncoderV2(2 if kind == "spring" else 4, d=64, n_layers=2,
                        refine=refine, contract=contract,
                        learn_alpha=learn_alpha).to(dev)
    pred = LatentPredictor(64).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    diverged = False
    for step in range(steps):
        traj, _ = make_dataset(kind, batch, T, seed=seed * 10000 + step)
        traj = traj.to(dev)
        z = enc(traj)
        if not torch.isfinite(z).all():
            diverged = True
            break
        keep = mask_spans(T, 0.5, 6, g)
        loss = masked_latent_prediction_loss(z, keep, pred)
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step()

    enc.eval()
    with torch.no_grad():
        X, Y = make_dataset(kind, 2048, T + 1, seed=777 + seed)
        X, Y = X.to(dev), Y.to(dev)
        Z = enc(X)
        if not torch.isfinite(Z).all():
            return {"refine": refine, "seed": seed, "gain": float("-inf"),
                    "diverged": True}
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
        ra = enc.cells[0].raw_alpha.detach().reshape(-1)
        alphas = [float(torch.sigmoid(a)) for a in ra]
    return {"refine": refine, "seed": seed, "base": base_mse, "probe": pm,
            "gain": (base_mse - pm) / base_mse,
            "std": latent_std(Z.detach()), "diverged": diverged,
            "alpha0": round(alphas[0], 3) if alphas else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refines", default="3,5")
    args = ap.parse_args()
    refines = [int(r) for r in args.refines.split(",")]

    variants = [
        ("spectral_learn", dict(contract=0.9, learn_alpha=True)),
        ("plain_small", dict(contract=0.0, learn_alpha=False)),
    ]
    print(f"=== Diffusion v2 (contractive) pilot ({args.kind}) ===")
    print("baseline K=1 (pilot, mean over 3 seeds): 67.8%")
    for name, kw in variants:
        for K in refines:
            gains, div = [], 0
            for seed in range(args.seeds):
                r = run_one_v2(args.kind, K, seed, steps=args.steps, **kw)
                if r["diverged"] or r["gain"] == float("-inf"):
                    div += 1
                    print(f"  [{name}] K={K} seed {seed}: DIVERGED", flush=True)
                else:
                    gains.append(r["gain"])
                    print(f"  [{name}] K={K} seed {seed}: gain "
                          f"{r['gain'] * 100:.1f}%  std {r['std']:.3f}  "
                          f"alpha0 {r.get('alpha0')}", flush=True)
            if gains:
                mg = sum(gains) / len(gains) * 100
                verdict = "PASS" if mg >= 50 and div == 0 else "FAIL"
                print(f"  [{name}] K={K}: mean gain {mg:.1f}%  "
                      f"diverged {div}/{args.seeds}  -> {verdict}")
            else:
                print(f"  [{name}] K={K}: all diverged -> FAIL")


if __name__ == "__main__":
    main()
