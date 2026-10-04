"""World M9 milestone: long-horizon system evolution (ThinkJEPA analog).

Plan (PLAN.md M9): with explicit latent variables (M3), the model evolves
system state over months-years horizons for capacity/maintenance planning.
Metric: calibrated horizon vs degradation vs a naive baseline. Success:
honest multi-path rollouts with stated uncertainty.

Data (synthetic degradation, the battery-capacity analog):
    C(t) = 1 - a * (1 - exp(-k * t / T)),   t in [0, T]
per unit (a, k) ~ U(0.05,0.5) x U(0.5,4) — the latent factors a unit's
history must identify. Horizon = 900 steps after a 100-step observation
window (the months-years analog at unit scale).

Protocol:
  - encode the first 100 steps -> (shared, z); decode the next 900 (MSE).
  - eval (held-out): sample K=20 prior paths, coverage = fraction of the
    true future inside the path envelope (10-90 pct), measured AT HORIZON
    BANDS (h=150/300/600/900). Calibrated horizon = the largest band with
    coverage >= 0.8.
  - baselines: persistence (last value) + final-slope linear extrapolation.

Run:  python -m experiments.world_m1.world_m9
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LiquidEncoder


def make_fade(batch: int, T: int = 1000, obs: int = 100,
              seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(observed, future): capacity curves; latent (a,k) per unit."""
    g = torch.Generator().manual_seed(seed)
    a = 0.05 + torch.rand(batch, generator=g) * 0.45
    k = 0.5 + torch.rand(batch, generator=g) * 3.5
    t = torch.arange(T, dtype=torch.float32)
    C = 1.0 - a.unsqueeze(1) * (1.0 - torch.exp(-k.unsqueeze(1) * t / T))
    C = C + torch.randn(batch, T, generator=g) * 0.005
    return C[:, :obs].unsqueeze(-1), C[:, obs:].unsqueeze(-1)


class FadeWorld(nn.Module):
    def __init__(self, d: int = 64, latent: int = 8, future: int = 900):
        super().__init__()
        self.encoder = LiquidEncoder(1, d=d, n_layers=2)
        self.shared = nn.Linear(d, d - latent)
        self.to_z = nn.Linear(d, latent)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, future))

    def encode(self, x):
        seq = self.encoder(x)
        return self.shared(seq[:, -1]), self.to_z(seq[:, -1])

    def decode(self, shared, z):
        b, n_paths, _ = z.shape
        sh = shared.unsqueeze(1).expand(b, n_paths, -1)
        return self.decoder(torch.cat([sh, z], dim=-1))


def train(model, steps: int = 6000, batch: int = 64, beta: float = 0.1,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for step in range(steps):
        obs, fut = make_fade(batch, seed=seed * 10000 + step)
        shared, z = model.encode(obs)
        pred = model.decode(shared, z.unsqueeze(1))[:, 0]
        mse = F.mse_loss(pred, fut.squeeze(-1))
        kl = 0.5 * (z.square()).sum(-1).mean()
        loss = mse + beta * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1} mse {mse.item():.5f} kl {kl.item():.3f}",
                  flush=True)


@torch.no_grad()
def evaluate(model, b0: int = 15000000, k_paths: int = 20) -> dict:
    model.eval()
    obs, fut = make_fade(256, seed=b0)
    fut = fut.squeeze(-1)                     # (B, 900)
    shared, _ = model.encode(obs)
    z = torch.randn(shared.shape[0], k_paths, 8)
    paths = model.decode(shared, z)           # (B, K, 900)
    lo = paths.quantile(0.1, dim=1)
    hi = paths.quantile(0.9, dim=1)
    out = {}
    for h in (150, 300, 600, 900):
        inside = ((fut[:, :h] >= lo[:, :h]) & (fut[:, :h] <= hi[:, :h]))
        out[f"coverage_h{h}"] = float(inside.float().mean())
    # naive baselines: persistence / slope extrapolation from the obs window
    last = obs[:, -1, 0:1].expand(-1, 900)
    slope = (obs[:, -1] - obs[:, 0]) / (obs.shape[1] - 1)
    lin = obs[:, -1, 0:1] + slope * torch.arange(1, 901).unsqueeze(0)
    out["persistence_mse"] = float(((last - fut) ** 2).mean())
    out["linear_mse"] = float(((lin - fut) ** 2).mean())
    out["model_mse"] = float(((paths.mean(1) - fut) ** 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("=== World M9: long-horizon evolution (degradation, 100->900) ===")
    rows = []
    for seed in range(args.seeds):
        model = FadeWorld()
        train(model, steps=args.steps, seed=seed)
        r = evaluate(model, b0=15000000 + seed)
        rows.append(r)
        print(f"  seed {seed}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in r.items()))

    ok_cov = all(r["coverage_h900"] >= 0.8 for r in rows)
    ok_mse = all(r["model_mse"] <= min(r["persistence_mse"], r["linear_mse"])
                 for r in rows)
    print(f"\ncoverage@900 >= 0.8: {ok_cov}; "
          f"model_mse <= best naive: {ok_mse}")
    print("VERDICT:", "PASS" if ok_cov and ok_mse else "FAIL")


if __name__ == "__main__":
    main()
