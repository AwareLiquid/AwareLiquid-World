"""M9c: learning prediction bands directly (quantile regression).

M9/M9b showed the bottleneck is not the sampling prior: an isotropic
Gaussian posterior + deterministic decoder cannot produce correctly
shaped bands (posterior sampling + sigma calibration still under-covered
at long horizons). M9c removes the Gaussian-band assumption entirely:

  decoder -> (q10, q50, q90) per horizon step
  loss    = pinball(q10, 0.1) + pinball(q50, 0.5) + pinball(q90, 0.9) + beta*KL
  coverage = frac(true in [q10, q90])   -- directly targets 0.8

Judge: all 3 seeds h900 coverage in [0.75, 0.85] (a calibrated band, not
just "wide enough"): widening alone fails h150 lower bound; the pinball
loss must find the right shape.

Run:  python -m experiments.world_m1.world_m9c
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .encoder import LiquidEncoder
from .world_m9 import make_fade


class FadeWorldQ(nn.Module):
    def __init__(self, d: int = 64, latent: int = 8, future: int = 900):
        super().__init__()
        self.encoder = LiquidEncoder(1, d=d, n_layers=2)
        self.shared = nn.Linear(d, d - latent)
        self.to_z = nn.Linear(d, latent)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 3 * future))
        self.future = future

    def encode(self, x):
        seq = self.encoder(x)
        return self.shared(seq[:, -1]), self.to_z(seq[:, -1])

    def decode(self, shared, z):
        fused = torch.cat([shared, z], dim=-1)
        out = self.decoder(fused).view(-1, 3, self.future)
        return out  # (b, 3, future) = q10, q50, q90


def pinball(pred, target, tau):
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1) * diff)


def train(model, steps: int = 6000, batch: int = 64, beta: float = 0.1,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for step in range(steps):
        obs, fut = make_fade(batch, seed=seed * 10000 + step)
        y = fut.squeeze(-1)                      # (b, future)
        shared, z = model.encode(obs)
        q = model.decode(shared, z)              # (b, 3, future)
        loss = (pinball(q[:, 0], y, 0.1).mean()
                + pinball(q[:, 1], y, 0.5).mean()
                + pinball(q[:, 2], y, 0.9).mean())
        kl = 0.5 * (z.square()).sum(-1).mean()
        loss = loss + beta * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step + 1} pinball {loss.item():.5f}")


@torch.no_grad()
def evaluate(model, batch: int = 256, seed: int = 15000000):
    model.eval()
    obs, fut = make_fade(batch, seed=seed)
    y = fut.squeeze(-1)
    shared, z = model.encode(obs)
    q = model.decode(shared, z)
    lo, mid, hi = q[:, 0], q[:, 1], q[:, 2]
    covs = {}
    for h in (150, 300, 600, 900):
        inside = (y[:, :h] >= lo[:, :h]) & (y[:, :h] <= hi[:, :h])
        covs[f"h{h}"] = float(inside.float().mean())
    mse = float(((mid - y) ** 2).mean())
    width = float((hi - lo).mean())
    last = obs[:, -1, 0:1].expand(-1, 900)
    per = float(((last - y) ** 2).mean())
    return {**covs, "mse": mse, "band_width": width, "persistence": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("=== M9c: quantile bands (learned directly) ===")
    rows = []
    for seed in range(args.seeds):
        model = FadeWorldQ()
        train(model, steps=args.steps, seed=seed)
        r = evaluate(model)
        rows.append(r)
        print(f"  seed {seed}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()),
              flush=True)

    ok = all(0.75 <= r["h900"] <= 0.85 for r in rows)
    print(f"\nall seeds h900 in [0.75, 0.85]: {ok}")
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
