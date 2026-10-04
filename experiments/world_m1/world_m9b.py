"""M9 fix: posterior-informed multi-path sampling + post-hoc sigma calibration.

The M9 pilot failed coverage at long horizons because paths were sampled
from the PRIOR z~N(0,1) — misaligned with each unit's posterior. Fix:

  1. sample paths around the POSTERIOR mean: z = z_post + sigma * eps
  2. fit sigma on a CALIBRATION set (the coverage->0.8 inverse problem,
     scanned over a grid) — the regression analogue of temperature scaling
  3. report coverage on a held-out set with the calibrated sigma

Same FadeWorld architecture and training as world_m9.py; this file only
replaces the evaluation protocol. Honest reporting: if calibrated coverage
still misses at h900, the mechanism (not just the sampling) is the limit.

Run:  python -m experiments.world_m1.world_m9b
"""

from __future__ import annotations

import argparse

import torch

from .world_m9 import FadeWorld, make_fade, train


@torch.no_grad()
def paths_from(model, obs, sigma: float, k: int = 20, seed: int = 0):
    shared, z_post = model.encode(obs)
    g = torch.Generator().manual_seed(seed)
    eps = torch.randn(z_post.shape[0], k, z_post.shape[1],
                      generator=g, device=z_post.device)
    z = z_post.unsqueeze(1) + sigma * eps
    return model.decode(shared, z)


def coverage_at(paths: torch.Tensor, fut: torch.Tensor,
                h: int) -> float:
    lo = paths.quantile(0.1, dim=1)
    hi = paths.quantile(0.9, dim=1)
    inside = (fut[:, :h] >= lo[:, :h]) & (fut[:, :h] <= hi[:, :h])
    return float(inside.float().mean())


def run_seed(seed: int, steps: int, target: float = 0.8) -> dict:
    model = FadeWorld()
    train(model, steps=steps, seed=seed)
    model.eval()

    # calibration set (different seed stream from train and eval)
    obs_c, fut_c = make_fade(256, seed=21000000 + seed)
    fut_c = fut_c.squeeze(-1)
    grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_sigma, best_gap = 0.0, 1e9
    for s in grid:
        cov = coverage_at(paths_from(model, obs_c, s, seed=1), fut_c, 900)
        gap = abs(cov - target)
        if gap < best_gap:
            best_gap, best_sigma = gap, s

    # held-out eval with the calibrated sigma
    obs, fut = make_fade(256, seed=15000000 + seed)
    fut = fut.squeeze(-1)
    paths = paths_from(model, obs, best_sigma, seed=2)
    covs = {f"h{h}": coverage_at(paths, fut, h) for h in (150, 300, 600, 900)}
    mse = float(((paths.mean(1) - fut) ** 2).mean())
    last = obs[:, -1, 0:1].expand(-1, 900)
    per = float(((last - fut) ** 2).mean())
    return {"seed": seed, "sigma": best_sigma, **covs,
            "h900_ok": covs["h900"] >= 0.8, "model_mse": mse,
            "persistence_mse": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("=== M9b: posterior sampling + sigma calibration ===")
    rows = []
    for seed in range(args.seeds):
        r = run_seed(seed, args.steps)
        rows.append(r)
        print(f"  seed {seed}: sigma {r['sigma']:.2f}  "
              + "  ".join(f"{k}={v:.3f}" for k, v in r.items()
                          if k.startswith("h"))
              + f"  mse {r['model_mse']:.4f} vs per {r['persistence_mse']:.4f}",
              flush=True)

    ok = all(r["h900_ok"] for r in rows)
    print(f"\nall seeds h900 coverage >= 0.8: {ok}")
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
