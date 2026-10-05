"""M9d: conformalized quantile regression (CQR) — the statistically valid fix.

M9c (quantile bands) showed the right mechanism but per-seed coverage
varies: 2/5 seeds in [0.75,0.85] for both the relu-12k and latent-16 configs
(one seed 0.627, one 0.855), and d128 is systematically narrow (median
0.745). The seed variance is not a training-length or capacity problem —
it is a calibration problem, and split conformal prediction (Romano et al.
2019, CQR) is the distribution-free fix:

    E_t = max(q10_t - y_t, y_t - q90_t)        signed miss distance
    Q_t = Quantile_{(1-a)(1+1/n)}(E_t)         on held-out calibration traj
    band_t = [q10_t - Q_t, q90_t + Q_t]

Under exchangeability this yields marginal coverage >= 1-a at every horizon
PER SEED, without touching the model. Raw (uncalibrated) coverage and the
average widening |Q| are reported alongside, so the price is visible.

Judge: >= 4/5 seeds h900 in [0.75, 0.85] AND median in [0.78, 0.82].

Run:  python -m experiments.world_m1.world_m9d
"""

from __future__ import annotations

import argparse

import torch

from .world_m9 import make_fade
from .world_m9c import FadeWorldQ, train

TARGET = 0.8
CALIB_BASE = 21000000      # disjoint from training (seed*10000+step) and eval
EVAL_BASE = 15000000


@torch.no_grad()
def bands_and_Q(model, batch: int = 512, calib_seed: int = CALIB_BASE,
                n_calib: int = 512) -> torch.Tensor:
    """Per-timestep conformal correction Q_t from a held-out calibration set."""
    model.eval()
    obs, fut = make_fade(n_calib, seed=calib_seed)
    y = fut.squeeze(-1)
    q = model.decode(*model.encode(obs))
    lo, hi = q[:, 0], q[:, 2]
    E = torch.maximum(lo - y, y - hi)              # (n, 900)
    level = min(1.0, TARGET * (1.0 + 1.0 / n_calib))
    return torch.quantile(E, level, dim=0)         # (900,)


@torch.no_grad()
def evaluate_cqr(model, Q: torch.Tensor, batch: int = 256,
                 seed: int = EVAL_BASE) -> dict:
    model.eval()
    obs, fut = make_fade(batch, seed=seed)
    y = fut.squeeze(-1)
    q = model.decode(*model.encode(obs))
    lo, mid, hi = q[:, 0], q[:, 1], q[:, 2]
    lo_c, hi_c = lo - Q, hi + Q
    out = {}
    for h in (150, 300, 600, 900):
        raw = ((y[:, :h] >= lo[:, :h]) & (y[:, :h] <= hi[:, :h])).float().mean()
        cal = ((y[:, :h] >= lo_c[:, :h]) & (y[:, :h] <= hi_c[:, :h])).float().mean()
        out[f"raw_h{h}"] = float(raw)
        out[f"cqr_h{h}"] = float(cal)
    out["mse"] = float(((mid - y) ** 2).mean())
    out["band_width_raw"] = float((hi - lo).mean())
    out["band_width_cqr"] = float((hi_c - lo_c).mean())
    out["Q_mean"] = float(Q.mean())
    last = obs[:, -1, 0:1].expand(-1, 900)
    out["persistence"] = float(((last - y) ** 2).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-calib", type=int, default=512,
                    help="calibration set size; 512 gives ~1.8%% level SE, "
                         "1024 ~1.2%%")
    args = ap.parse_args()

    print(f"=== M9d: CQR calibration of quantile bands "
          f"(steps={args.steps} seeds={args.seeds} n_calib={args.n_calib}) ===")
    rows = []
    for seed in range(args.seeds):
        model = FadeWorldQ()
        train(model, steps=args.steps, seed=seed)
        Q = bands_and_Q(model, calib_seed=CALIB_BASE + seed,
                        n_calib=args.n_calib)
        r = evaluate_cqr(model, Q, seed=EVAL_BASE + seed)
        rows.append(r)
        print(f"  seed {seed}: cqr h900 {r['cqr_h900']:.3f} "
              f"(raw {r['raw_h900']:.3f})  width {r['band_width_raw']:.3f}"
              f"->{r['band_width_cqr']:.3f}  mse {r['mse']:.4f}", flush=True)

    n_ok = sum(1 for r in rows if 0.75 <= r["cqr_h900"] <= 0.85)
    h = sorted(r["cqr_h900"] for r in rows)
    med = h[len(h) // 2]
    print(f"\nin-range CQR h900: {n_ok}/{len(rows)}  median {med:.3f}")
    ok = n_ok >= 4 and 0.78 <= med <= 0.82
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
