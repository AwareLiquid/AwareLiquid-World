"""Experiment 3 — irregular sampling robustness (liquid core's home turf).

Train on uniform-spaced planted-dynamics series (same as experiment 1),
then at test time drop a fraction of steps — the recurrent models must
carry their state across gaps of varying length, the linear model just
misses steps. Prior repo evidence (NASA battery): liquid degrades +7.7%
vs LSTM/GRU +31% at 80% sample loss. This checks the same property on
the synthetic financial stream.

Run:
  python experiments/finance_fitting/irregular_test.py
"""
from __future__ import annotations

import torch

from finance_fitting import (GruTrader, LinearTrader, LiquidTrader,
                             gen_series, make_features, train)

torch.manual_seed(0)


def drop_steps(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Remove a random frac of steps, keeping the rest in order."""
    if frac <= 0:
        return x
    g = torch.Generator().manual_seed(int(frac * 1000))
    keep = torch.rand(x.shape[0], generator=g) > frac
    return x[keep]


def backtest_full(model, series, drop: float, cost: float = 5e-4) -> float:
    """Direction accuracy over the NON-dropped prediction steps."""
    model.eval()
    x, _, nxt = make_features(series)
    xd = drop_steps(x, drop)
    with torch.no_grad():
        p = model(xd.unsqueeze(1)).squeeze(-1)
    # labels on the kept steps only (re-derive via the kept indices)
    g = torch.Generator().manual_seed(int(drop * 1000))
    keep = torch.rand(x.shape[0], generator=g) > drop
    nxtk = nxt[keep]
    acc = ((p > 0.5).float() == ((nxtk > 0).float())).float().mean().item()
    return acc


def main() -> int:
    print("training on uniform-spaced planted series (as experiment 1)", flush=True)
    train_series = [gen_series(1000, s) for s in range(15)]
    held_series = [gen_series(1000, 100 + s) for s in range(10)]
    xs, ys = [], []
    for s in train_series:
        x, y, _ = make_features(s)
        xs.append(x)
        ys.append(y)
    F_DIM = xs[0].shape[-1]

    models = {
        "liquid": LiquidTrader(F_DIM, 32),
        "gru": GruTrader(F_DIM, 32),
        "linear": LinearTrader(F_DIM),
    }
    for name, m in models.items():
        train(m, xs, ys)
        print(f"[{name}] trained", flush=True)

    print("\ndirection accuracy vs dropped fraction (held-out):", flush=True)
    print(f"{'drop':>6} " + " ".join(f"{name:>9}" for name in models), flush=True)
    for drop in (0.0, 0.3, 0.6, 0.8):
        row = {}
        for name, m in models.items():
            accs = [backtest_full(m, s, drop) for s in held_series]
            row[name] = sum(accs) / len(accs)
        print(f"{drop:>5.0%} " + " ".join(f"{row[name]:>8.1%}" for name in models),
              flush=True)
    print("\nHONEST NOTE: models trained on uniform spacing, stressed with gaps.")
    print("The liquid core's per-channel tau should carry state across gaps;")
    print("GRU is also recurrent — the question is how much each degrades.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
