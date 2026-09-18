"""Experiment 2 — GARCH no-signal discipline test.

GARCH(1,1)-style returns: zero mean, volatility clustering, NO planted
directional structure. Direction is a fair coin by construction.

Question: do the models FALSELY find signal on no-signal data?
Expected honest answer: direction accuracy ~50% for every model, and
strategy returns NEGATIVE after costs (the models cannot invent alpha).

This is the anti-overfitting gate: if any model shows >55% on this data,
the training/backtest setup itself is leaking signal (e.g. lookahead),
and experiment 1's numbers would be untrustworthy.

Run:
  python experiments/finance_fitting/garch_test.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from finance_fitting import (GruTrader, LinearTrader, LiquidTrader,
                             MlpTrader, backtest, make_features, train)

torch.manual_seed(0)


def gen_garch(n: int, seed: int) -> torch.Tensor:
    """GARCH(1,1): r_t = sigma_t * eps_t, sigma^2 = w + a*r^2 + b*sigma^2."""
    g = torch.Generator().manual_seed(seed)
    w, a, b = 1e-5, 0.10, 0.85
    sigma2 = 1e-4
    prices = torch.ones(n)
    for t in range(1, n):
        r = math.sqrt(sigma2) * torch.randn(1, generator=g).item()
        prices[t] = prices[t - 1] * (1.0 + r)
        sigma2 = w + a * r * r + b * sigma2
    return prices


def main() -> int:
    print("generating GARCH series (zero mean, no directional signal)", flush=True)
    train_series = [gen_garch(1000, s) for s in range(15)]
    held_series = [gen_garch(1000, 100 + s) for s in range(10)]

    xs, ys = [], []
    for s in train_series:
        x, y, _ = make_features(s)
        xs.append(x)
        ys.append(y)
    F_DIM = xs[0].shape[-1]

    models = {
        "liquid": LiquidTrader(F_DIM, 32),
        "gru": GruTrader(F_DIM, 32),
        "mlp": MlpTrader(F_DIM, 32),
        "linear": LinearTrader(F_DIM),
    }
    for name, m in models.items():
        loss = train(m, xs, ys)
        print(f"[{name}] train loss {loss:.4f}", flush=True)

    print("\nheld-out backtest on NO-SIGNAL data:", flush=True)
    agg = {name: {"acc": 0.0, "ret": 0.0} for name in models}
    for i, s in enumerate(held_series):
        row = []
        for name, m in models.items():
            r = backtest(m, s)
            agg[name]["acc"] += r["acc"]
            agg[name]["ret"] += r["strat"]
            row.append(f"{name}={r['acc']:.1%}/{r['strat']:+.1%}")
        print(f"  series {i}: " + "  ".join(row), flush=True)

    n = len(held_series)
    print("\nmean over held-out no-signal series:", flush=True)
    for name in models:
        print(f"  {name:>7}: acc {agg[name]['acc'] / n:.1%}  "
              f"log-ret {agg[name]['ret'] / n:+.1%}", flush=True)
    print("\nVERDICT: acc ~50% and negative returns = no false signal found.", flush=True)
    print("If any model shows acc > 55%, the pipeline leaks — investigate.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
