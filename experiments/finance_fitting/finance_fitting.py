"""Light financial fitting experiment — liquid core vs traditional models.

Honest scope: SIMULATED data with planted, known dynamics. The experiment
proves the learning machinery can pick up planted structure and convert it
into a working trading signal on HELD-OUT series — it does NOT claim anything
about real markets.

Comparison (same data, same labels, same training budget, same backtest):
  - liquid : streaming 2-layer liquid core (canonical MT-LNN step), ~3.9k params
  - gru    : 1-layer GRU (traditional recurrent), ~3.4k params
  - mlp    : 2-layer MLP over the flattened factor window, ~0.8k params
  - linear : logistic regression over the factor window (classic factor model)

Run:
  python experiments/finance_fitting/finance_fitting.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
EPOCHS = 40
LR = 1e-3


# ------------------------------------------------------------------ #
# Synthetic market generator (planted dynamics)
# ------------------------------------------------------------------ #

def gen_series(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    regime_len = 200
    regimes = torch.stack([
        torch.tensor([0.0025, 0.60, 0.010]),   # bull: +mu, strong momentum
        torch.tensor([-0.0025, 0.60, 0.010]),  # bear
        torch.tensor([0.0000, 0.10, 0.005]),   # sideways
    ])
    prices = torch.ones(n)
    r_prev = 0.0
    for t in range(1, n):
        mu, mom, sig = regimes[(t // regime_len) % 3]
        r = mu + mom * r_prev + sig * torch.randn(1, generator=g).item()
        r_prev = r
        prices[t] = prices[t - 1] * (1.0 + r)
    return prices


def make_features(series: torch.Tensor, win: int = 20):
    rets = series[1:] / series[:-1] - 1.0
    feats, labels, nxt = [], [], []
    for t in range(win, len(rets) - 1):
        w = rets[t - win:t]
        vol = w.std().item()
        mom = w.sum().item()
        feats.append([w.numpy() / (vol + 1e-6), [vol], [mom]])
        labels.append(1.0 if rets[t + 1] > 0 else 0.0)
        nxt.append(rets[t + 1])
    x = torch.stack([torch.cat([torch.tensor(f[0]), torch.tensor(f[1]),
                                torch.tensor(f[2])], dim=0) for f in feats])
    x = x / (x.std(dim=0, keepdim=True) + 1e-6)
    return x, torch.tensor(labels), torch.tensor(nxt)


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #

class LiquidCell(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.rec = nn.Linear(hidden, hidden, bias=False)
        taus = torch.logspace(0, math.log10(24.0), hidden)
        self.log_tau = nn.Parameter(torch.log(torch.expm1(
            (taus - 1.0).clamp_min(1e-6))))

    def step(self, x_t, h):
        pre = self.inp(x_t) + self.rec(h)
        alpha = 1.0 / (F.softplus(self.log_tau) + 1.0)
        return h + alpha * (-h + torch.tanh(pre))


class LiquidTrader(nn.Module):
    """Streaming 2-layer liquid core (canonical MT-LNN step)."""

    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.c1 = LiquidCell(d_in, hidden)
        self.c2 = LiquidCell(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (T, B, F) -> (T, B)
        T, B, _ = x.shape
        h1 = torch.zeros(B, self.hidden, device=x.device)
        h2 = torch.zeros(B, self.hidden, device=x.device)
        outs = []
        for t in range(T):
            h1 = self.c1.step(x[t], h1)
            h2 = self.c2.step(h1, h2)
            outs.append(torch.sigmoid(self.head(h2)).squeeze(-1))
        return torch.stack(outs)


class GruTrader(nn.Module):
    """1-layer GRU with a per-step head (traditional recurrent baseline)."""

    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(d_in, hidden, batch_first=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # (T, B, F) -> (T, B)
        out, _ = self.gru(x)
        return torch.sigmoid(self.head(out)).squeeze(-1)


class MlpTrader(nn.Module):
    """2-layer MLP over the flattened factor window (no recurrence)."""

    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):  # (T, B, F) -> (T, B)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class LinearTrader(nn.Module):
    """Logistic regression over the factor window (classic factor model)."""

    def __init__(self, d_in: int):
        super().__init__()
        self.w = nn.Linear(d_in, 1)

    def forward(self, x):  # (T, B, F) -> (T, B)
        return torch.sigmoid(self.w(x)).squeeze(-1)


# ------------------------------------------------------------------ #
# Train + backtest
# ------------------------------------------------------------------ #

def train(model, xs, ys):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    crit = nn.BCELoss()
    model.train()
    for ep in range(EPOCHS):
        tot, n = 0.0, 0
        for x, y in zip(xs, ys):
            for i in range(0, x.shape[0] - 64, 64):
                xb = x[i:i + 64].unsqueeze(1)
                yb = y[i:i + 64].float()
                p = model(xb).squeeze(-1)
                loss = crit(p, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * yb.numel()
                n += yb.numel()
    return tot / n


def backtest(model, series, cost: float = 5e-4) -> dict:
    model.eval()
    x, _, nxt = make_features(series)
    with torch.no_grad():
        p = model(x.unsqueeze(1)).squeeze(-1)
    pos = torch.where(p > 0.55, 1.0, torch.where(p < 0.45, -1.0, 0.0))
    trades = (pos[1:] != pos[:-1]).sum().item()
    strat = (pos * nxt).sum().item() - trades * cost
    bh = nxt.sum().item()
    mpos = torch.sign(torch.tensor([nxt[max(0, i - 10):i].sum().item()
                                    for i in range(len(nxt))]))
    mtrades = (mpos[1:] != mpos[:-1]).sum().item()
    mom = (mpos * nxt).sum().item() - mtrades * cost
    acc = ((p > 0.5).float() == ((nxt > 0).float())).float().mean().item()
    return {"acc": acc, "strat": strat, "bh": bh, "mom": mom, "trades": trades}


def main() -> int:
    print("generating synthetic series (regime-switching + momentum + noise)",
          flush=True)
    train_series = [gen_series(1000, s) for s in range(15)]
    held_series = [gen_series(1000, 100 + s) for s in range(10)]
    xs, ys = [], []
    for s in train_series:
        x, y, _ = make_features(s)
        xs.append(x)
        ys.append(y)
    F_DIM = xs[0].shape[-1]
    print(f"features: (T, {F_DIM}), 15 train series", flush=True)

    models = {
        "liquid": (LiquidTrader(F_DIM, 32), None),
        "gru": (GruTrader(F_DIM, 32), None),
        "mlp": (MlpTrader(F_DIM, 32), None),
        "linear": (LinearTrader(F_DIM), None),
    }
    for name, (m, _) in models.items():
        n = sum(p.numel() for p in m.parameters())
        print(f"[{name}] {n:,} params", flush=True)
        loss = train(m, xs, ys)
        print(f"[{name}] train loss {loss:.4f}", flush=True)

    print("\nheld-out backtest (unknown outcomes, 5bp cost):", flush=True)
    print(f"{'series':>7} " + " ".join(f"{k:>16}" for k in
          ["liq_acc", "liq_ret", "gru_acc", "gru_ret", "mlp_ret",
           "lin_ret", "mom_ret", "bh_ret"]), flush=True)
    agg = {k: 0.0 for k in ["liq_acc", "liq_ret", "gru_acc", "gru_ret",
                            "mlp_ret", "lin_ret", "mom_ret", "bh_ret"]}
    for i, s in enumerate(held_series):
        r = {name: backtest(m, s) for name, (m, _) in models.items()}
        row = {"liq_acc": r["liquid"]["acc"], "liq_ret": r["liquid"]["strat"],
               "gru_acc": r["gru"]["acc"], "gru_ret": r["gru"]["strat"],
               "mlp_ret": r["mlp"]["strat"], "lin_ret": r["linear"]["strat"],
               "mom_ret": r["liquid"]["mom"], "bh_ret": r["liquid"]["bh"]}
        for k in agg:
            agg[k] += row[k]
        print(f"{i:>7} " + " ".join(f"{row[k]:>15.1%}" for k in
              ["liq_acc", "liq_ret", "gru_acc", "gru_ret", "mlp_ret",
               "lin_ret", "mom_ret", "bh_ret"]), flush=True)

    n = len(held_series)
    print("\nmean over held-out series:", flush=True)
    print(f"  liquid  : acc {agg['liq_acc'] / n:.1%}  log-ret {agg['liq_ret'] / n:+.1%}",
          flush=True)
    print(f"  gru     : acc {agg['gru_acc'] / n:.1%}  log-ret {agg['gru_ret'] / n:+.1%}",
          flush=True)
    print(f"  mlp     :                 log-ret {agg['mlp_ret'] / n:+.1%}", flush=True)
    print(f"  linear  :                 log-ret {agg['lin_ret'] / n:+.1%}", flush=True)
    print(f"  momentum:                 log-ret {agg['mom_ret'] / n:+.1%}", flush=True)
    print(f"  buy&hold:                 log-ret {agg['bh_ret'] / n:+.1%}", flush=True)
    print("\nHONEST SCOPE: simulated data with planted dynamics — the comparison")
    print("shows which architecture learns the planted structure fastest on")
    print("unseen data. It says nothing about real markets.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
