"""Minimal liquid trajectory encoder for the world-model line.

Self-contained (no M1 import): closed-form liquid time-constant (CfLTC)
recurrence — per-unit learnable tau, continuous-time decay, O(1) carried
state — mapping (B, T, state_dim) -> (B, T, D) latent states.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidCell(nn.Module):
    """Closed-form LTC cell: h_t = decay*h_{t-1} + (1-decay)*tanh(Wx + Uh + b).

    decay per unit = exp(-dt/tau), tau = softplus(log_tau) + tau_min.
    """

    def __init__(self, in_dim: int, d: int, tau_min: float = 0.1):
        super().__init__()
        self.d = d
        self.W = nn.Linear(in_dim, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.U.weight, std=0.02)
        self.b = nn.Parameter(torch.zeros(d))
        # Tau ladder across units: geometric from 0.5 to 90 steps
        # (the O-series range) — the slow units carry long-horizon
        # dynamics, the fast units the instantaneous signal.
        lo, hi = 0.5, 90.0
        taus = lo * (hi / lo) ** (torch.arange(d) / max(d - 1, 1))
        self.log_tau = nn.Parameter(torch.log(torch.clamp(taus - tau_min, min=1e-3)))
        self.tau_min = tau_min

    def forward(self, x: torch.Tensor,
                h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, in) -> (y: (B,T,d), h_last: (B,d))."""
        B, T, _ = x.shape
        tau = F.softplus(self.log_tau) + self.tau_min      # (d,)
        decay = torch.exp(-1.0 / tau).view(1, 1, self.d)   # (1,1,d)
        z = torch.tanh(self.W(x) + self.b)                 # (B,T,d)
        h = torch.zeros(B, self.d, device=x.device)
        if h_prev is not None:
            h = h + h_prev
        hs: list[torch.Tensor] = []
        for t in range(T):
            rec = torch.tanh(self.U(h))
            h = decay[:, 0] * h + (1 - decay[:, 0]) * (z[:, t] + rec)
            hs.append(h)
        return torch.stack(hs, dim=1), h


class LiquidEncoder(nn.Module):
    """Stacked liquid cells: (B, T, state_dim) -> (B, T, D) latent."""

    def __init__(self, in_dim: int, d: int = 64, n_layers: int = 2):
        super().__init__()
        self.cells = nn.ModuleList(
            LiquidCell(in_dim if i == 0 else d, d) for i in range(n_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for cell in self.cells:
            h, _ = cell(h)
        return self.norm(h)
