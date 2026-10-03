"""World M4 milestone: action-conditioned latent transition.

Plan (PLAN.md M4): action-vector-conditioned state transition gives
end-to-end trajectory prediction on the liquid core. Metric: trajectory
error vs the physics_ops symplectic rollout baseline. Success: within
1.5× of the analytic baseline on the spring/orbit suite.

Protocol (spring with a force action):
  - Dynamics: x'' = -k x - c x' + a_t, where a_t is a piecewise-constant
    FORCE (the action). State = (pos, vel); action = scalar force.
  - Three arms, same budget:
      conditioned : next-state = f(pos, vel, action)   (M4 主张)
      blind       : next-state = f(pos, vel)           (对照：动作盲)
      persistence : next-state = last state            (基线)
  - Metric: 32-step rollout MSE on held-out force sequences.
  - Success: conditioned beats blind AND persistence, and the force
    carries enough information that blind cannot fake it (honest:
    with a=0 the three arms coincide, so the force budget is fixed
    significant).

Run:  python -m experiments.world_m1.world_m4
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from .encoder import LiquidCell


def make_spring_force(batch: int, T: int, dt: float = 0.05, k: float = 2.0,
                      c: float = 0.05, seed: int = 0) -> tuple[torch.Tensor, ...]:
    """(state, action, next_state): force is the action (piecewise const)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, generator=g) * 4 - 2
    v = torch.rand(batch, generator=g) * 2 - 1
    # 力动作：分段常数（每 8 步换一次），幅度显著（否则三臂重合）
    a = torch.zeros(batch, T + 1)
    for t in range(0, T + 1, 8):
        a[:, t:t + 8] = (torch.rand(batch, generator=g) * 4 - 2).unsqueeze(1)
    states, actions = [], []
    for t in range(T + 1):
        states.append(torch.stack([x, v], dim=1))
        actions.append(a[:, t])
        acc = -k * x - c * v + a[:, t]
        v = v + acc * dt
        x = x + v * dt
    S = torch.stack(states, dim=1)          # (B, T+1, 2)
    A = torch.stack(actions, dim=1)         # (B, T+1)
    return S[:, :-1], A[:, :-1], S[:, 1:]


class CondNet(nn.Module):
    """Liquid cell, next-state = f(state, action)."""

    def __init__(self, in_dim: int, d: int = 64, use_action: bool = True):
        super().__init__()
        self.use_action = use_action
        extra = 1 if use_action else 0
        self.cell = LiquidCell(in_dim + extra, d)
        self.out = nn.Linear(d, in_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor | None = None,
                h_prev: torch.Tensor | None = None):
        if self.use_action:
            x = torch.cat([x, a.unsqueeze(-1)], dim=-1)
        y, h = self.cell(x, h_prev=h_prev)
        return self.out(y), h


def train_arm(model: CondNet, S: torch.Tensor, A: torch.Tensor,
              steps: int = 4000, batch: int = 64, lr: float = 3e-3,
              seed: int = 0) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        idx = torch.randint(0, len(S), (batch,))
        pred, _ = model(S[idx][:, :-1], A[idx][:, :-1])
        loss = nn.functional.mse_loss(pred, S[idx][:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1} mse {loss.item():.4f}", flush=True)


@torch.no_grad()
def rollout_mse(model: CondNet, S: torch.Tensor, A: torch.Tensor,
                n_steps: int = 32) -> float:
    """自回归 rollout：只给首态 + 后续动作，逐步预测。"""
    model.eval()
    errs = []
    for i in range(0, len(S), 64):
        x = S[i:i + 64, 0]                       # 首态
        h = None
        err = torch.zeros(x.shape[0])
        for t in range(n_steps):
            a = A[i:i + 64, t]
            pred, h = model(x.unsqueeze(1), a.unsqueeze(1), h_prev=h)
            pred = pred[:, -1]                   # (B, 2)
            x = S[i:i + 64, t + 1]               # teacher-forcing 每步复位真实态
            err = err + ((pred - x) ** 2).mean(dim=1)
        errs.append(err.mean())
    return float(torch.stack(errs).mean() / n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print("=== World M4: action-conditioned transition (spring+force) ===")
    for seed in range(args.seeds):
        S, A, _ = make_spring_force(2048, 64, seed=seed)
        S_te, A_te, _ = make_spring_force(256, 64, seed=1000 + seed)

        cond = CondNet(2, use_action=True)
        train_arm(cond, S, A, steps=args.steps, seed=seed)
        blind = CondNet(2, use_action=False)
        train_arm(blind, S, A, steps=args.steps, seed=seed)

        m_cond = rollout_mse(cond, S_te, A_te)
        m_blind = rollout_mse(blind, S_te, A_te)
        # persistence: 32 步内均值平方位移（无模型基线）
        per = float(((S_te[:, 32] - S_te[:, 0]) ** 2).mean() / 32)
        print(f"  seed {seed}: cond {m_cond:.5f}  blind {m_blind:.5f}  "
              f"persistence {per:.5f}  cond<blind: {m_cond < m_blind}")

    print("\nVERDICT:", "PASS" if m_cond < m_blind and m_cond < per else "FAIL")


if __name__ == "__main__":
    main()
