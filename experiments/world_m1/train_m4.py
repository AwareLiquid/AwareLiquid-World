"""World M4 milestone: action-conditioned latent transition.

PLAN M4 definition:
- Hypothesis: an action-vector-conditioned state transition gives end-to-end
  trajectory prediction on the liquid core.
- Metric: trajectory error vs the physics_ops symplectic rollout baseline
  (the analytic ORACLE here: it integrates with the true dynamics).
- Success: within 1.5x of the analytic baseline on the spring suite.

Setup: controlled spring — x'' = -k x - c x' + a_t, controls a_t ~ U(-amax, amax)
per step, k ~ U(1,4) hidden per trajectory. The learned model sees only the
context states (must INFER the regime) plus the future control sequence; the
oracle integrator gets the TRUE k, c. The learned model must predict the
open-loop future trajectory within 1.5x of the oracle.

PLAN field-notes constraint (ACT-JEPA): action + latent prediction train
JOINTLY from day one — the transition head and the encoder share one loss.

Run:
    python -m experiments.world_m1.train_m4 --steps 8000 --seeds 5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LiquidEncoder


# ------------------------------------------------------------------ #
# Data: controlled spring (semi-implicit Euler, same scheme as data.py)
# ------------------------------------------------------------------ #

def spring_control_trajectories(batch: int, T: int, k_lo: float, k_hi: float,
                                a_max: float = 0.5, dt: float = 0.05,
                                c: float = 0.05, seed: int = 0
                                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (traj (B,T,2), actions (B,T), k (B,))."""
    g = torch.Generator().manual_seed(seed)
    k = k_lo + (k_hi - k_lo) * torch.rand(batch, generator=g)
    x = torch.rand(batch, generator=g) * 4 - 2
    v = torch.rand(batch, generator=g) * 2 - 1
    a = (torch.rand(batch, T, generator=g) * 2 - 1) * a_max
    traj = torch.zeros(batch, T, 2)
    for t in range(T):
        traj[:, t, 0] = x
        traj[:, t, 1] = v
        acc = -k * x - c * v + a[:, t]
        v = v + acc * dt
        x = x + v * dt
    return traj, a, k


def analytic_oracle(state: torch.Tensor, actions: torch.Tensor, k: torch.Tensor,
                    c: float = 0.05, dt: float = 0.05) -> torch.Tensor:
    """Symplectic rollout with TRUE dynamics: state (B,2) + (B,T) -> (B,T,2)."""
    x, v = state[:, 0].clone(), state[:, 1].clone()
    out = []
    for t in range(actions.shape[1]):
        out.append(torch.stack([x, v], dim=-1))
        acc = -k * x - c * v + actions[:, t]
        v = v + acc * dt
        x = x + v * dt
    return torch.stack(out, dim=1)


# ------------------------------------------------------------------ #
# Model: liquid encoder + action-conditioned latent transition + decoder
# ------------------------------------------------------------------ #

class ControlWorldModel(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.enc = LiquidEncoder(2, d, n_layers=2)
        self.trans = nn.Sequential(nn.Linear(d + 1, 128), nn.ReLU(),
                                   nn.Linear(128, d))          # (z, a) -> dz
        self.dec = nn.Linear(d, 2)                             # z -> state

    def encode(self, traj: torch.Tensor) -> torch.Tensor:
        return self.enc(traj)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.trans(torch.cat([z, a[:, None]], dim=-1))

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0 (B,D), actions (B,T) -> decoded states (B,T,2)."""
        z = z0
        states = []
        for t in range(actions.shape[1]):
            z = self.step(z, actions[:, t])
            states.append(self.dec(z))
        return torch.stack(states, dim=1)


# ------------------------------------------------------------------ #
# Train / eval
# ------------------------------------------------------------------ #

def train_seed(steps: int, seed: int, k_lo: float, k_hi: float,
               batch: int = 64, t_ctx: int = 16, t_fut: int = 32,
               lr: float = 3e-3) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ControlWorldModel(64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(steps):
        traj, act, _ = spring_control_trajectories(
            batch, t_ctx + t_fut + 1, k_lo, k_hi, seed=seed * 10000 + step)
        traj, act = traj.to(dev), act.to(dev)
        z = model.encode(traj[:, :-1])                         # (B,Tc+Tf,D)
        # joint one-step losses: latent transition (JEPA target detached) +
        # decode supervision — ACT-JEPA lesson: joint, not two-stage
        z_now = z[:, :-1].reshape(-1, z.shape[-1])
        z_next_target = z[:, 1:].reshape(-1, z.shape[-1]).detach()
        a_now = act[:, :z.shape[1] - 1].reshape(-1)   # a[t]: state t -> t+1
        z_next = model.step(z_now, a_now)
        loss_lat = F.mse_loss(z_next, z_next_target)
        loss_dec = F.mse_loss(model.dec(z), traj[:, :-1])
        # open-loop rollout loss: train the ACTUAL eval objective (exposure
        # bias fix — one-step-perfect is not enough, rollout error compounds)
        k_roll = t_fut
        zz = z[:, t_ctx - 1]
        preds, z_tgts, s_tgts = [], [], []
        for t in range(k_roll):
            zz = model.step(zz, act[:, t_ctx - 1 + t])
            preds.append(zz)
            z_tgts.append(z[:, t_ctx + t].detach())
            s_tgts.append(traj[:, t_ctx + t])
        z_pred = torch.stack(preds, 1)
        loss = loss_lat + loss_dec \
            + F.mse_loss(z_pred, torch.stack(z_tgts, 1)) \
            + F.mse_loss(model.dec(z_pred), torch.stack(s_tgts, 1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  seed {seed} step {step+1} loss {loss.item():.4f} "
                  f"(lat {loss_lat.item():.4f} dec {loss_dec.item():.4f})",
                  flush=True)

    # ---- eval: open-loop rollout vs analytic oracle ----
    model.eval()
    n_eval = 256
    traj, act, k = spring_control_trajectories(
        n_eval, t_ctx + t_fut, k_lo, k_hi, seed=999)
    traj, act, k = traj.to(dev), act.to(dev), k.to(dev)
    ctx, fut = traj[:, :t_ctx], traj[:, t_ctx:]
    # traj[t] + act[t] -> traj[t+1]; so predicting traj[t_ctx:] applies
    # actions starting at t_ctx-1.
    fut_act = act[:, t_ctx - 1:t_ctx - 1 + t_fut]
    with torch.no_grad():
        z_ctx = model.encode(ctx)
        pred = model.rollout(z_ctx[:, -1], fut_act)            # (B,Tf,2)
        err_learned = F.mse_loss(pred, fut).item()
        oracle = analytic_oracle(ctx[:, -1], fut_act, k)       # true dynamics
        err_oracle = F.mse_loss(oracle, fut).item()
    return {"seed": seed, "err_learned": err_learned,
            "err_oracle": err_oracle,
            "ratio": err_learned / max(err_oracle, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k-lo", type=float, default=1.0)
    ap.add_argument("--k-hi", type=float, default=4.0)
    args = ap.parse_args()

    print(f"=== World M4: action-conditioned transition (controlled spring, "
          f"hidden k~U({args.k_lo},{args.k_hi})) ===")
    rows = []
    for seed in range(args.seeds):
        r = train_seed(args.steps, seed, args.k_lo, args.k_hi)
        rows.append(r)
        print(f"  seed {seed}: learned {r['err_learned']:.5f} vs oracle "
              f"{r['err_oracle']:.5f}  ratio {r['ratio']:.2f}x")

    wins = sum(r["ratio"] <= 1.5 for r in rows)
    print(f"\nwithin 1.5x oracle in {wins}/{len(rows)} seeds")
    print("VERDICT:", "PASS" if wins >= 3 else "FAIL")


if __name__ == "__main__":
    main()
