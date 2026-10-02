"""World M7 milestone: latent-space MPC planning (V-JEPA 2-AC lesson).

PLAN M7 hypothesis: CEM/MPPI over imagined latent rollouts, with the goal
state as the energy, gives closed-loop planning at edge speed.

Design:
- world model = the M4 action-conditioned control model (trained here briefly
  on the controlled spring, hidden k per trajectory);
- planning task: from a start state, find an H-step action sequence that drives
  the decoded state to a GOAL state (spring is controllable);
- planner = CEM over action sequences: sample N sequences, roll out in latent
  space, score by final latent distance to the goal's encoding, refit the
  sampling distribution for ~5 rounds;
- baseline = greedy: at each step pick the action (from a small discrete set)
  minimizing the one-step predicted goal distance;
- metric: goal-reaching success rate (final state within tol) + wall time/plan.

Success (pre-registered): CEM success > greedy success on >=60% of the pairs,
or CEM >= greedy + 15 points overall; wall time reported honestly.

Run:
    python -m experiments.world_m1.train_m7 --steps 8000
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from .train_m4 import ControlWorldModel, spring_control_trajectories


def train_world_model(steps: int, seed: int, k_lo: float, k_hi: float,
                      batch: int = 64, t_ctx: int = 16, t_fut: int = 32,
                      lr: float = 3e-3) -> ControlWorldModel:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ControlWorldModel(64).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        traj, act, _ = spring_control_trajectories(
            batch, t_ctx + t_fut + 1, k_lo, k_hi, seed=seed * 10000 + step)
        traj, act = traj.to(dev), act.to(dev)
        z = model.encode(traj[:, :-1])
        z_now = z[:, :-1].reshape(-1, z.shape[-1])
        z_next = z[:, 1:].reshape(-1, z.shape[-1]).detach()
        a_now = act[:, :z.shape[1] - 1].reshape(-1)
        loss = F.mse_loss(model.step(z_now, a_now), z_next) \
            + F.mse_loss(model.dec(z), traj[:, :-1])
        # open-loop rollout loss (the M4 exposure-bias fix)
        zz = z[:, t_ctx - 1]
        preds, st = [], []
        for t in range(t_fut):
            zz = model.step(zz, act[:, t_ctx - 1 + t])
            preds.append(zz)
            st.append(traj[:, t_ctx + t])
        zp = torch.stack(preds, 1)
        loss = loss + F.mse_loss(zp, z[:, t_ctx:t_ctx + t_fut].detach()) \
            + F.mse_loss(model.dec(zp), torch.stack(st, 1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  [wm] step {step+1} loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def plan_cem(model: ControlWorldModel, z0: torch.Tensor, goal_z: torch.Tensor,
             horizon: int, n_samples: int = 128, n_elite: int = 16,
             rounds: int = 5, a_max: float = 0.5) -> torch.Tensor:
    """CEM over action sequences: (B,H). z0/goal_z (B,D).

    Energy is computed in DECODED state space (goal_state vs rollout state) —
    latent-space distance is warped and unusable as an MPC objective.
    """
    b = z0.shape[0]
    dev = z0.device
    goal_state = model.dec(goal_z)                           # (B,2)
    mean = torch.zeros(b, horizon, device=dev)
    std = torch.full((b, horizon), a_max, device=dev)
    for _ in range(rounds):
        acts = mean[:, None, :] + std[:, None, :] * torch.randn(
            b, n_samples, horizon, device=dev)
        acts = acts.clamp(-a_max, a_max)
        flat = acts.reshape(b * n_samples, horizon)
        z = z0[:, None, :].expand(b, n_samples, -1).reshape(b * n_samples, -1)
        for t in range(horizon):
            z = model.step(z, flat[:, t])
        states = model.dec(z).reshape(b, n_samples, -1)      # (B,N,2)
        d = ((states - goal_state[:, None, :]) ** 2).sum(-1)  # (B, N)
        elite = torch.topk(d, n_elite, largest=False).indices  # (B, K)
        elite_acts = acts.gather(1, elite[..., None].expand(-1, -1, horizon))
        mean = elite_acts.mean(1)
        std = elite_acts.std(1) + 1e-3
    return mean


@torch.no_grad()
def plan_greedy(model: ControlWorldModel, z0: torch.Tensor, goal_z: torch.Tensor,
                horizon: int, a_choices: int = 9, a_max: float = 0.5) -> torch.Tensor:
    """Greedy one-step lookahead over a discrete action grid."""
    b = z0.shape[0]
    dev = z0.device
    grid = torch.linspace(-a_max, a_max, a_choices, device=dev)
    z = z0
    acts = []
    for _ in range(horizon):
        cand = z[:, None, :].expand(b, a_choices, -1)        # (B, A, D)
        a = grid[None, :].expand(b, a_choices)               # (B, A)
        zc = model.step(cand.reshape(b * a_choices, -1), a.reshape(-1))
        zc = zc.reshape(b, a_choices, -1)
        d = ((zc - goal_z[:, None, :]) ** 2).sum(-1)         # (B, A)
        pick = d.argmin(1)
        a_best = grid[pick]
        acts.append(a_best)
        z = model.step(z, a_best)
    return torch.stack(acts, 1)


def evaluate(model: ControlWorldModel, k_lo: float, k_hi: float,
             n_pairs: int = 64, t_ctx: int = 16, horizon: int = 16,
             tol: float = 0.10) -> dict:
    """Well-posed task: context = first t_ctx steps; start = latent at t_ctx-1;
    goal = the state `horizon` steps ahead, which is reached by the
    trajectory's OWN actions a[t_ctx-1 : t_ctx-1+horizon] — a solution exists
    by construction. The planner (free actions) must reach that goal."""
    model.eval()
    dev = next(model.parameters()).device
    T = t_ctx + horizon + 1
    traj, act, _ = spring_control_trajectories(n_pairs, T, k_lo, k_hi, seed=4242)
    traj, act = traj.to(dev), act.to(dev)
    with torch.no_grad():
        z_ctx = model.encode(traj[:, :t_ctx])
        z0 = z_ctx[:, -1]                                    # state t_ctx-1
        goal_state = traj[:, t_ctx - 1 + horizon]            # state ahead
        # oracle: follow the trajectory's own actions (upper-bound reference)
        z = z0
        for t in range(horizon):
            z = model.step(z, act[:, t_ctx - 1 + t])
        oracle_err = ((model.dec(z) - goal_state) ** 2).sum(-1).sqrt()

    t0 = time.time()
    acts_cem = plan_cem(model, z0, model.encode(
        traj[:, t_ctx - 1 + horizon:t_ctx + horizon])[:, -1], horizon)
    t_cem = time.time() - t0
    t0 = time.time()
    acts_grd = plan_greedy(model, z0, model.encode(
        traj[:, t_ctx - 1 + horizon:t_ctx + horizon])[:, -1], horizon)
    t_grd = time.time() - t0

    out = {"oracle": {"mean_err": float(oracle_err.mean()),
                      "success": float((oracle_err < tol).float().mean())}}
    for name, acts, dt in (("cem", acts_cem, t_cem), ("greedy", acts_grd, t_grd)):
        z = z0
        for t in range(horizon):
            z = model.step(z, acts[:, t])
        err = ((model.dec(z) - goal_state) ** 2).sum(-1).sqrt()
        out[name] = {"success": float((err < tol).float().mean()),
                     "mean_err": float(err.mean()),
                     "sec_per_plan": dt / n_pairs * 1000}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--k-lo", type=float, default=1.0)
    ap.add_argument("--k-hi", type=float, default=4.0)
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    print(f"=== World M7: latent-space MPC (CEM vs greedy), horizon "
          f"{args.horizon}, controlled spring ===")
    model = train_world_model(args.steps, 0, args.k_lo, args.k_hi)
    res = evaluate(model, args.k_lo, args.k_hi, horizon=args.horizon)
    for name, r in res.items():
        timing = f"  {r['sec_per_plan']:.1f} ms/plan" if "sec_per_plan" in r else "  (reference)"
        print(f"  {name}: success {r['success']*100:.1f}%  mean_err "
              f"{r['mean_err']:.4f}{timing}")
    ok = res["cem"]["success"] > res["greedy"]["success"] + 0.15
    print(f"\nCEM vs greedy: {res['cem']['success']*100:.1f}% vs "
          f"{res['greedy']['success']*100:.1f}%")
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
