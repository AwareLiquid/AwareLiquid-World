"""Physics trajectory generators for the world-model line.

Two deterministic systems with analytic ground truth, from the PLAN's
"spring/orbit" modality:
- spring: damped harmonic oscillator, state = (pos, vel)
- orbit: Kepler 2-body, state = (x, y, vx, vy)

Each generator emits (B, T, state_dim) trajectories plus the next-step
targets for the probe evaluation.
"""

from __future__ import annotations

import torch


def spring_trajectories(batch: int, T: int, dt: float = 0.05,
                        k: float = 2.0, c: float = 0.05,
                        seed: int = 0) -> torch.Tensor:
    """Damped oscillator: x'' = -k x - c x'. State (pos, vel)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, device="cpu", generator=g) * 4 - 2
    v = torch.rand(batch, device="cpu", generator=g) * 2 - 1
    traj = torch.zeros(batch, T, 2)
    for t in range(T):
        traj[:, t, 0] = x
        traj[:, t, 1] = v
        a = -k * x - c * v
        v = v + a * dt
        x = x + v * dt
    return traj


def orbit_trajectories(batch: int, T: int, dt: float = 0.05,
                       seed: int = 0) -> torch.Tensor:
    """Kepler orbit via symplectic Euler. State (x, y, vx, vy)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(batch, generator=g) * 0.8 + 0.5
    theta = torch.rand(batch, generator=g) * 6.2832
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    vx = -y / (r ** 1.5)
    vy = x / (r ** 1.5)
    traj = torch.zeros(batch, T, 4)
    for t in range(T):
        traj[:, t, 0] = x
        traj[:, t, 1] = y
        traj[:, t, 2] = vx
        traj[:, t, 3] = vy
        r3 = (x ** 2 + y ** 2) ** 1.5
        ax, ay = -x / r3, -y / r3
        vx = vx + ax * dt
        vy = vy + ay * dt
        x = x + vx * dt
        y = y + vy * dt
    return traj


def make_dataset(kind: str, batch: int, T: int, seed: int = 0
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Trajectories + next-step targets (targets = trajectory[1:])."""
    if kind == "spring":
        traj = spring_trajectories(batch, T + 1, seed=seed)
        return traj[:, :-1], traj[:, 1:]
    if kind == "orbit":
        traj = orbit_trajectories(batch, T + 1, seed=seed)
        return traj[:, :-1], traj[:, 1:]
    raise ValueError(kind)
