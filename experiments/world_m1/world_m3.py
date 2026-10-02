"""World M3 milestone: explicit latent variables — multi-path rollout.

Plan (PLAN.md M3): a latent-factor dimension yields multiple future
evolution paths. Physics modality (spring): trajectories differ by their
initial conditions; the encoder's posterior z must capture WHICH mode a
trajectory is in, so that sampling z ~ N(0,1) at inference produces a
SPREAD of futures that covers the observed futures.

Protocol:
  - Encode the FIRST half of a spring trajectory -> posterior z (B, L).
  - Decoder: (shared, z) -> the SECOND half (the future), MSE-trained,
    with a KL term pulling the posterior toward N(0,1) (weight beta).
  - Eval (held-out): sample K prior paths per trajectory; coverage =
    fraction of true futures lying within the sampled-path envelope
    (per-dim 10-90 percentile band). Baseline = single mean path (K=1).

Success (PLAN): conditioned rollouts cover the observed futures; honest
failure mode documented otherwise.

Run:  python -m experiments.world_m1.world_m3
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import make_dataset
from .encoder import LiquidEncoder


class LatentWorld(nn.Module):
    """First half -> (shared, z) -> decoded second half (multi-path)."""

    def __init__(self, in_dim: int, d: int = 64, latent: int = 8,
                 future: int = 32):
        super().__init__()
        self.encoder = LiquidEncoder(in_dim, d=d, n_layers=2)
        self.shared = nn.Linear(d, d - latent)
        self.to_z = nn.Linear(d, latent)
        self.decoder = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, future * in_dim))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T/2, in) -> (shared (B, d-L), z (B, L))."""
        z_seq = self.encoder(x)
        state = z_seq[:, -1]
        return self.shared(state), self.to_z(state)

    def decode(self, shared: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(B, d-L), (B, n_paths, L) -> (B, n_paths, future, in)."""
        b, n_paths, _ = z.shape
        sh = shared.unsqueeze(1).expand(b, n_paths, -1)
        h = torch.cat([sh, z], dim=-1)
        return self.decoder(h).view(b, n_paths, -1, 2)


def train(model: LatentWorld, kind: str, steps: int = 6000, batch: int = 64,
          latent: int = 8, beta: float = 0.1, lr: float = 3e-3,
          seed: int = 0) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        X, Y = make_dataset(kind, batch, 64, seed=seed * 10000 + step)
        half = X.shape[1] // 2
        past, future = X[:, :half], X[:, half:]
        shared, z = model.encode(past)
        pred = model.decode(shared, z.unsqueeze(1))[:, 0]     # (B, 32, 2)
        mse = F.mse_loss(pred, future)
        kl = 0.5 * (z.square() + 1 - 1 - 0).sum(-1).mean()    # KL(N(z,1), N(0,1))
        loss = mse + beta * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  step {step+1} mse {mse.item():.4f} kl {kl.item():.4f}",
                  flush=True)


def coverage(model: LatentWorld, kind: str, k_paths: int = 20,
             seed: int = 777) -> dict:
    """Held-out: K prior paths per trajectory; coverage of true futures."""
    model.eval()
    X, Y = make_dataset(kind, 256, 64, seed=seed)
    half = X.shape[1] // 2
    past, future = X[:, :half], X[:, half:]
    with torch.no_grad():
        shared, _ = model.encode(past)
        z = torch.randn(shared.shape[0], k_paths, 8)
        paths = model.decode(shared, z)                     # (B, K, 32, 2)
    lo = paths.quantile(0.1, dim=1)
    hi = paths.quantile(0.9, dim=1)
    inside = ((future >= lo) & (future <= hi)).float().mean().item()
    # baseline: single mean path envelope collapses to zero coverage
    base_lo = paths.mean(dim=1, keepdim=True).expand(-1, 1, -1, -1)
    base_cov = ((future >= base_lo[:, 0]) & (future <= base_lo[:, 0])).float().mean().item()
    return {"coverage_k20": inside, "coverage_mean_path": base_cov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring", choices=("spring", "orbit"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"=== World M3: latent multi-path rollout ({args.kind}) ===")
    rows = []
    for seed in range(args.seeds):
        model = LatentWorld(2 if args.kind == "spring" else 4)
        train(model, args.kind, steps=args.steps, seed=seed)
        cov = coverage(model, args.kind, seed=1000 + seed)
        rows.append(cov)
        print(f"  seed {seed}: coverage_k20 {cov['coverage_k20']:.3f}  "
              f"mean_path {cov['coverage_mean_path']:.3f}")

    covs = [r["coverage_k20"] for r in rows]
    print(f"\ncoverage: {sum(covs)/len(covs):.3f} (mean over seeds)")
    print("VERDICT:", "PASS" if all(c > 0.3 for c in covs) else "FAIL")


if __name__ == "__main__":
    main()
