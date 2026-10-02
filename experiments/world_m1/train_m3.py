"""World M3 milestone: explicit latent variables -> multi-path future rollout.

PLAN M3 definition:
- Hypothesis: a latent-factor dimension yields multiple future evolution
  paths, useful for decision support.
- Metric: rollout diversity vs ground-truth coverage (physics modality).
- Success: conditioned rollouts cover the observed futures; honest failure
  mode documented.

Design (why hidden parameters): the M1/M2 spring/orbit data is deterministic
per trajectory, so multiple futures are meaningless. M3 introduces STRUCTURED
ambiguity — each trajectory draws a hidden stiffness k ~ U(1,4) and only a
SHORT context is observed — the analog of an industrial system whose hidden
degradation mode must be inferred. A latent-variable model should emit
multiple paths that COVER the true future; a collapsed model emits one path.

Training: winner-take-all masked-latent prediction — the best path (min loss)
drives the gradient, which forces path specialisation (coverage) instead of
path averaging.

Run:
    python -m experiments.world_m1.train_m3 --steps 8000 --seeds 5
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.objectives import LatentVariableHead

from .encoder import LiquidEncoder


# ------------------------------------------------------------------ #
# Data: spring with a per-trajectory hidden stiffness k ~ U(k_lo, k_hi)
# ------------------------------------------------------------------ #

def spring_randk_trajectories(batch: int, T: int, k_lo: float, k_hi: float,
                              dt: float = 0.05, c: float = 0.05,
                              seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = k_lo + (k_hi - k_lo) * torch.rand(batch, generator=g)  # (B,) hidden
    x = torch.rand(batch, generator=g) * 4 - 2
    v = torch.rand(batch, generator=g) * 2 - 1
    traj = torch.zeros(batch, T, 2)
    for t in range(T):
        traj[:, t, 0] = x
        traj[:, t, 1] = v
        a = -k * x - c * v
        v = v + a * dt
        x = x + v * dt
    return traj


# ------------------------------------------------------------------ #
# Model: liquid encoder + latent-variable head + path->future decoder
# ------------------------------------------------------------------ #

class MultiPathWorldModel(nn.Module):
    def __init__(self, in_dim: int = 2, d: int = 64, n_latent: int = 8,
                 n_paths: int = 8, t_future: int = 32):
        super().__init__()
        self.enc = LiquidEncoder(in_dim, d, n_layers=2)
        self.head = LatentVariableHead(d, n_latent, n_paths)
        self.n_paths = n_paths
        self.t_future = t_future
        self.future = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, t_future * d),
        )

    def encode(self, traj: torch.Tensor) -> torch.Tensor:
        return self.enc(traj)

    def predict_paths(self, context: torch.Tensor) -> torch.Tensor:
        """context (B, Tc, in) -> (B, P, T_f, d) path latents."""
        z = self.enc(context)
        h = z[:, -1]                                        # (B, d)
        b = h.shape[0]
        zs = torch.randn(b, self.n_paths, self.head.n_latent, device=h.device)
        cond = self.head(h, zs)                             # (B, P, d)
        pred = self.future(cond).view(b, self.n_paths, self.t_future, -1)
        return pred


# ------------------------------------------------------------------ #
# Train / eval
# ------------------------------------------------------------------ #

def wta_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Winner-take-all path loss.

    pred:   (B, P, T, D) path futures
    target: (B, T, D)    true future latents (detached)
    Returns (loss, per-sample best-path index).
    """
    tgt = target.unsqueeze(1)                                # (B,1,T,D)
    mse = ((pred - tgt) ** 2).mean(dim=(2, 3))               # (B, P)
    cos = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).mean(dim=-1)  # (B,P)
    per_path = mse + cos
    best, idx = per_path.min(dim=1)
    return best.mean(), idx


@torch.no_grad()
def evaluate(model: MultiPathWorldModel, k_lo: float, k_hi: float,
             n_eval: int = 512, t_ctx: int = 8, t_fut: int = 32,
             seed: int = 999) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    traj = spring_randk_trajectories(n_eval, t_ctx + t_fut, k_lo, k_hi,
                                     seed=seed).to(dev)
    ctx, fut = traj[:, :t_ctx], traj[:, t_ctx:]
    z_ctx = model.encode(ctx)
    z_fut = model.encode(fut)                                # (B, T_f, d)
    h_last = z_ctx[:, -1]                                    # (B, d)

    pred = model.predict_paths(ctx)                          # (B, P, T_f, d)
    tgt = z_fut.unsqueeze(1)
    per_path_err = ((pred - tgt) ** 2).mean(dim=(2, 3)).sqrt()   # (B, P)
    err_best = per_path_err.min(dim=1).values.mean().item()      # coverage
    err_median = per_path_err.median(dim=1).values.mean().item()  # single-path ref

    # single-path baseline: persistence in latent (repeat last context latent)
    base_err = ((h_last.unsqueeze(1) - z_fut) ** 2).mean(dim=(1, 2)).sqrt().mean().item()

    # diversity: mean pairwise distance between path predictions
    diffs = pred.unsqueeze(2) - pred.unsqueeze(1)            # (B, P, P, T, d)
    n = pred.shape[1]
    pair = torch.triu(torch.ones(n, n, dtype=torch.bool, device=pred.device), 1)
    diversity = diffs.pow(2).mean(dim=(3, 4))[:, pair].sqrt().mean().item()

    return {"coverage_err": err_best, "base_err": base_err,
            "median_err": err_median,
            "rel_gain": (base_err - err_best) / base_err,
            "diversity": diversity}


def train_seed(steps: int, seed: int, k_lo: float, k_hi: float,
               batch: int = 64, t_ctx: int = 8, t_fut: int = 32,
               n_latent: int = 8, n_paths: int = 8,
               lr: float = 3e-3) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiPathWorldModel(2, 64, n_latent, n_paths, t_fut).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step in range(steps):
        traj = spring_randk_trajectories(batch, t_ctx + t_fut, k_lo, k_hi,
                                         seed=seed * 10000 + step).to(dev)
        ctx, fut = traj[:, :t_ctx], traj[:, t_ctx:]
        with torch.no_grad():
            z_fut = model.encode(fut)                        # target latents
        pred = model.predict_paths(ctx)
        loss, _ = wta_loss(pred, z_fut)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  seed {seed} step {step+1} loss {loss.item():.4f}", flush=True)

    r = evaluate(model, k_lo, k_hi)
    r["seed"] = seed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k-lo", type=float, default=1.0)
    ap.add_argument("--k-hi", type=float, default=4.0)
    ap.add_argument("--n-latent", type=int, default=8)
    ap.add_argument("--n-paths", type=int, default=8)
    args = ap.parse_args()

    print(f"=== World M3: explicit latent variables / multi-path rollout "
          f"(spring hidden k~U({args.k_lo},{args.k_hi}), {args.n_paths} paths) ===")
    rows = []
    for seed in range(args.seeds):
        r = train_seed(args.steps, seed, args.k_lo, args.k_hi,
                       n_latent=args.n_latent, n_paths=args.n_paths)
        rows.append(r)
        print(f"  seed {seed}: coverage err {r['coverage_err']:.4f} vs base "
              f"{r['base_err']:.4f}  rel_gain {r['rel_gain']*100:.1f}%  "
              f"median_path {r['median_err']:.4f}  diversity {r['diversity']:.4f}")

    wins = sum(r["rel_gain"] > 0 for r in rows)
    # diversity floor: paths must not collapse to one (empirical: >2% of base)
    diverse = sum(r["diversity"] > 0.02 * r["base_err"] for r in rows)
    print(f"\ncoverage beats persistence in {wins}/{len(rows)} seeds; "
          f"diverse (non-collapsed) in {diverse}/{len(rows)}")
    print("VERDICT:", "PASS" if wins >= 3 and diverse >= 3 else "FAIL")


if __name__ == "__main__":
    main()
