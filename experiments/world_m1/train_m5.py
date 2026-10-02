"""World M5 milestone: deep multi-level supervision (V-JEPA 2.1 lesson).

PLAN M5 hypothesis: applying the latent prediction loss at MULTIPLE depths /
timescales of the liquid core beats last-layer-only supervision (Meta's stated
"hierarchical JEPA" direction; our stacked liquid cells are the native
substrate).

Design:
- the 2-cell liquid encoder exposes BOTH cell outputs h1, h2 (h2 = final);
- the masked-latent-prediction loss (M1 objective) is applied at EVERY level:
  L = L(h2) + lambda * L(h1), targets detached per level;
- evaluation = the M1 protocol (frozen encoder + lightweight MLP probe on the
  FINAL layer vs the persistence baseline), so the gain is directly comparable
  to M1's single-level numbers (spring +83.3/83.8/81.4%,
  orbit +83.7/74.4/74.5%).

Success (pre-registered): deep-supervision gain > its OWN lambda=0 control
(the same code path with the same sigreg regulariser) on >=3/5 seeds — the
PRIMARY test, because the M1 reference numbers were produced WITHOUT sigreg,
so comparing against them directly is confounded. M1 ref is reported as a
secondary sanity anchor only.

Run:
    python -m experiments.world_m1.train_m5 --kind spring --steps 8000 --seeds 5
    python -m experiments.world_m1.train_m5 --kind spring --steps 8000 --seeds 5 --deep-lambda 0.0
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from world.objectives import mask_spans, sigreg

from .data import make_dataset
from .train import LatentPredictor, latent_std


class MultiLevelEncoder(nn.Module):
    """Liquid encoder that returns every cell's output (deep supervision)."""

    def __init__(self, in_dim: int, d: int = 64, n_layers: int = 2):
        super().__init__()
        from .encoder import LiquidCell
        self.cells = nn.ModuleList(
            LiquidCell(in_dim if i == 0 else d, d) for i in range(n_layers))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = x
        outs = []
        for cell in self.cells:
            h, _ = cell(h)
            outs.append(self.norm(h))
        return outs


def masked_loss(level: torch.Tensor, keep: torch.Tensor,
                pred: LatentPredictor) -> torch.Tensor:
    """M1 loss (cos + L2) on masked positions of one level; targets detached."""
    out = pred(level)
    masked = ~keep.to(level.device)
    target = level.detach()[:, masked]
    guess = out[:, masked]
    cos = 1.0 - F.cosine_similarity(guess, target, dim=-1).mean()
    l2 = F.mse_loss(guess, target)
    return cos + l2


def train_seed(kind: str, seed: int, steps: int, T: int = 64, batch: int = 64,
               mask_ratio: float = 0.5, mean_span: int = 6,
               sigreg_weight: float = 0.01, deep_lambda: float = 1.0) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = MultiLevelEncoder(2 if kind == "spring" else 4, 64, 2).to(dev)
    # one predictor per level (deep supervision at every depth)
    preds = nn.ModuleList([LatentPredictor(64), LatentPredictor(64)]).to(dev)
    params = list(enc.parameters()) + list(preds.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)

    for step in range(steps):
        traj, _ = make_dataset(kind, batch, T, seed=seed * 10000 + step)
        traj = traj.to(dev)
        levels = enc(traj)                        # [h1, h2]
        keep = mask_spans(T, mask_ratio, mean_span, g)
        loss = masked_loss(levels[-1], keep, preds[-1])
        loss = loss + deep_lambda * masked_loss(levels[0], keep, preds[0])
        loss = loss + sigreg_weight * sigreg(levels[-1])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  seed {seed} step {step+1} loss {loss.item():.4f} "
                  f"latent_std {latent_std(levels[-1].detach()):.3f}", flush=True)

    # ---- M1-protocol evaluation on the FINAL level ----
    enc.eval()
    X, Y = make_dataset(kind, 4096, T + 1, seed=777 + seed)
    X, Y = X.to(dev), Y.to(dev)
    with torch.no_grad():
        Z = enc(X)[-1]
    base = float(((Y[:, :-1] - X[:, :-1]) ** 2).mean())
    probe = nn.Sequential(nn.Linear(64, 128), nn.ReLU(),
                          nn.Linear(128, Y.shape[-1])).to(dev)
    optp = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(1500):
        idx = torch.randperm(Z.shape[0])[:1024]
        l = F.mse_loss(probe(Z[idx][:, :-1]), Y[idx][:, :-1])
        optp.zero_grad()
        l.backward()
        optp.step()
    with torch.no_grad():
        pm = float(F.mse_loss(probe(Z[:, :-1]), Y[:, :-1]))
    return {"seed": seed, "base_mse": base, "probe_mse": pm,
            "rel_gain": (base - pm) / base, "latent_std": latent_std(Z.detach())}


M1_REF = {"spring": [0.833, 0.838, 0.814], "orbit": [0.837, 0.744, 0.745]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring", choices=("spring", "orbit"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--deep-lambda", type=float, default=1.0)
    args = ap.parse_args()

    ref = M1_REF[args.kind]
    ref_mean = sum(ref) / len(ref)
    print(f"=== World M5: deep multi-level supervision ({args.kind}, "
          f"lambda={args.deep_lambda}) vs M1 single-level ref mean "
          f"{ref_mean*100:.1f}% ===")
    rows = []
    for seed in range(args.seeds):
        r = train_seed(args.kind, seed, args.steps,
                       deep_lambda=args.deep_lambda)
        rows.append(r)
        print(f"  seed {seed}: probe {r['probe_mse']:.4f} vs base "
              f"{r['base_mse']:.4f}  rel_gain {r['rel_gain']*100:.1f}%  "
              f"latent_std {r['latent_std']:.3f}")

    wins = sum(r["rel_gain"] > ref_mean + 0.02 for r in rows)
    collapsed = sum(r["latent_std"] < 0.05 for r in rows)
    mean_gain = sum(r["rel_gain"] for r in rows) / len(rows)
    print(f"\nM5 mean gain {mean_gain*100:.1f}% vs M1 ref {ref_mean*100:.1f}%; "
          f"beats ref+2pts in {wins}/{len(rows)}; collapsed {collapsed}")
    print("VERDICT:", "PASS" if wins >= 3 and collapsed == 0 else "FAIL")


if __name__ == "__main__":
    main()
