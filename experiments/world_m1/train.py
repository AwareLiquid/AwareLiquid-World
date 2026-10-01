"""World model M1 milestone: masked latent-state prediction (core JEPA).

Plan (PLAN.md M1): train a liquid encoder + a latent predictor with the
masked-latent-prediction objective; evaluate with a FROZEN encoder +
linear probe against the next-step (persistence) baseline; ≥3 seeds, no
collapse. GPU budget: A100.

Run:
    python experiments/world_m1/train.py --kind spring --steps 8000 --seeds 3
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from world.objectives import masked_latent_prediction_loss, mask_spans

from .data import make_dataset
from .encoder import LiquidEncoder


class LatentPredictor(nn.Module):
    """2-layer bidirectional GRU over latents -> latent reconstruction."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.fwd = nn.GRU(d, hidden, batch_first=True)
        self.bwd = nn.GRU(d, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        f, _ = self.fwd(z)
        b, _ = self.bwd(torch.flip(z, dims=[1]))
        return self.out(torch.cat([f, torch.flip(b, dims=[1])], dim=-1))


def latent_std(z: torch.Tensor) -> float:
    """Collapse signal: per-dim std averaged; 0 = collapsed."""
    return float(z.std(dim=(0, 1)).mean())


def train_seed(kind: str, seed: int, steps: int, T: int = 64,
               batch: int = 64, mask_ratio: float = 0.5,
               mean_span: int = 6) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = LiquidEncoder(2 if kind == "spring" else 4, d=64, n_layers=2).to(dev)
    pred = LatentPredictor(64).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)   # CPU generator (mask_spans uses CPU)

    for step in range(steps):
        traj, _ = make_dataset(kind, batch, T, seed=seed * 10000 + step)
        traj = traj.to(dev)
        z = enc(traj)
        keep = mask_spans(T, mask_ratio, mean_span, g)
        loss = masked_latent_prediction_loss(z, keep, pred)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  seed {seed} step {step+1} loss {loss.item():.4f} "
                  f"latent_std {latent_std(z.detach()):.3f}", flush=True)

    # ---- evaluation: frozen encoder + linear probe -------------------
    enc.eval()
    X, Y = make_dataset(kind, 4096, T + 1, seed=777 + seed)
    X, Y = X.to(dev), Y.to(dev)
    with torch.no_grad():
        Z = enc(X)                            # (B, T, D) latents
        Z1 = enc(Y)[:, -1]                    # latent of the next state
    # persistence baseline: predict next state = current state
    base_mse = float(((Y[:, :-1] - X[:, :-1]) ** 2).mean())

    # lightweight attentive probe (PLAN spec): latent_t -> next raw state
    probe = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, Y.shape[-1]),
    ).to(dev)
    optp = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for _ in range(1500):
        idx = torch.randperm(Z.shape[0])[:1024]
        loss = nn.functional.mse_loss(probe(Z[idx][:, :-1]), Y[idx][:, :-1])
        optp.zero_grad()
        loss.backward()
        optp.step()
    with torch.no_grad():
        probe_mse = float(nn.functional.mse_loss(probe(Z[:, :-1]), Y[:, :-1]))

    return {"seed": seed, "base_mse": base_mse, "probe_mse": probe_mse,
            "rel_gain": (base_mse - probe_mse) / base_mse,
            "latent_std": latent_std(Z.detach())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="spring", choices=("spring", "orbit"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    print(f"=== World M1: masked-latent prediction ({args.kind}) ===")
    rows = []
    for seed in range(args.seeds):
        r = train_seed(args.kind, seed, args.steps)
        rows.append(r)
        print(f"  seed {seed}: probe {r['probe_mse']:.4f} vs baseline "
              f"{r['base_mse']:.4f}  rel_gain {r['rel_gain']*100:.1f}%  "
              f"latent_std {r['latent_std']:.3f}")

    wins = sum(r["rel_gain"] > 0 for r in rows)
    collapsed = sum(r["latent_std"] < 0.05 for r in rows)
    print(f"\nprobe beats baseline in {wins}/{len(rows)} seeds; "
          f"collapsed seeds: {collapsed}")
    print("VERDICT:", "PASS" if wins >= 3 and collapsed == 0
          else ("FAIL" if wins < 3 else "PARTIAL-COLLAPSE"))


if __name__ == "__main__":
    main()
