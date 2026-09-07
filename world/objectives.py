"""JEPA-style objectives for the industrial world-model line.

Each objective is pure torch math over latent state sequences; none of them
rebuild raw signals. All are UNTRAINED research scaffolding — the milestones
in PLAN.md define the first experiments that exercise them.

Design notes:
- ``masked_latent_prediction_loss`` — mask spans of the latent sequence and
  predict the masked states from the visible context (the core JEPA move:
  the target is latent state, never raw input).
- ``sigreg`` — isotropic Gaussian regularisation on latent states (SIGReg):
  pulls each state dimension toward N(0,1). Replaces EMA/SimSiam collapse
  plumbing at zero inference cost.
- ``LatentVariableHead`` — an explicit latent-factor dimension; conditioning
  on different latent values produces different future paths.
- ``action_conditioned_step`` — inject an action vector into the transition
  for trajectory/world-model planning.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# M1 — masked latent-state prediction (core JEPA objective)
# ------------------------------------------------------------------ #

def mask_spans(length: int, mask_ratio: float, mean_span: int,
               generator: torch.Generator) -> torch.Tensor:
    """Deterministic-ish span mask: 1 = keep, 0 = mask (predict)."""
    keep = torch.ones(length, dtype=torch.bool)
    n_masked = int(length * mask_ratio)
    masked = 0
    while masked < n_masked:
        span = max(1, int(torch.poisson(
            torch.tensor(float(mean_span)), generator=generator).item()))
        start = int(torch.randint(0, length, (1,), generator=generator).item())
        for i in range(start, min(start + span, length)):
            if keep[i] and masked < n_masked:
                keep[i] = False
                masked += 1
    return keep


def masked_latent_prediction_loss(
        states: torch.Tensor, keep: torch.Tensor,
        predictor: nn.Module) -> torch.Tensor:
    """Predict masked latent states from visible ones.

    states:  (B, T, D) latent sequence from the liquid core
    keep:    (T,) bool mask (1 = visible)
    predictor: module mapping (B, T, D) -> (B, T, D); the masked positions
               of its output are scored against the true states.
    The target states are detached — the target is the frozen latent, the
    gradient flows only through the predictor (the JEPA asymmetry).
    """
    pred = predictor(states)
    masked = ~keep.to(states.device)
    target = states.detach()[:, masked]
    guess = pred[:, masked]
    # Cosine + L2: unit-direction agreement plus scale, both in latent space.
    cos = 1.0 - F.cosine_similarity(guess, target, dim=-1).mean()
    l2 = F.mse_loss(guess, target)
    return cos + l2


# ------------------------------------------------------------------ #
# M2 — SIGReg: isotropic Gaussian regularisation on the latent state
# ------------------------------------------------------------------ #

def sigreg(states: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """KL-ish penalty pulling the latent distribution toward N(0, I).

    mean² + var − log var − 1, averaged over the state dimensions — the
    Gaussian term of a VAE ELBO without a decoder. Zero inference cost:
    it is a training-only regulariser.
    """
    mean = states.mean(dim)
    var = states.var(dim, unbiased=False) + 1e-6
    return (mean.square() + var - var.log() - 1.0).mean()


# ------------------------------------------------------------------ #
# M3 — explicit latent variables: multi-path future rollout
# ------------------------------------------------------------------ #

class LatentVariableHead(nn.Module):
    """Split the recurrent state into a shared part and an explicit latent
    factor; conditioning on different latent values yields different future
    paths.

    state:        (B, D) hidden state
    n_latent:     dimension of the explicit latent factor
    path:         (B, n_paths, n_latent) optional conditioning values
    Returns:      (B, n_paths, D) path-conditioned states, or (B, D) when
                  ``path`` is None (the latent is then sampled from N(0,1)).
    """

    def __init__(self, d_state: int, n_latent: int, n_paths: int = 1):
        super().__init__()
        self.n_latent = n_latent
        self.n_paths = n_paths
        self.shared = nn.Linear(d_state, d_state - n_latent)
        self.latent = nn.Linear(d_state, n_latent)
        self.mix = nn.Linear((d_state - n_latent) + n_latent, d_state)

    def forward(self, state: torch.Tensor,
                path: torch.Tensor | None = None) -> torch.Tensor:
        shared = self.shared(state)                       # (B, D - L)
        if path is None:
            z = torch.randn(state.shape[0], self.n_latent, device=state.device)
            return self.mix(torch.cat([shared, z], dim=-1))
        # path: (B, n_paths, L) -> broadcast over shared
        b, n_paths, _ = path.shape
        shared = shared.unsqueeze(1).expand(b, n_paths, -1)
        return self.mix(torch.cat([shared, path], dim=-1))


# ------------------------------------------------------------------ #
# M4 — action-conditioned latent transition
# ------------------------------------------------------------------ #

def action_conditioned_step(
        cell: nn.Module, x_t: torch.Tensor, h: torch.Tensor,
        action: torch.Tensor, action_proj: nn.Linear) -> torch.Tensor:
    """Drive a liquid-core cell with an action vector.

    The action is projected into the input space and added before the
    recurrent step — the state transition becomes a function of both the
    observation and the commanded action, which is the end-to-end
    trajectory-planning form (no separate controller).
    """
    x_aug = x_t + action_proj(action)
    return cell.step(x_aug, h)


# ------------------------------------------------------------------ #
# Sanity self-test (no training)
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, 64, 32
    states = torch.randn(B, T, D)
    gen = torch.Generator().manual_seed(1)
    keep = mask_spans(T, 0.8, mean_span=8, generator=gen)
    print(f"mask: {int((~keep).sum())}/{T} masked")
    pred = nn.Linear(D, D)
    loss = masked_latent_prediction_loss(states, keep, pred)
    print(f"masked latent loss: {loss.item():.4f}")
    reg = sigreg(states)
    print(f"sigreg: {reg.item():.4f}")
    head = LatentVariableHead(D, 4)
    out = head(states[:, 0])
    print(f"latent head: {tuple(out.shape)}")
    path = torch.randn(B, 3, 4)
    multi = head(states[:, 0], path)
    print(f"multi-path: {tuple(multi.shape)}")
