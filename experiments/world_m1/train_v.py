"""World V-module prototype: hypothesis scoring V(h) on multi-path candidates.

Einstein-test cluster requires a global value-modulation unit V whose ablation
collapses convergence ("关 V → 假说爆炸"). This is the v0 analytic prototype
on the World line: score the P candidates from the M3 multi-path generator
with

    V(h) = w1*C(h) + w2*S(h) + w3*F(h) - w4*K(h)

    C  consistency   — boundary continuity: the path must connect to the
                       observed context without teleporting
    S  simplicity    — smoothness: penalise wiggly (overfit) paths
    F  falsifiability— sharpness: deviation from the ensemble mean (a bold,
                       specific prediction is testable; a hedge is not)
    K  complexity    — oscillation count of the latent path

Selection test: argmax-V path vs random pick vs oracle best, measured against
the TRUE future. Ablation "V off" = random selection (hypothesis explosion
analog).

Pilot history (300-step CPU smoke, 3 seeds, documented BEFORE the full run):
- oscillation-count K was replaced by dimensional redundancy (PR): in the
  spring domain the hidden k sets the oscillation rate, so penalising
  oscillations systematically favoured the wrong (low-k) candidates;
- velocity-continuity was added to C — did NOT help (same-context candidates
  share boundary conditions), C stays weak; S (smoothness) is the load-bearing
  term at 300 steps (corr +0.41~+0.53);
- F (deviation-from-mean sharpness) was ANTI-correlated with correctness
  (corr -0.41) — weight set to 0 with this note;
- the initial "sel_ratio < 0.5" bar was UNCALIBRATED for a ~0.4-correlated
  selector (pilot achievable: 0.79-0.83).

FULL-BUDGET result (8000 steps) — the smoke signal did NOT survive:
- v0 (boundary C + smoothness S): all components collapse to |corr| < 0.1;
  V-selection ~ random (seed 0: V-sel 0.078 vs random 0.075, corr 0.052).
- v1 (curvature-match Cm, added after seeing v0 fail): also fails
  (seed 0 local: Cm corr -0.10, S +0.01, V corr -0.050).
- Conclusion: analytic path-intrinsic proxies do not identify the best latent
  sample once the predictor is trained; the 300-step signal was undertraining
  diversity. V works on STRUCTURALLY heterogeneous hypothesis spaces (see the
  Kepler prototype: survivor-conic fraction 1.0 vs 0.125 with V on/off) but
  not on same-structure latent samples. Fix direction: a LEARNED/adaptive V.
  This is an honest DEAD for the analytic V prototype at full budget.

Run:
    python -m experiments.world_m1.train_v --steps 8000 --seeds 5 --variant v1
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .train_m3 import MultiPathWorldModel, spring_randk_trajectories, wta_loss


# ------------------------------------------------------------------ #
# V(h) components (each returns (B, P), higher = better)
# ------------------------------------------------------------------ #

def _participation_ratio(path: torch.Tensor) -> torch.Tensor:
    """Dimensional complexity: PR of the path's covariance spectrum, (B, P).

    Physical 2-D dynamics lie on a low-dimensional latent manifold; noisy /
    redundant paths spread across many dims. NOTE: an oscillation-count K was
    REJECTED before the full run — in the spring domain the hidden k sets the
    oscillation rate, so penalising oscillations systematically favours the
    WRONG (low-k) candidates. Dimensional redundancy is k-neutral.
    """
    x = path - path.mean(dim=2, keepdim=True)                # (B,P,T,D)
    cov = torch.einsum("bptd,bpte->bpde", x, x) / x.shape[2]  # (B,P,D,D)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)             # (B,P,D)
    s = ev.sum(dim=-1)
    s2 = (ev ** 2).sum(dim=-1)
    return s ** 2 / (s2 + 1e-9)                              # (B,P) higher=complex


def v_components(paths: torch.Tensor, h_last: torch.Tensor,
                 h_prev: torch.Tensor | None = None,
                 ctx_z: torch.Tensor | None = None) -> dict:
    """paths (B,P,T,D); h_last/h_prev (B,D); ctx_z (B,Tc,D) context latents.
    Returns raw C/Cm/S/F/K each (B,P)."""
    b, p, t, d = paths.shape
    # C: boundary continuity (position AND velocity)
    c_pos = -((paths[:, :, 0] - h_last.unsqueeze(1)) ** 2).mean(-1)      # (B,P)
    if h_prev is not None:
        v_ctx = h_last - h_prev                                          # (B,D)
        v_path = paths[:, :, 1] - paths[:, :, 0]                         # (B,P,D)
        c_vel = -((v_path - v_ctx.unsqueeze(1)) ** 2).mean(-1)
        c = c_pos + c_vel
    else:
        c = c_pos
    # S: smoothness — mean squared second difference
    dd = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
    s = -(dd ** 2).mean(dim=(2, 3))
    # F: sharpness — deviation from ensemble mean
    f = ((paths - paths.mean(dim=1, keepdim=True)) ** 2).mean(dim=(2, 3))
    # K: complexity — dimensional redundancy (participation ratio)
    k = _participation_ratio(paths)
    # Cm (v1): CURVATURE MATCH — the context reveals the regime's curvature
    # scale; the correct continuation has the SAME scale (the v0 boundary-C
    # fails because all paths share the condition point; S alone carries no
    # regime signal). curv(x) = mean |second difference| over (t, d).
    if ctx_z is not None:
        ctx_dd = ctx_z[:, 2:] - 2 * ctx_z[:, 1:-1] + ctx_z[:, :-2]       # (B,Tc-2,D)
        curv_ctx = ctx_dd.abs().mean(dim=(1, 2))                          # (B,)
        curv_path = dd.abs().mean(dim=(2, 3))                             # (B,P)
        cm = -(curv_path - curv_ctx.unsqueeze(1)).abs()
    else:
        cm = torch.zeros_like(s)
    return {"C": c, "S": s, "F": f, "K": k, "Cm": cm}


# v0: 边界一致性 + 平滑 + 复杂度（全量训练下失效，见 pilot 记录）
# v1: 曲率匹配 Cm 主导（上下文揭示制度的曲率尺度，正确续轨同尺度）
VARIANTS = {
    "v0": {"C": 0.5, "Cm": 0.0, "S": 1.0, "F": 0.0, "K": 0.5},
    "v1": {"C": 0.0, "Cm": 1.0, "S": 0.5, "F": 0.0, "K": 0.5},
}


def v_score(comp: dict, weights: dict | None = None) -> torch.Tensor:
    """Normalised weighted V: (B,P). Normalise each term across P per sample.

    F (deviation-from-mean "sharpness") stays weight 0: pilot-measured
    ANTI-correlation (-0.41~-0.52) — wrong proxy for this domain.
    """
    if weights is None:
        weights = VARIANTS["v1"]
    out = None
    for name in ("C", "Cm", "S", "F", "K"):
        x = comp[name]
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + 1e-6
        sign = -1 if name == "K" else +1
        term = sign * weights.get(name, 0.0) * (x - mu) / sd
        out = term if out is None else out + term
    return out


# ------------------------------------------------------------------ #
# Train (M3 objective) + evaluate the V selector
# ------------------------------------------------------------------ #

def train_seed(steps: int, seed: int, k_lo: float, k_hi: float,
               batch: int = 64, t_ctx: int = 8, t_fut: int = 32,
               n_paths: int = 8, lr: float = 3e-3,
               weights: dict | None = None) -> dict:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiPathWorldModel(2, 64, 8, n_paths, t_fut).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for step in range(steps):
        traj = spring_randk_trajectories(batch, t_ctx + t_fut, k_lo, k_hi,
                                         seed=seed * 10000 + step).to(dev)
        ctx, fut = traj[:, :t_ctx], traj[:, t_ctx:]
        with torch.no_grad():
            z_fut = model.encode(fut)
        pred = model.predict_paths(ctx)
        loss, _ = wta_loss(pred, z_fut)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 2000 == 0:
            print(f"  seed {seed} step {step+1} loss {loss.item():.4f}", flush=True)

    # ---- V evaluation ----
    model.eval()
    n_eval = 512
    traj = spring_randk_trajectories(n_eval, t_ctx + t_fut, k_lo, k_hi,
                                     seed=999).to(dev)
    ctx, fut = traj[:, :t_ctx], traj[:, t_ctx:]
    with torch.no_grad():
        z_ctx = model.encode(ctx)
        z_fut = model.encode(fut)
        h_last = z_ctx[:, -1]
        paths = model.predict_paths(ctx)                     # (B,P,T,D)
        per_err = ((paths - z_fut.unsqueeze(1)) ** 2).mean(dim=(2, 3)).sqrt()  # (B,P)
        comp = v_components(paths, h_last, z_ctx[:, -2], ctx_z=z_ctx)
        v = v_score(comp, weights=weights)                    # (B,P)

    idx_v = v.argmax(dim=1)
    err_v = per_err.gather(1, idx_v[:, None]).squeeze(1).mean().item()
    err_rand = per_err.mean(dim=1).mean().item()             # expected random pick
    err_best = per_err.min(dim=1).values.mean().item()       # oracle

    def _rank_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        """Per-sample Spearman (Pearson on ranks), x/y (B,P) -> mean corr."""
        rx = x.argsort(dim=1).argsort(dim=1).float()
        ry = y.argsort(dim=1).argsort(dim=1).float()
        rx = rx - rx.mean(dim=1, keepdim=True)
        ry = ry - ry.mean(dim=1, keepdim=True)
        return ((rx * ry).sum(1) / (rx.norm(dim=1) * ry.norm(dim=1) + 1e-9)
                ).mean().item()

    corr = _rank_corr(v, -per_err)
    per_comp_corr = {name: _rank_corr(comp[name], -per_err)
                     for name in ("C", "Cm", "S", "F", "K")}

    # ablation "V off": random pick, 16 draws for a stable estimate
    g = torch.Generator().manual_seed(seed)
    draws = torch.stack([per_err[torch.arange(n_eval),
                                 torch.randint(0, n_paths, (n_eval,), generator=g)]
                         for _ in range(16)])
    err_voff = draws.mean().item()

    return {"seed": seed, "err_v": err_v, "err_rand": err_rand,
            "err_best": err_best, "err_voff": err_voff, "corr": corr,
            "comp_corr": per_comp_corr,
            "sel_ratio": err_v / max(err_rand, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--k-lo", type=float, default=1.0)
    ap.add_argument("--k-hi", type=float, default=4.0)
    ap.add_argument("--n-paths", type=int, default=8)
    ap.add_argument("--variant", choices=("v0", "v1"), default="v1")
    args = ap.parse_args()
    weights = VARIANTS[args.variant]

    print(f"=== World V-module [{args.variant}]: V(h) scoring on "
          f"{args.n_paths}-path candidates "
          f"(spring hidden k~U({args.k_lo},{args.k_hi})) ===")
    rows = []
    for seed in range(args.seeds):
        r = train_seed(args.steps, seed, args.k_lo, args.k_hi,
                       n_paths=args.n_paths, weights=weights)
        rows.append(r)
        print(f"  seed {seed}: V-sel {r['err_v']:.4f} | random {r['err_rand']:.4f} "
              f"| oracle {r['err_best']:.4f} | V-off {r['err_voff']:.4f} "
              f"| sel_ratio {r['sel_ratio']:.2f} | corr {r['corr']:.3f}")
        print("    comp corr:", " ".join(
            f"{k}={v:+.2f}" for k, v in r["comp_corr"].items()))

    wins = sum(r["sel_ratio"] < 0.9 for r in rows)
    corrs = sum(r["corr"] > 0.3 for r in rows)
    print(f"\nV-selection < 0.9x random in {wins}/{len(rows)} seeds; "
          f"corr(V,-err) > 0.3 in {corrs}/{len(rows)}")
    print("VERDICT:", "PASS" if wins >= 3 and corrs >= 3 else "FAIL")


if __name__ == "__main__":
    main()
