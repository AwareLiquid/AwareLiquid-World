"""Consolidation v2: replay consolidation on EPISODE-structured dynamics.

v1 lesson (M2 consolidation_loop_probe, 2026-09): toy point-tracking on 2-D
oscillators was seed-noise dominated — replay effect flipped across seeds.
Per data-form.md the episodic-memory / consolidation components need
EPISODE-structured data and an offline replay protocol, not thin point
tracking. This redo:

- Tasks = dynamical REGIMES (spring k = 1 / 3 / 6), each episode a full
  long trajectory (T=64) — the model must capture regime dynamics;
- Metric = open-loop ROLLOUT error (structured, like M4) on task A after
  learning task B — forgetting measured on structure, not position;
- Arms = no_replay / real_replay (stored episodes) / genreplay (replay set
  generated OFFLINE from the frozen phase-1 model + stored seed contexts);
- 3 seeds minimum to escape the v1 noise.

v2 smoke (800/400 steps, 2 seeds): CLEAN signal — forgetting real
(err_A 0.005 -> 0.34-0.39), real_replay 0.014-0.026x no_replay, OFFLINE
genreplay 0.025-0.039x. The v1 "replay is unstable" verdict was a protocol
artifact: generating replay ONLINE while the model drifts contaminates it
(measured: 0.098x, barely better than no_replay); the offline set fixes it.

Success (pre-registered): real_replay AND genreplay A-error < 0.7x no_replay
in >=2/3 seeds; genreplay-vs-real gap reported honestly.

Run:
    python -m experiments.world_m1.train_consol --steps1 3000 --steps2 1500
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import LiquidEncoder


# ------------------------------------------------------------------ #
# Episodes: spring trajectories under a fixed regime k
# ------------------------------------------------------------------ #

def gen_episodes(k: float, n: int, T: int, dt: float = 0.05, c: float = 0.05,
                 seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g) * 4 - 2
    v = torch.rand(n, generator=g) * 2 - 1
    traj = torch.zeros(n, T, 2)
    for t in range(T):
        traj[:, t, 0] = x
        traj[:, t, 1] = v
        acc = -k * x - c * v
        v = v + acc * dt
        x = x + v * dt
    return traj


class TransitionModel(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.enc = LiquidEncoder(2, d, n_layers=2)
        self.trans = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, d))
        self.dec = nn.Linear(d, 2)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.trans(z)

    def rollout(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        z = z0
        out = []
        for _ in range(steps):
            z = self.step(z)
            out.append(self.dec(z))
        return torch.stack(out, dim=1)


@torch.no_grad()
def generate_replay_set(model: TransitionModel, starts: torch.Tensor,
                        t_ctx: int = 16, t_fut: int = 32) -> torch.Tensor:
    """OFFLINE generative replay: with the phase-1 model FROZEN, roll out a
    synthetic A-episode for every stored seed context. The frozen set is then
    mixed into phase-2 training (data-form.md: 离线回放协议 — generating
    online while the model drifts contaminates the replay, see v1/v2 note)."""
    model.eval()
    zz = model.enc(starts)[:, -1]
    gen = [starts]
    for _ in range(t_fut):
        zz = model.step(zz)
        gen.append(model.dec(zz).unsqueeze(1))
    return torch.cat(gen, dim=1)


def train_model(model: TransitionModel, opt, batches: torch.Tensor, steps: int,
                t_ctx: int = 16, t_fut: int = 32, tag: str = "",
                replay_data: torch.Tensor | None = None,
                log_every: int = 500) -> None:
    """Train on stream B with an optional offline replay set mixed in."""
    model.train()
    n = batches.shape[0]
    for step in range(steps):
        idx = torch.randint(0, n, (32,))
        seg = batches[idx]
        loss, z = _seg_loss(model, seg, t_ctx, t_fut)
        if replay_data is not None:
            ia = torch.randint(0, replay_data.shape[0], (32,))
            loss = loss + _seg_loss(model, replay_data[ia], t_ctx, t_fut)[0]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % log_every == 0:
            print(f"    [{tag}] step {step+1} loss {loss.item():.5f}", flush=True)


def _seg_loss(model: TransitionModel, seg: torch.Tensor, t_ctx: int,
              t_fut: int) -> tuple[torch.Tensor, torch.Tensor]:
    z = model.enc(seg)
    zz = z[:, t_ctx - 1]
    preds, z_tgts, s_tgts = [], [], []
    for t in range(t_fut):
        zz = model.step(zz)
        preds.append(zz)
        z_tgts.append(z[:, t_ctx + t].detach())
        s_tgts.append(seg[:, t_ctx + t])
    z_pred = torch.stack(preds, 1)
    loss = F.mse_loss(z_pred, torch.stack(z_tgts, 1)) \
        + F.mse_loss(model.dec(z_pred), torch.stack(s_tgts, 1)) \
        + F.mse_loss(model.dec(z), seg)
    return loss, z


@torch.no_grad()
def eval_task(model: TransitionModel, data: torch.Tensor, t_ctx: int = 16,
              t_fut: int = 32) -> float:
    model.eval()
    seg = data[:128]
    z = model.enc(seg)
    pred = model.rollout(z[:, t_ctx - 1], t_fut)
    target = seg[:, t_ctx:t_ctx + pred.shape[1]]
    return F.mse_loss(pred, target).item()


def run_arm(base: TransitionModel, stream_b: torch.Tensor,
            data_a: torch.Tensor, arm: str, steps2: int, seed: int,
            t_ctx: int = 16, t_fut: int = 32, n_stored: int = 64) -> dict:
    model = copy.deepcopy(base)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    if arm == "real_replay":
        replay_data = data_a                              # full stored episodes
    elif arm == "genreplay":
        # only the seed contexts are stored; the replay SET is generated
        # OFFLINE from the frozen phase-1 model before phase 2 starts
        starts = data_a[:n_stored, :t_ctx]
        replay_data = generate_replay_set(base, starts, t_ctx, t_fut)
    else:
        replay_data = None
    train_model(model, opt, stream_b, steps2, t_ctx, t_fut, tag=arm,
                replay_data=replay_data)
    return {"arm": arm,
            "err_a": eval_task(model, data_a, t_ctx, t_fut),
            "err_b": eval_task(model, stream_b, t_ctx, t_fut)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps1", type=int, default=3000)
    ap.add_argument("--steps2", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--T", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_ctx, t_fut = 16, 32
    print("=== Consolidation v2: episode-structured regimes (k=1 -> k=3), "
          "open-loop rollout metric ===")
    results = []
    for seed in range(args.seeds):
        # Task A: regime k=1; Task B: regime k=3 (disjoint episodes)
        data_a = gen_episodes(1.0, 256, args.T, seed=seed * 100 + 1).to(dev)
        stream_b = gen_episodes(3.0, 256, args.T, seed=seed * 100 + 2).to(dev)

        # Phase 1: learn A (shared starting point for every arm)
        torch.manual_seed(seed)
        base = TransitionModel(64).to(dev)
        opt = torch.optim.AdamW(base.parameters(), lr=3e-3, weight_decay=1e-4)
        print(f"  seed {seed}: Phase 1 (learn A)")
        train_model(base, opt, data_a, args.steps1, t_ctx, t_fut, tag="A")
        err_a0 = eval_task(base, data_a, t_ctx, t_fut)
        print(f"    after A: err_A {err_a0:.5f}")

        # Phase 2: three arms continue from the SAME phase-1 weights
        for arm in ("no_replay", "real_replay", "genreplay"):
            torch.manual_seed(seed)
            r = run_arm(base, stream_b, data_a, arm, args.steps2, seed,
                        t_ctx, t_fut)
            r["seed"] = seed
            r["err_a0"] = err_a0
            results.append(r)
            print(f"    [{arm}] err_A {r['err_a']:.5f} (phase1 {err_a0:.5f}) "
                  f"err_B {r['err_b']:.5f}")

    # verdict per seed
    wins_real = wins_gen = 0
    for seed in range(args.seeds):
        rows = {r["arm"]: r for r in results if r["seed"] == seed}
        no = rows["no_replay"]["err_a"]
        if rows["real_replay"]["err_a"] < 0.7 * no:
            wins_real += 1
        if rows["genreplay"]["err_a"] < 0.7 * no:
            wins_gen += 1
    print(f"\nreal_replay < 0.7x no_replay in {wins_real}/{args.seeds}; "
          f"genreplay in {wins_gen}/{args.seeds}")
    print("VERDICT:", "PASS" if wins_real >= 2 and wins_gen >= 2 else "FAIL")


if __name__ == "__main__":
    main()
