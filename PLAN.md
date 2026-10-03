# PLAN — Industrial World-Model Line (JEPA route on the liquid core)

Status: **scaffolding**. Nothing below is trained or claimed; this is the
decision-complete roadmap so the first GPU rental can start immediately.

## 1. Positioning (one sentence)

We and LeCun's JEPA family share one route — non-autoregressive, latent-space
state prediction — with JEPA proving it on vision/robotics and this line
pushing it into industrial continuous-time sensing and edge control.

## 2. Reuse map (what we do NOT build)

All in `m1/mt_lnn`, all already tested in their own right:

- `world_model.py::PredictiveStateHead` — V-JEPA-style latent predictor, EMA
  target, collapse-free (3 seeds). → evaluation head for every milestone.
- `sensory_frontend.py` — jittered multi-channel sensor stream → tokens.
- `physics_ops.py` — Hamiltonian + symplectic rollout → physics modality target.
- `salience_events.py` + `pipeline.py` — dual-speed ignition loop.
- O1-Sound — audio frontend (log-mel) + the streaming C-kernel discipline.

## 3. Milestones (each = one hypothesis, one run, one honest verdict)

### M1 — masked latent prediction (core JEPA objective)
- **Hypothesis**: predicting the latent state of a 70–90% masked temporal
  segment beats next-step prediction for robustness to dropout/non-uniform
  sampling.
- **Setup**: O-series liquid core (10–100M config), modality = battery/sensor
  stream (NASA PCoE data already in M1), mask 80% spans.
- **Metric**: downstream probe (SOH regression / missing-sample robustness),
  compared against the same core trained on next-step prediction.
- **Success**: probe error drops vs next-step baseline on ≥3 seeds; no collapse.
- **GPU**: 1× A100, ~4–6h.

### M2 — SIGReg
- **Hypothesis**: isotropic Gaussian regularisation on the latent state
  prevents collapse with zero inference cost, replacing the EMA/SimSiam
  machinery.
- **Metric**: collapse rate across seeds + probe parity with M1.
- **Success**: no collapse over 5 seeds; probe within noise of M1.

### M3 — explicit latent variables (multi-path rollout)
- **Hypothesis**: a latent-factor dimension yields multiple future evolution
  paths, useful for decision support.
- **Metric**: rollout diversity vs ground-truth coverage (physics modality —
  spring/orbit datasets already in M1).
- **Success**: conditioned rollouts cover the observed futures; honest failure
  mode documented.

### M4 — action-conditioned latent prediction
- **Hypothesis**: action-vector-conditioned state transition gives end-to-end
  trajectory prediction on the liquid core.
- **Metric**: trajectory error vs the physics_ops symplectic rollout baseline.
- **Success**: within 1.5× of the analytic baseline on the spring/orbit suite.

## 3.5 跨窗口状态板（2026-10-03，EverestAn 多窗口在同一个仓库并行）

窗口 A（本窗口）按序补 M1-M4；窗口 B 已冲 M5/M7/M8。为避免互相踩，
当前里程碑归属与交叉提示如下：

| 里程碑 | 状态 | 归属/提示 |
|---|---|---|
| M1 掩码隐空间 | ✅ PASS 6/6 | 窗口 A（results 见 experiments/world_m1/RESULTS.md） |
| M2 SIGReg | ⚠️ 5/5 无塌缩，seed0 增益 19.5% 离群 | 窗口 A。**给窗口 B 的提示：M8 d64 42.1% 的"sigreg 拖低"与 M2 的 seed0 离群同源——SIGReg 权重未扫（固定 0.01）。M8 重试前先做权重扫 0.001/0.01/0.1 × ≥3 seeds，d64 档很可能恢复**。**已扫完（10-04）：离群是 0.01 特有（seed0 19.5% 确定性复现）；0.001 与 0.1 均健康（72-91%/64-88%）——M8 重试请用 sigreg_weight=0.001，勿用 0.01** |
| M3 隐变量多路径 | ✅ PASS 覆盖率 0.969 | 窗口 A |
| M4 动作条件化 | 🔄 服务器训练中（spring+force）| 窗口 A。预赛：cond 0.00004 vs blind 0.00087 vs 持久化 0.10111（力占主导设定，动作信息被吃到）。与窗口 B 的 M5（λ 多路径检验）不重叠 |
| M5 λ 多路径 | ❌ DEAD | 窗口 B |
| M7 潜空间 MPC | ✅ PASS | 窗口 B |
| M8 极简世界模型 | ❌ 见 SIGReg 提示 | 窗口 B |

**汇合纪律**：里程碑结果各自写进 `experiments/world_m1/RESULTS.md`；状态
变更改本板；不互删对方的登记。

## 4. Data strategy

- Start with what M1 already ships: NASA battery (irregular streams), physics
  operator suites, O1-Sound audio.
- No vision-language modality — that lane is taken; ours is industrial sensing.

## 5. Honest non-goals

- Not a general vision world model (JEPA's lane).
- Not an LLM retrofit (the M-series LM line stays in M1).
- No M1 code changes — this repo is strictly an experiment layer.

## 7. JEPA-family field notes (what the route teaches us)

Tracked against the family's papers (2023–2026). Each item ends in a
concrete design consequence for this repo.

| Source | Finding | Consequence for us |
|---|---|---|
| I-JEPA (2023) | static representations, no action | baseline only |
| V-JEPA (2023) | temporal latent prediction works on video | our M1 core is the same move on continuous streams |
| ACT-JEPA (2025, 2501.14622) | **joint IL + SSL beats two-stage**: 41% vs 27% Push-T; two-stage fails 0% on ManiSkill ("representation misalignment") | **M4 must train action + latent prediction jointly from day one**, never freeze-then-fine-tune |
| ACT-JEPA | EMA target encoder prevents collapse; action chunking beats single-step | already have EMA (PredictiveStateHead); chunking enters the robot milestone |
| V-JEPA 2 (2025-06, 2506.09985) | 1.2B params, 1M+ h video; SSv2 77.3, EK100 39.7 R@5; scaling levers: data +1.0, model +1.5, iters +0.8, **spatiotemporal resolution +4.0 (biggest)** | our "resolution" lever = sensor sampling rate + window duration — test it explicitly |
| V-JEPA 2.1 | Dense Predictive Loss (all tokens), Deep Self-Supervision (**loss at multiple layers**), model+data scaling | **apply the latent loss at multiple depths/timescales of the liquid core** — our τ ladder is a native multi-level substrate (see M5) |
| V-JEPA 2-AC | action-conditioned world model from 62h robot data; zero-shot Franka pick-and-place; MPC+CEM over latent rollouts with image goals; 16s/action vs Cosmos 4min | our M4 + M7: latent-space MPC on physics_ops rollouts |
| V-JEPA 2 future directions (Meta, explicit) | **hierarchical JEPA (multi timescale)** + **multimodal JEPA (vision+audio+touch)** | the liquid τ ladder already IS hierarchical; audio+sensor+physics is our multimodal — we are pre-positioned on their stated roadmap |
| C-JEPA (2026) | object-centric, 1% of tokens, 8× faster planning | compression-of-context is a real lever; consider sparse state |
| LeWorldModel (2026) | minimal world model: **2 loss terms, 15M params, 48× faster planning** | our target scale is exactly this — a 15M-class O-series world model is the headline deliverable (see M8) |
| ThinkJEPA (2026) | VLM-guided long-horizon semantic reasoning | long-horizon system evolution (months-years) is the industrial analog — see M9 |
| SIGReg (2025) | Gaussian regularisation replaces collapse plumbing | M2 |
| Evaluation discipline (all JEPA) | frozen encoder + lightweight attentive probes | every milestone's metric = frozen backbone + probe, not fine-tuned |

## 8. Milestones M5–M9 (added from the field notes)

### M5 — deep multi-level supervision (V-JEPA 2.1)
- **Hypothesis**: applying the latent prediction loss at multiple depths /
  timescales of the liquid core beats last-layer-only supervision.
- **Setup**: JEPA loss at the output of each cell (the τ ladder gives
  native multi-timescale states — Meta's stated "hierarchical JEPA"
  direction, which our architecture hosts natively).
- **Success**: probe improvement over single-level M1 on ≥3 seeds.

### M6 — joint IL + world model (ACT-JEPA lesson)
- **Hypothesis**: joint action-chunking + latent prediction training beats
  the two-stage freeze-then-fine-tune (the ManiSkill 0% failure mode).
- **Success**: policy + world-model probes both ≥ two-stage baseline.

### M7 — latent-space MPC planning (V-JEPA 2-AC)
- **Hypothesis**: CEM/MPPI over imagined latent rollouts, with goal
  embeddings as the energy, gives closed-loop planning at edge speed.
- **Setup**: reuse `m1/mt_lnn/imagination.py` rollouts + `physics_ops`
  goal states; planning loop in `world/`.
- **Success**: task success on the physics suite ≥ greedy baseline; wall
  time per action comparable to V-JEPA 2-AC's 16s on much smaller hardware.

### M8 — minimal world model (LeWorldModel target)
- **Target**: a 2-loss, ~15M-param O-series world model that passes the
  same probes — the "smallest industrial world model that works" claim.
- **Success**: within noise of the full config's probes at ≤15M params.

### M9 — long-horizon system evolution (ThinkJEPA analog)
- **Hypothesis**: with explicit latent variables (M3), the model evolves
  system state over months-years horizons for capacity/maintenance
  planning (the industrial "推演引擎" product story).
- **Metric**: calibrated horizon vs degradation vs a naive baseline.
- **Success**: honest multi-path rollouts with stated uncertainty.

## 9. GPU rental checklist (when the next A100 lands)

1. `git clone --recurse-submodules` this repo.
2. `pip install -r m1/requirements.txt`.
3. Run M1 milestone script → record probe numbers in `results/`.
4. Each milestone writes its own honest status block (what passed, what
   failed, seeds, raw data paths) — same discipline as M1's RESULTS.md.
