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

## 4. Data strategy

- Start with what M1 already ships: NASA battery (irregular streams), physics
  operator suites, O1-Sound audio.
- No vision-language modality — that lane is taken; ours is industrial sensing.

## 5. Honest non-goals

- Not a general vision world model (JEPA's lane).
- Not an LLM retrofit (the M-series LM line stays in M1).
- No M1 code changes — this repo is strictly an experiment layer.

## 6. GPU rental checklist (when the next A100 lands)

1. `git clone --recurse-submodules` this repo.
2. `pip install -r m1/requirements.txt`.
3. Run M1 milestone script → record probe numbers in `results/`.
4. Each milestone writes its own honest status block (what passed, what
   failed, seeds, raw data paths) — same discipline as M1's RESULTS.md.
