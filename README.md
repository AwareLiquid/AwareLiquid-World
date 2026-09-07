# AwareLiquid World

**Industrial world-model research line** — JEPA-style latent state prediction on
the O-series liquid core.

> Same next-generation route as LeCun's JEPA family: abandon raw-signal
> reconstruction and autoregressive generation; predict and evolve **abstract
> latent states** instead. JEPA validates the route on vision/robotics; this
> repo pushes it down the industrial lane — **continuous time, O(1) memory,
> edge-native** — across audio, sensor, event-stream and physics modalities.

## What this repo is

An experiment layer **on top of** [AwareLiquid/M1](https://github.com/AwareLiquid/M1)
(vendored read-only as `m1/`). M1 owns the core (`mt_lnn`); this repo owns the
world-model objectives and training loop only. **No M1 files are modified.**

```
AwareLiquid-World/
├── m1/                    # git submodule -> AwareLiquid/M1 (read-only reference)
├── world/                 # world-model experiment layer
│   ├── objectives.py      # JEPA-style objectives (masked latent prediction,
│   │                      #   SIGReg, latent-variable head, action conditioning)
│   ├── data.py            # modality frontends reuse: log-mel, sensor streams, events
│   └── train.py           # training loop skeleton
├── PLAN.md                # the roadmap: what exists, what to build, GPU needs
└── tests/                 # smoke tests (no training)
```

## What already exists in M1 (reused, not rewritten)

| Module | Role in the world-model line |
|---|---|
| `mt_lnn/world_model.py` — `PredictiveStateHead` | **V-JEPA-style latent predictor** (BYOL/V-JEPA EMA target, collapse-free across 3 seeds). Already the evaluation head. |
| `mt_lnn/imagination.py` — `LatentImagination` | multi-step latent rollout, 0 params |
| `mt_lnn/sensory_frontend.py` | raw sensor stream → backbone tokens (the modality path) |
| `mt_lnn/physics_ops.py` | Hamiltonian + symplectic rollout (the physics modality) |
| `mt_lnn/salience_events.py` / `pipeline.py` | salience ignition + dual-speed loop (the "awareness" wiring) |
| O1-Sound (`AwareLiquid/O1-Sound`) | audio frontend reference (log-mel, streaming C kernel) |

## What this repo adds (the JEPA borrowings)

1. **Masked latent-state prediction** — mask 70–90% of a temporal segment and
   predict the *latent* state of the masked region (never reconstruct the raw
   signal). Objective in `world/objectives.py`.
2. **SIGReg (isotropic Gaussian regularisation)** — constrain the latent state
   distribution toward a unit Gaussian; replaces teacher-student/EMA plumbing
   for collapse prevention, zero inference cost.
3. **Explicit latent variables** — a latent-factor dimension in the state that
   produces multi-path future rollouts (per-scenario evolution, not one number).
4. **Action-conditioned latent prediction** — drive the state transition with an
   action vector for end-to-end trajectory/world-model planning.

## Honest status

**Research scaffolding, nothing measured yet.** The modules exist in M1 and are
tested in their own contexts; the world-model objectives here are untrained.
Milestones and success criteria in [PLAN.md](PLAN.md). No GPU is currently
attached — the first training run needs an A100-class machine.

## Setup

```bash
git clone --recurse-submodules https://github.com/AwareLiquid/AwareLiquid-World
cd AwareLiquid-World
pip install -r m1/requirements.txt -r requirements.txt   # after they exist
```

## Related

- [AwareLiquid/M1](https://github.com/AwareLiquid/M1) — the MT-LNN core
- [AwareLiquid/O1-Sound](https://github.com/AwareLiquid/O1-Sound) — audio modality reference
- [awareliquid.ai](https://awareliquid.ai) — benchmarks and retractions
