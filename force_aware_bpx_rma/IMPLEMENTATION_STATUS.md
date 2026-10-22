# Force-aware BPX RMA implementation status

2026-10-22

Implemented and tested locally:

- Contract loader for 64-frame estimator, 49-dim policy, 25x49 history and rates.
- World-to-body force adapter with 45 N clipping, normalization and confidence fallback.
- Oldest-first 25-frame cache with reset fill and timestamp skew validation.
- 49-dim BPX observation builder.
- Seeded force corruption model for teacher training.

Verification: `python3 -m pytest -q tests` -> 10 passed.

Not started in this workspace:

- IsaacLab/rsl-rl teacher PPO and student RMA training.
- BPX ONNX export and parity.
- MuJoCo paired blind acceptance.

Reason: the current workspace has no Git checkout or IsaacLab/rsl-rl runtime; those stages require the `bpx_traction` checkout and its documented `bpx_rma` environment. The original BPX model and current force estimator remain unchanged.
