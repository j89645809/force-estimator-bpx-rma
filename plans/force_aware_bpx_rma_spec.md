# Force-Aware BPX RMA Specification

## Goal
Integrate the existing 25-frame force estimator with the BPX-Pro RMA locomotion policy so the policy can use estimated external force while preserving the original no-force walking behavior.

## Fixed contracts

- Force estimator input: `[batch, 25, 64]`, 20 ms sample period.
- Force estimator output: world-frame external force `[Fx,Fy,Fz]` in N plus force confidence in `[0,1]`.
- BPX policy base observation: 45 dimensions.
- New force-aware policy observation: 49 dimensions = 45 base + 3 body-frame normalized force + 1 force confidence.
- New RMA adaptation history: `[batch, 25, 49]`, flattened to 1225 dimensions only at the ONNX boundary if required by the runtime.
- Force transform: `force_body = quat_apply_inverse(root_quat_w, force_world)`.
- Force normalization: `clip(force_body / 45.0, -1, 1)`.
- Policy rate: 50 Hz. PD rate: 500 Hz.
- Action output remains 12 dimensions and keeps the BPX joint order and target mapping.
- The existing 45/1125 BPX ONNX artifact is never modified in place.
- The default deployed force estimator remains unchanged until all gates pass.

## Training design

- Stage A: train a privileged teacher with true force plus a documented force corruption model.
- Stage B: freeze the teacher base policy and train the RMA adaptation module from estimated-force histories.
- Preserve no-force and low-force anchor samples in every training and validation split.
- Keep a fresh blind seed set for final evaluation; do not select checkpoints using blind data.

## Acceptance gates

1. Shape, ordering, frame, unit, timestamp and reset-cache contract tests pass.
2. Torch and ONNX action parity passes with a documented tolerance.
3. No-force regression remains within the frozen BPX baseline tolerance.
4. Force-aware closed-loop tests cover 0–45 N, high-Fz, Full3D, 2–5 N direction tail, Ramp, Release and Reverse.
5. Termination risk is not worse than the baseline under paired seeds; inconclusive results are not treated as equivalence.
6. Formal application acceptance remains undetermined until application error and recovery budgets are supplied.
