# Force-Aware BPX RMA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a force-aware BPX RMA locomotion policy that consumes the current 25-frame force estimator without breaking the original BPX observation and action semantics.

**Architecture:** Keep the force estimator as a separate 25×64 TorchScript/ONNX observer. Add an explicit adapter that converts its world-frame Newton output into a body-frame, clipped, normalized force plus confidence, then append those four values to the BPX 45-dimensional policy observation and every frame of the 25-frame RMA history. Train a privileged teacher first, then freeze its base policy and distill the RMA adaptation module from estimated-force histories; export a new 49/1225 ONNX contract alongside the untouched 45/1125 baseline.

**Tech Stack:** Python, PyTorch, TorchScript, ONNX Runtime, IsaacLab/rsl-rl for BPX RMA training, MuJoCo/MJLab for paired closed-loop evaluation, YAML contracts, pytest.

**Spec:** `docs/superpowers/specs/force_aware_bpx_rma_spec.md`

## Global Constraints

- Force estimator input is `[batch,25,64]` at 20 ms.
- New BPX policy input is 49 dimensions; new RMA history is 25×49=1225 dimensions.
- External force is transformed from world frame to body frame before policy input.
- Force is normalized as `clip(force_body / 45.0, -1, 1)`.
- BPX action output remains 12 dimensions with the existing joint order and target mapping.
- Policy runs at 50 Hz and PD runs at 500 Hz.
- The original BPX 45/1125 model and the default force estimator are immutable artifacts.
- Application acceptance is not declared until an application error budget and recovery-time budget are defined.

## Review Focus

- World/body frame mismatch: a known world-frame force must rotate correctly under yaw, roll and pitch; test in the adapter task.
- History order and timestamp skew: the newest frame must be at the documented position and estimator/policy steps must share one timestamp; test in the runtime task.
- Reset contamination: no pre-reset force or action may remain in either 25-frame cache; test in the cache task.
- Low-confidence and NaN behavior: the policy must receive a bounded fallback and finite action; test in the safety task.
- No-force regression: adding the force channels must not degrade the original walk; test in the training/evaluation task.

### Task 1: Freeze the cross-module data contract

**Files:**
- Create: `bpx_force_rma/contracts/force_estimator_contract.yaml`
- Create: `bpx_force_rma/contracts/policy_observation_contract.yaml`
- Create: `bpx_force_rma/contracts/deploy.yaml`
- Create: `tests/contracts/test_force_policy_contract.py`

**Interfaces:**
- Produces `ForceEstimatorFrame(force_world_N: float[3], confidence: float, timestamp_ns: int, valid: bool)`.
- Produces `PolicyFrame(base_obs: float[45], force_body_norm: float[3], force_confidence: float, timestamp_ns: int)`.
- Produces shape constants `ESTIMATOR_FRAME_DIM=64`, `HISTORY_STEPS=25`, `POLICY_DIM=49`, `HISTORY_DIM=1225`, `ACTION_DIM=12`.

- [ ] **Step 1: Write failing contract tests** for dimensions, feature order, units, frame names, 20 ms period, 50 Hz/500 Hz rates, and unchanged action dimension.
- [ ] **Step 2: Run** `pytest tests/contracts/test_force_policy_contract.py -v`; expected initial failures for missing contracts.
- [ ] **Step 3: Implement** the three YAML contracts and a typed loader that rejects mismatched dimensions or frame/unit declarations.
- [ ] **Step 4: Run** the contract tests; expected PASS.
- [ ] **Step 5: Commit** `feat: freeze force-aware BPX contracts`.

### Task 2: Implement the estimator-to-policy adapter

**Files:**
- Create: `bpx_force_rma/runtime/force_adapter.py`
- Create: `tests/runtime/test_force_adapter.py`

**Interfaces:**
- Consumes `ForceEstimatorFrame` and current root quaternion.
- Produces `PolicyFrame`.
- Signature: `adapt_force(force_world_N: Tensor[B,3], root_quat_w: Tensor[B,4], confidence: Tensor[B,1], timestamp_ns: Tensor[B,1]) -> PolicyFrame`.

- [ ] **Step 1: Write failing tests** for identity rotation, 90° yaw, nonzero roll/pitch, clipping at ±45 N, confidence clipping, NaN fallback and timestamp preservation.
- [ ] **Step 2: Run** `pytest tests/runtime/test_force_adapter.py -v`; expected FAIL before implementation.
- [ ] **Step 3: Implement** `quat_apply_inverse`, divide by 45 N, clip to [-1,1], clip confidence to [0,1], and replace invalid force with zero plus confidence zero.
- [ ] **Step 4: Run** the tests; expected PASS with exact known-vector rotation checks.
- [ ] **Step 5: Commit** `feat: add world-to-body force adapter`.

### Task 3: Implement synchronized 25-frame caches

**Files:**
- Create: `bpx_force_rma/runtime/history.py`
- Create: `tests/runtime/test_history.py`

**Interfaces:**
- `HistoryBuffer(capacity: int, feature_dim: int)`.
- `push(frame: Tensor[B,D], timestamp_ns: Tensor[B,1]) -> Tensor[B,T,D]`.
- `reset(initial_frame: Tensor[B,D], timestamp_ns: Tensor[B,1]) -> None`.
- `latest() -> Tensor[B,D]`.
- `validate_timestamp_pair(estimator_ts, policy_ts) -> None`.

- [ ] **Step 1: Write failing tests** for oldest-first ordering, `history[:, -1] == current_frame`, reset fill, reset clearing and 20 ms timestamp checks.
- [ ] **Step 2: Run** the history tests; expected FAIL.
- [ ] **Step 3: Implement** one shared 50 Hz step transaction that updates estimator and policy caches exactly once.
- [ ] **Step 4: Run** the tests; expected PASS and no stale pre-reset values.
- [ ] **Step 5: Commit** `feat: synchronize estimator and RMA history buffers`.

### Task 4: Build the 49-dimensional BPX observation

**Files:**
- Create: `bpx_force_rma/runtime/observation.py`
- Modify: BPX observation builder in the imported `bpx_traction` project.
- Create: `tests/runtime/test_observation.py`

**Interfaces:**
- `build_policy_frame(base_obs_45, force_body_norm_3, force_confidence_1, timestamp_ns) -> Tensor[B,49]`.
- `build_history_frame(policy_frame_49) -> Tensor[B,49]`.

- [ ] **Step 1: Write failing tests** asserting exact slice offsets, finite values, preserved first 45 dimensions, and action-independent feature order.
- [ ] **Step 2: Run** observation tests; expected FAIL.
- [ ] **Step 3: Implement** append-only observation construction; never reorder or rescale the original 45 dimensions.
- [ ] **Step 4: Run** tests; expected PASS.
- [ ] **Step 5: Commit** `feat: add force-aware BPX observation`.

### Task 5: Add force-aware teacher configuration

**Files:**
- Create: `bpx_force_rma/train/teacher_config.py`
- Create: `bpx_force_rma/train/force_corruption.py`
- Create: `tests/train/test_force_corruption.py`

**Interfaces:**
- `sample_force_corruption(true_force_body_N, rng, config) -> CorruptedForceSample`.
- Corruption config includes additive ±N error, multiplicative error, direction rotation, delay steps, dropout probability and confidence masking.

- [ ] **Step 1: Write failing tests** for zero corruption identity, deterministic seeded corruption, bounded output, delay queue behavior and confidence masking.
- [ ] **Step 2: Run** the tests; expected FAIL.
- [ ] **Step 3: Implement** the corruption model used by the teacher and the exact 0–45 N scenario mixture, including high-Fz, Full3D, 2–5 N direction-tail and Reverse/Release windows.
- [ ] **Step 4: Run** tests and a one-episode smoke rollout; expected PASS with no NaN and documented scenario counts.
- [ ] **Step 5: Commit** `feat: configure force-corrupted privileged teacher`.

### Task 6: Train teacher and RMA student without blind-data leakage

**Files:**
- Create: `bpx_force_rma/train/train_teacher.py`
- Create: `bpx_force_rma/train/train_student.py`
- Create: `bpx_force_rma/train/checkpoint_selection.py`
- Create: `tests/train/test_checkpoint_selection.py`

**Interfaces:**
- `train_teacher(config) -> TeacherCheckpoint`.
- `train_student(config, teacher_checkpoint) -> StudentCheckpoint`.
- `select_checkpoint(validation_metrics, anchor_metrics, transient_metrics) -> SelectionDecision`.

- [ ] **Step 1: Write failing selection tests** requiring fixed-validation improvement, no-force anchor guard, transient guard and no blind metrics.
- [ ] **Step 2: Run** selection tests; expected FAIL.
- [ ] **Step 3: Implement** teacher PPO, frozen-base RMA student distillation, anchor loss, action/latent/stability losses and checkpoint selection.
- [ ] **Step 4: Run** teacher and student smoke training with a small environment count; expected finite losses, checkpoint creation and unchanged action dimension.
- [ ] **Step 5: Run** full training using separate train/validation/independent blind seeds; expected TensorBoard logs for train, validation, anchors and transient metrics.
- [ ] **Step 6: Commit** `feat: train force-aware BPX RMA teacher and student`.

### Task 7: Export and verify Torch/ONNX interfaces

**Files:**
- Create: `bpx_force_rma/export/export_policy.py`
- Create: `bpx_force_rma/export/export_adaptation.py`
- Create: `bpx_force_rma/export/verify_parity.py`
- Create: `tests/export/test_export_contract.py`

**Interfaces:**
- `export_policy(checkpoint, output_dir) -> policy.onnx` with input `[1,49]`, history `[1,1225]`, output `[1,12]`.
- `verify_parity(torch_model, onnx_model, test_inputs, atol, rtol) -> ParityReport`.

- [ ] **Step 1: Write failing export tests** for exact input names, shapes, output shape, finite outputs and deterministic parity.
- [ ] **Step 2: Run** export tests; expected FAIL.
- [ ] **Step 3: Implement** export with dynamic batch only, fixed feature/history dimensions, and generated `deploy.yaml`.
- [ ] **Step 4: Run** parity tests on random, zero-force, high-force and reset-warmup inputs; expected PASS within the configured FP32 tolerance.
- [ ] **Step 5: Commit** `feat: export and verify force-aware BPX ONNX`.

### Task 8: Integrate closed-loop MuJoCo/MJLab runtime

**Files:**
- Create: `bpx_force_rma/runtime/closed_loop.py`
- Create: `bpx_force_rma/runtime/safety_gate.py`
- Create: `tests/runtime/test_closed_loop.py`

**Interfaces:**
- `step(sensor_frame, timestamp_ns) -> ActionFrame`.
- `SafetyGate.apply(policy_frame, previous_action) -> GatedPolicyFrame`.

- [ ] **Step 1: Write failing tests** for estimator timeout, low confidence, force jump, reset, NaN and normal pass-through.
- [ ] **Step 2: Run** tests; expected FAIL.
- [ ] **Step 3: Implement** estimator inference → adapter → cache → policy inference → action scaling, with bounded fallback and logging of raw/normalized force.
- [ ] **Step 4: Run** one 6000-step MuJoCo smoke rollout; expected no shape errors, no stale cache and finite 12-D actions.
- [ ] **Step 5: Commit** `feat: integrate force-aware BPX closed loop`.

### Task 9: Run paired acceptance and update reports/TensorBoard

**Files:**
- Create: `bpx_force_rma/eval/run_paired_acceptance.py`
- Create: `bpx_force_rma/eval/metrics.py`
- Create: `bpx_force_rma/eval/report.py`
- Create: `bpx_force_rma/eval/PROTOCOL.md`

**Interfaces:**
- `run_paired_acceptance(checkpoint, seeds, protocol) -> PairedAcceptanceResult`.
- `compute_metrics(runs) -> MetricsTable`.
- `write_report(result, output_path) -> None`.

- [ ] **Step 1: Freeze the protocol** with 10 fresh paired seeds, unchanged absolute gates, relative metrics, transient peak/recovery metrics and termination comparison.
- [ ] **Step 2: Run baseline and candidate on identical force plans and seeds;** expected complete paired runs with raw provenance.
- [ ] **Step 3: Compute component, magnitude, direction, relative, transient and termination metrics with confidence intervals.
- [ ] **Step 4: Publish namespaced TensorBoard runs;** expected candidate, baseline and aggregate tags.
- [ ] **Step 5: Render the updated desktop report;** expected explicit PASS/FAIL/UNDETERMINED status and no gate relaxation.
- [ ] **Step 6: Commit** `eval: add force-aware BPX paired acceptance`.

### Task 10: Package, rollback and deployment gate

**Files:**
- Create: `bpx_force_rma/deploy/manifest.yaml`
- Create: `bpx_force_rma/deploy/package.py`
- Create: `bpx_force_rma/deploy/rollback.md`
- Create: `tests/deploy/test_manifest.py`

**Interfaces:**
- `build_package(checkpoint, exports, contracts, report) -> DeploymentManifest`.
- `validate_manifest(manifest) -> None`.

- [ ] **Step 1: Write failing manifest tests** requiring model hashes, contract version, training data hash, export parity result, acceptance result and rollback artifact.
- [ ] **Step 2: Implement** package creation that keeps the old 45/1125 BPX and current estimator available for immediate rollback.
- [ ] **Step 3: Run** manifest tests and verify hashes; expected PASS.
- [ ] **Step 4: Allow promotion only when all mandatory gates pass;** otherwise leave the new model experimental and keep default deployment unchanged.
- [ ] **Step 5: Commit** `ops: package force-aware BPX deployment with rollback`.

## Execution order

Run Tasks 1–4 first and review the interface tests. Then run Tasks 5–6 for training, Tasks 7–8 for export and closed-loop integration, and Tasks 9–10 only after the candidate is frozen. Do not start full training before Tasks 1–4 pass.

## Completion criteria

The implementation is complete only when the new 49/1225 Torch and ONNX interfaces pass contract/parity tests, no-force regression is within the frozen baseline tolerance, all fresh paired MuJoCo acceptance results are published, and the deployment manifest contains a verified rollback path. A failure or inconclusive acceptance result leaves the new candidate unpromoted.
