from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PolicyForce:
    force_body_norm: np.ndarray
    confidence: float
    timestamp_ns: int

def quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w = q[..., 0:1]; xyz = -q[..., 1:4]
    # Apply the conjugate quaternion for the world-to-body inverse rotation.
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)

def adapt_force(force_world_N: np.ndarray, root_quat_w: np.ndarray, confidence: np.ndarray, timestamp_ns: int) -> PolicyForce:
    force = np.asarray(force_world_N, dtype=np.float64)
    quat = np.asarray(root_quat_w, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if force.ndim != 2 or force.shape[1] != 3 or quat.shape != force.shape[:1] + (4,):
        raise ValueError("force must be [B,3] and quaternion must be [B,4]")
    valid = np.isfinite(force).all(axis=1) & np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > 1e-8)
    force_body = np.zeros_like(force)
    if valid.any():
        force_body[valid] = quat_apply_inverse(quat[valid], force[valid])
    force_norm = np.clip(np.nan_to_num(force_body / 45.0), -1.0, 1.0)
    conf_out = float(np.clip(conf[0], 0.0, 1.0)) if len(conf) else 0.0
    if not valid.all():
        conf_out = 0.0
    return PolicyForce(force_norm, conf_out, int(timestamp_ns))
