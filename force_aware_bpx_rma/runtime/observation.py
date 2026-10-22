from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PolicyFrame:
    values: np.ndarray
    timestamp_ns: int

def build_policy_frame(base_obs_45: np.ndarray, force_body_norm_3: np.ndarray, force_confidence_1: np.ndarray, timestamp_ns: int) -> PolicyFrame:
    base = np.asarray(base_obs_45, dtype=np.float32)
    force = np.asarray(force_body_norm_3, dtype=np.float32)
    conf = np.asarray(force_confidence_1, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != 45:
        raise ValueError("base observation must be [B,45]")
    if force.shape != (base.shape[0], 3) or conf.shape != (base.shape[0], 1):
        raise ValueError("force/confidence dimensions do not match base observation")
    values = np.concatenate((base, np.clip(force, -1.0, 1.0), np.clip(conf, 0.0, 1.0)), axis=1)
    if not np.isfinite(values).all():
        raise ValueError("policy frame must be finite")
    return PolicyFrame(values=values, timestamp_ns=int(timestamp_ns))

def build_history_frame(policy_history: np.ndarray) -> np.ndarray:
    history = np.asarray(policy_history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (25,49):
        raise ValueError("policy history must be [B,25,49]")
    if not np.isfinite(history).all():
        raise ValueError("policy history must be finite")
    return history.copy()
