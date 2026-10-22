from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ForceCorruptionConfig:
    additive_bias_N: float = 0.0
    multiplicative_bias: float = 0.0
    direction_noise_deg: float = 0.0
    delay_steps: int = 0
    dropout_prob: float = 0.0
    max_force_N: float = 45.0

@dataclass(frozen=True)
class CorruptedForceSample:
    force_body_N: np.ndarray
    confidence: float

def _rotate(vec, angle, axis):
    axis=np.asarray(axis,dtype=float); axis=axis/np.linalg.norm(axis)
    v=np.asarray(vec,dtype=float); return v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+axis*np.dot(axis,v)*(1-np.cos(angle))

def sample_force_corruption(true_force_body_N, config: ForceCorruptionConfig, rng=None, history=None):
    rng = rng or np.random.RandomState()
    history = history or []
    if config.delay_steps > 0 and len(history) >= config.delay_steps:
        force = np.asarray(history[-config.delay_steps], dtype=float).copy()
    else:
        force = np.asarray(true_force_body_N, dtype=float).copy()
    confidence = 1.0
    if config.dropout_prob > 0 and rng.rand() < config.dropout_prob:
        return CorruptedForceSample(np.zeros(3), 0.0)
    if config.multiplicative_bias:
        force *= 1.0 + rng.uniform(-config.multiplicative_bias, config.multiplicative_bias)
    if config.additive_bias_N:
        force += rng.uniform(-config.additive_bias_N, config.additive_bias_N, size=3)
    if config.direction_noise_deg:
        axis=rng.normal(size=3); angle=np.deg2rad(rng.uniform(-config.direction_noise_deg, config.direction_noise_deg)); force=_rotate(force,angle,axis)
    force=np.clip(force,-config.max_force_N,config.max_force_N)
    confidence=max(0.0, 1.0 - min(1.0, config.additive_bias_N/5.0 + config.direction_noise_deg/30.0 + (config.delay_steps/10.0)))
    return CorruptedForceSample(force, confidence)
