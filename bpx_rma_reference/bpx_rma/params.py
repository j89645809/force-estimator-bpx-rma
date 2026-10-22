"""读 `data/bpx_pro_training_params.yaml`，给出训练要用的标称值和随机化区间。

所有返回的 12 维列表都按 :data:`bpx_rma.spec.JOINT_NAMES`（SDK / 策略顺序）排列。

用环境变量 ``BPX_PRO_PARAMS_YAML`` 可以换一份参数文件——做"厂家参数 vs 辨识参数"
的对照实验时就是换这个（见 ``data/bpx_pro_baseline_params.yaml``）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import spec


@dataclass(frozen=True)
class JointParamRanges:
    """单个参数在 12 个关节上的标称值与随机化上下界。"""

    nominal: list[float]
    low: list[float]
    high: list[float]

    def __post_init__(self) -> None:
        n = spec.NUM_JOINTS
        assert len(self.nominal) == len(self.low) == len(self.high) == n


@dataclass(frozen=True)
class KneeLoadFrictionCfg:
    """承重摩擦的负载模型 F = f0 + mu · L（方案 B）。"""

    enabled: bool
    f0: float
    f0_range: tuple[float, float]
    mu: float
    mu_range: tuple[float, float]
    tau_s: float
    v_move: float
    clamp: tuple[float, float]


@dataclass(frozen=True)
class TrainingParams:
    """`bpx_pro_training_params.yaml` 的结构化视图。"""

    source: Path
    armature: JointParamRanges
    damping: JointParamRanges
    frictionloss: JointParamRanges
    mass_total_kg: float
    mass_range_kg: tuple[float, float]
    delay_steps: float
    delay_steps_range: tuple[float, float]
    encoder_bias_range: tuple[float, float]
    effort_limit_deploy: float
    saturation_factor: float
    kp: float
    kd: float
    action_scale: float
    default_joint_pos: list[float]
    knee_load_friction: KneeLoadFrictionCfg
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def max_delay_steps(self) -> int:
        """PaceDCMotor 的延迟缓冲长度，必须 >= 随机化上界。"""
        return int(-(-self.delay_steps_range[1] // 1))  # ceil


def _ranges(nominal: list[float], lo: list[float], hi: list[float]) -> JointParamRanges:
    return JointParamRanges(nominal=nominal, low=lo, high=hi)


def _frac_range(nominal: list[float], fracs: list[float]) -> tuple[list[float], list[float]]:
    """按比例给上下界，下界不小于 0。"""
    lo = [max(0.0, v * (1.0 - f)) for v, f in zip(nominal, fracs)]
    hi = [v * (1.0 + f) for v, f in zip(nominal, fracs)]
    return lo, hi


def load_params(path: str | os.PathLike | None = None) -> TrainingParams:
    """读参数 yaml。``path`` 为 None 时用环境变量 BPX_PRO_PARAMS_YAML，再退回默认文件。"""
    if path is None:
        path = os.environ.get("BPX_PRO_PARAMS_YAML") or spec.DEFAULT_PARAMS_YAML
    path = Path(path).expanduser().resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)

    nominal = raw["nominal"]
    missing = [n for n in spec.JOINT_NAMES if n not in nominal]
    if missing:
        raise ValueError(f"{path} 的 nominal 缺关节：{missing}")

    arm_n = [float(nominal[n]["armature"]) for n in spec.JOINT_NAMES]
    dmp_n = [float(nominal[n]["damping"]) for n in spec.JOINT_NAMES]
    fri_n = [float(nominal[n]["frictionloss"]) for n in spec.JOINT_NAMES]

    rnd = raw["randomization"]

    def bounds(key: str, nominal: list[float]) -> tuple[list[float], list[float]]:
        """每个参数可以按比例给（``<key>_frac``）或按绝对区间给（``<key>_range``）。

        标称值是 0 的时候（比如厂家 baseline 的 armature / 摩擦）比例法会退化成 [0, 0]，
        必须用绝对区间。
        """
        lo: list[float] = []
        hi: list[float] = []
        for name, v in zip(spec.JOINT_NAMES, nominal):
            cfg = rnd[spec.joint_type(name)]
            if f"{key}_range" in cfg:
                a, b = (float(x) for x in cfg[f"{key}_range"])
            elif f"{key}_frac" in cfg:
                f = float(cfg[f"{key}_frac"])
                a, b = max(0.0, v * (1.0 - f)), v * (1.0 + f)
            else:
                raise ValueError(f"{path} 的 randomization.{spec.joint_type(name)} 缺 {key}_frac 或 {key}_range")
            lo.append(a)
            hi.append(b)
        return lo, hi

    arm_lo, arm_hi = bounds("armature", arm_n)
    dmp_lo, dmp_hi = bounds("damping", dmp_n)
    fri_lo, fri_hi = bounds("frictionloss", fri_n)

    delay_ms = raw["delay"]
    ms_to_steps = 1e-3 / spec.SIM_DT   # 1 ms = 0.5 步 @ 500 Hz
    kf = raw["knee_friction"]["load_model"]

    return TrainingParams(
        source=path,
        armature=_ranges(arm_n, arm_lo, arm_hi),
        damping=_ranges(dmp_n, dmp_lo, dmp_hi),
        frictionloss=_ranges(fri_n, fri_lo, fri_hi),
        mass_total_kg=float(raw["mass"]["total_kg"]),
        mass_range_kg=tuple(float(x) for x in raw["mass"]["range_kg"]),  # type: ignore[arg-type]
        delay_steps=float(delay_ms["physical_ms"]) * ms_to_steps,
        delay_steps_range=tuple(float(x) * ms_to_steps for x in delay_ms["range_ms"]),  # type: ignore[arg-type]
        encoder_bias_range=tuple(float(x) for x in raw["encoder_bias"]["range_rad"]),  # type: ignore[arg-type]
        effort_limit_deploy=float(raw["actuator"]["effort_limit_deploy"]),
        saturation_factor=float(raw["actuator"]["saturation_factor"]),
        kp=float(raw["pd"]["kp"]),
        kd=float(raw["pd"]["kd"]),
        action_scale=float(raw["pd"]["action_scale"]),
        default_joint_pos=list(raw["pd"]["default_joint_pos"]) * 4,
        knee_load_friction=KneeLoadFrictionCfg(
            enabled=bool(kf["enabled"]),
            f0=float(kf["f0"]),
            f0_range=tuple(float(x) for x in kf["f0_range"]),  # type: ignore[arg-type]
            mu=float(kf["mu"]),
            mu_range=tuple(float(x) for x in kf["mu_range"]),  # type: ignore[arg-type]
            tau_s=float(kf["tau_s"]),
            v_move=float(kf["v_move"]),
            clamp=tuple(float(x) for x in kf["clamp"]),  # type: ignore[arg-type]
        ),
        raw=raw,
    )


@lru_cache(maxsize=4)
def get_params(path: str | None = None) -> TrainingParams:
    """带缓存的 :func:`load_params`，配置类在 import 期就会调它。"""
    p = load_params(path)
    # 和部署契约交叉检查：参数文件里的 PD / action scale / q0 必须和板子一致
    if (p.kp, p.kd, p.action_scale) != (spec.KP_POLICY, spec.KD_POLICY, spec.ACTION_SCALE):
        raise ValueError(
            f"{p.source} 的 PD/action_scale 与部署契约不符："
            f"yaml=({p.kp}, {p.kd}, {p.action_scale}) vs 板子=({spec.KP_POLICY}, {spec.KD_POLICY}, {spec.ACTION_SCALE})"
        )
    if p.default_joint_pos != spec.DEFAULT_JOINT_POS:
        raise ValueError(f"{p.source} 的 default_joint_pos 与板子上的 DEFAULT_POSE 不符")
    return p
