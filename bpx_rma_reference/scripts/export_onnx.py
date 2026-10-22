#!/usr/bin/env python3
"""把蒸馏出来的 student 导成板子能直接吃的 ONNX。

板子上 `make_policy_session()` 会逐项校验，必须完全对上：

    输入  policy              float32  [1, 45]
    输入  adaptation_history  float32  [1, 1125]
    输出  actions             float32  [1, 12]

归一化统计量烤进计算图，所以板子送的是原始物理量，部署代码一行不用改。
``clamp(±4)`` 也烤进去了 —— 板子那边还会再 clip 一次，是故意的双保险。

    conda activate bpx_rma
    python scripts/export_onnx.py --headless

默认导出 `logs/rsl_rl/bpx_pro_flat/` 下最新 `*_student` run 的最新 checkpoint，
输出到该 run 的 `exported/policy.onnx`。导完会用随机输入核对 ONNX 和 torch 的输出
（要求最大差 < 1e-5），不一致直接报错。
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="导出 BPX-Pro RMA student 为部署用 ONNX")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-Play-v0")
parser.add_argument("--load-run", dest="load_run", type=str, default=".*_student")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--experiment-name", dest="experiment_name", type=str, default=None,
                    help="覆盖实验名，用来导对照组（如 bpx_pro_flat_baseline）")
parser.add_argument("--out", type=str, default=None, help="输出目录，默认 <run>/exported")
parser.add_argument("--opset", type=int, default=17)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401
from bpx_rma import spec  # noqa: E402


class RmaOnnxPolicy(nn.Module):
    """部署图：ẑ = φ(历史)，a = clamp(π(本体感知, ẑ), ±4)。

    归一化器也复制进来，所以板子送原始物理量即可。
    """

    def __init__(self, student) -> None:
        super().__init__()
        if student._base_group != "policy" or student._enc_groups != ["history"]:
            raise ValueError(
                f"student 的观测组是 base={student._base_group}、encoder={student._enc_groups}，"
                "期望 base='policy'、encoder=['history']"
            )
        self.base_normalizer = copy.deepcopy(student.base_normalizer).eval()
        self.enc_normalizer = copy.deepcopy(student.enc_normalizer).eval()
        self.encoder = copy.deepcopy(student.encoder).eval()
        self.mlp = copy.deepcopy(student.mlp).eval()
        self.clip = float(spec.ACTION_CLIP)

    def forward(self, policy: torch.Tensor, adaptation_history: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.enc_normalizer(adaptation_history))
        x = torch.cat([self.base_normalizer(policy), z], dim=-1)
        return torch.clamp(self.mlp(x), -self.clip, self.clip)


class RmaOnnxAdaptation(nn.Module):
    """只出隐变量 ẑ 的图，**不部署**，给离线分析用。

    拿真机录的 `rec_*_policy50.bin`（里面有 obs00..obs44）重建 25 帧历史喂进来，
    得到真机上的 ẑ(t)；再把辨识参数喂给 teacher 的 μ 得到 z*，两者比对 ——
    这才是"用策略验证辨识结果"的定量证据。
    """

    def __init__(self, student) -> None:
        super().__init__()
        self.enc_normalizer = copy.deepcopy(student.enc_normalizer).eval()
        self.encoder = copy.deepcopy(student.encoder).eval()

    def forward(self, adaptation_history: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.enc_normalizer(adaptation_history))


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_distillation_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name

    log_root = (PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    ckpt = get_checkpoint_path(str(log_root), args_cli.load_run, args_cli.checkpoint)
    print(f"[bpx] student checkpoint：{ckpt}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt, load_cfg={"student": True, "teacher": True, "iteration": False})
    student = runner.alg.student
    student.eval()

    # --- 维度必须和板子写死的一致 -------------------------------------------
    obs = env.get_observations()          # rsl_rl 5.x 的 wrapper 只返回 TensorDict
    n_policy = obs["policy"].shape[-1]
    n_hist = obs["history"].shape[-1]
    if (n_policy, n_hist) != (spec.OBS_FRAME_SIZE, spec.HISTORY_INPUT_SIZE):
        raise ValueError(
            f"观测维度和部署契约不符：policy={n_policy}（要 {spec.OBS_FRAME_SIZE}）、"
            f"history={n_hist}（要 {spec.HISTORY_INPUT_SIZE}）"
        )

    wrapper = RmaOnnxPolicy(student).to("cpu").eval()
    out_dir = Path(args_cli.out) if args_cli.out else Path(ckpt).parent / "exported"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "policy.onnx"

    dummy_policy = torch.zeros(1, spec.OBS_FRAME_SIZE)
    dummy_hist = torch.zeros(1, spec.HISTORY_INPUT_SIZE)
    torch.onnx.export(
        wrapper,
        (dummy_policy, dummy_hist),
        str(onnx_path),
        input_names=[spec.ONNX_INPUT_POLICY, spec.ONNX_INPUT_HISTORY],
        output_names=[spec.ONNX_OUTPUT],
        dynamic_axes=None,          # 板子要求 batch 固定为 1
        opset_version=args_cli.opset,
        do_constant_folding=True,
    )
    print(f"[bpx] 已导出 {onnx_path}")

    # --- 和 torch 对数 -------------------------------------------------------
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = {i.name: list(i.shape) for i in sess.get_inputs()}
    expect = {
        spec.ONNX_INPUT_POLICY: [1, spec.OBS_FRAME_SIZE],
        spec.ONNX_INPUT_HISTORY: [1, spec.HISTORY_INPUT_SIZE],
    }
    assert names == expect, f"ONNX 输入不符：{names} != {expect}"
    out0 = sess.get_outputs()[0]
    assert out0.name == spec.ONNX_OUTPUT and list(out0.shape) == [1, 12], f"ONNX 输出不符：{out0}"

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(32):
        p = rng.standard_normal((1, spec.OBS_FRAME_SIZE)).astype(np.float32)
        h = rng.standard_normal((1, spec.HISTORY_INPUT_SIZE)).astype(np.float32)
        a_onnx = sess.run(None, {spec.ONNX_INPUT_POLICY: p, spec.ONNX_INPUT_HISTORY: h})[0]
        with torch.no_grad():
            a_torch = wrapper(torch.from_numpy(p), torch.from_numpy(h)).numpy()
        worst = max(worst, float(np.abs(a_onnx - a_torch).max()))
    if worst > 1e-5:
        raise AssertionError(f"ONNX 和 torch 输出不一致，最大差 {worst:.3e}")
    print(f"[bpx] ONNX vs torch 最大差 {worst:.2e}，通过")

    # --- 隐变量图（离线分析用，不上板子）-----------------------------------
    adapt = RmaOnnxAdaptation(student).to("cpu").eval()
    adapt_path = out_dir / "adaptation.onnx"
    torch.onnx.export(
        adapt,
        (dummy_hist,),
        str(adapt_path),
        input_names=[spec.ONNX_INPUT_HISTORY],
        output_names=["latent"],
        dynamic_axes={spec.ONNX_INPUT_HISTORY: {0: "batch"}, "latent": {0: "batch"}},
        opset_version=args_cli.opset,
        do_constant_folding=True,
    )
    print(f"[bpx] 已导出 {adapt_path}（z 维度 {student.latent_dim}，离线比对辨识参数用，不要放板子上）")

    # --- 部署侧的配置文件 ----------------------------------------------------
    deploy_yaml = out_dir / "deploy.yaml"
    deploy_yaml.write_text(
        "# 这份要和 policy.onnx 一起放到板子的\n"
        "#   deploy/robots/bpx/config/policy/velocity_rma/v0/params/deploy.yaml\n"
        "#\n"
        "# hardware_joint_convention_confirmed 是板子上的安全闸门（门2）：\n"
        "# 必须人工在真机上确认关节正负方向、零位、顺序和训练一致之后才能改成 true。\n"
        "# 训练脚本不会替你确认，所以这里固定导成 false。\n"
        "hardware_joint_convention_confirmed: false\n"
        f"\n# 导出来源\ncheckpoint: {ckpt}\n"
        f"obs_frame_size: {spec.OBS_FRAME_SIZE}\n"
        f"history_length: {spec.HISTORY_LENGTH}\n"
        f"action_scale: {spec.ACTION_SCALE}\n"
        f"action_clip: {spec.ACTION_CLIP}\n"
        f"kp: {spec.KP_POLICY}\nkd: {spec.KD_POLICY}\n"
        f"default_joint_pos: {spec.DEFAULT_JOINT_POS}\n"
        f"joint_order: {spec.JOINT_NAMES}\n"
    )
    print(f"[bpx] 已写 {deploy_yaml}（注意 hardware_joint_convention_confirmed 是 false，要人工确认后才改）")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
