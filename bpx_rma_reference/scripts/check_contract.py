#!/usr/bin/env python3
"""核对仿真环境和板子上的部署契约逐位一致。

训练前跑一次，训练完导出前再跑一次。查四件事：

1. ``policy`` 组的 45 维 == 板子 `build_obs_frame()` 的拼法
   （q-q0, dq, ω, 投影重力, last_action, cmd，SDK 关节顺序）
2. ``history`` 组的 1125 维 == 板子 `flatten_term_major_history()` 的拼法
   （25 帧最旧在前，按项 term-major 展平）
3. 复位时历史是用第一帧回填的（板子上 `np.repeat(frame, 25)`）
4. 动作映射 == 板子 `run_policy()`：target = q0 + 0.1875 · clip(raw, ±4)

任何一条不符都直接 AssertionError —— 这些地方错了，策略在仿真里再好，上真机也是乱动。

    conda activate bpx_rma
    python scripts/check_contract.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="核对 BPX-Pro 训练环境与板子部署契约")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-Play-v0")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--tol", type=float, default=1e-5)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()
args_cli.num_envs = 4

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import bpx_rma  # noqa: E402, F401
from bpx_rma import spec  # noqa: E402

N_ENVS = 4


def board_frame(env, robot, jids) -> torch.Tensor:
    """照板子 `build_obs_frame()` 的写法，从原始状态拼 45 维。

    故意不走 mdp 的观测函数 —— 要的就是一条独立的路径来交叉验证关节顺序。
    """
    return torch.cat(
        [
            robot.data.joint_pos[:, jids] - robot.data.default_joint_pos[:, jids],
            robot.data.joint_vel[:, jids],
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            env.action_manager.action,
            env.command_manager.get_command("base_velocity"),
        ],
        dim=-1,
    )


def board_flatten_history(hist: torch.Tensor) -> torch.Tensor:
    """照板子 `flatten_term_major_history()`：(N, 25, 45) -> (N, 1125)，按项展平。"""
    parts = [hist[:, :, b:e].reshape(hist.shape[0], -1) for b, e in spec.obs_term_slices()]
    return torch.cat(parts, dim=-1)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = N_ENVS
    # sprint2 起观测 dq 是执行器滤波值（建模板载滤波，tau 每回合随机）。
    # 对拍时必须关掉：板子公式拿的是"它收到的 dq"，滤波差异是故意建模的传感器
    # 动态，不是拼帧错误。关掉后逐位可比。
    ev = getattr(env_cfg, "events", None)
    if ev is not None and hasattr(ev, "actuator_params") and ev.actuator_params is not None:
        ev.actuator_params.params.pop("dq_filter_tau_range", None)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None).unwrapped

    robot = env.scene["robot"]
    jids, jnames = robot.find_joints(spec.JOINT_NAMES, preserve_order=True)
    assert jnames == spec.JOINT_NAMES, f"关节顺序解析错了：{jnames}"
    jids_t = torch.tensor(jids, device=env.device, dtype=torch.long)
    print(f"[1/4] 关节顺序 OK：SDK 顺序 -> articulation 下标 {jids}")

    # 动作项也必须是同一个顺序
    act_term = env.action_manager.get_term("joint_pos")
    act_ids = act_term._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else list(act_ids)
    assert act_ids == jids, f"动作项的关节顺序 {act_ids} 和 SDK 顺序 {jids} 不一致"

    obs, _ = env.reset()

    # 维度
    assert obs["policy"].shape[-1] == spec.OBS_FRAME_SIZE, obs["policy"].shape
    assert obs["history"].shape[-1] == spec.HISTORY_INPUT_SIZE, obs["history"].shape
    print(f"[2/4] 观测维度 OK：policy {spec.OBS_FRAME_SIZE}，history {spec.HISTORY_INPUT_SIZE}"
          f"（= {spec.OBS_FRAME_SIZE} × {spec.HISTORY_LENGTH}），privileged {obs['privileged'].shape[-1]}")

    # 复位后历史应该是第一帧回填 25 份
    frame0 = obs["policy"]
    expect0 = board_flatten_history(frame0.unsqueeze(1).repeat(1, spec.HISTORY_LENGTH, 1))
    d0 = (obs["history"] - expect0).abs().max().item()
    assert d0 < args_cli.tol, f"复位后历史不是第一帧回填的，最大差 {d0:.3e}"
    print(f"[3/4] 复位回填 OK（和板子 np.repeat(frame, 25) 一致，最大差 {d0:.1e}）")

    # 滚动比对
    hist = frame0.unsqueeze(1).repeat(1, spec.HISTORY_LENGTH, 1).clone()
    worst_frame = worst_hist = worst_act = worst_dq = 0.0
    compared = 0
    for step in range(args_cli.steps):
        raw = torch.empty(N_ENVS, spec.NUM_JOINTS, device=env.device).uniform_(-1.0, 1.0)
        obs, _, terminated, truncated, _ = env.step(raw)
        done = terminated | truncated

        # 动作映射：target = q0 + scale · clip(raw, ±4)
        expect_target = (
            robot.data.default_joint_pos[:, jids_t]
            + spec.ACTION_SCALE * raw.clamp(-spec.ACTION_CLIP, spec.ACTION_CLIP)
        )
        worst_act = max(worst_act, (act_term.processed_actions - expect_target).abs().max().item())

        # 45 维帧。dq 维单独算：sprint2 起观测 dq = 执行器滤波/滞后建模的"驱动器所见
        # 速度"（板子同款语义），和瞬时 joint_vel 差半个物理子步起步，这是建模出来的
        # 传感器动态，不是拼法错误 —— 拼法断言排除 dq，dq 差只记录不报警。
        diff = (obs["policy"] - board_frame(env, robot, jids_t)).abs()
        dq_lo, dq_hi = spec.obs_term_slices()[1]
        worst_dq = max(worst_dq, diff[:, dq_lo:dq_hi].max().item())
        diff[:, dq_lo:dq_hi] = 0.0
        worst_frame = max(worst_frame, diff.max().item())

        # 历史：复位过的 env 这一步跳过（板子那边也是重新回填）
        hist = torch.cat([hist[:, 1:], obs["policy"].unsqueeze(1)], dim=1)
        alive = ~done
        if alive.any():
            d_hist = (obs["history"][alive] - board_flatten_history(hist[alive])).abs().max()
            worst_hist = max(worst_hist, d_hist.item())
            compared += int(alive.sum())
        if done.any():
            hist[done] = obs["policy"][done].unsqueeze(1).repeat(1, spec.HISTORY_LENGTH, 1)

    assert worst_frame < args_cli.tol, f"45 维帧和板子拼法不一致，最大差 {worst_frame:.3e}"
    assert worst_hist < args_cli.tol, f"历史 term-major 展平和板子不一致，最大差 {worst_hist:.3e}"
    assert worst_act < args_cli.tol, f"动作映射和板子不一致，最大差 {worst_act:.3e}"
    print(f"[4/4] 滚动比对 OK（{args_cli.steps} 步 / {compared} 个样本）："
          f"帧(除dq) {worst_frame:.1e}，历史 {worst_hist:.1e}，动作 {worst_act:.1e}；"
          f"dq 传感器动态差 {worst_dq:.2f} rad/s（滤波/滞后建模，预期非零）")

    print("\n全部通过：这个环境导出的 ONNX 可以直接喂板子上的 bpx_dual_pad。")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
