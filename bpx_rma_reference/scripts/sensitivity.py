#!/usr/bin/env python3
"""可观测性检查：走路时辨识出来的那几个参数到底占多大分量？

这是开对照实验**之前**必须先回答的问题。落地实验的教训是：动作太慢的话，
惯量 / 阻尼 / 延迟占膝力矩不到 1%，两套参数训出来的策略会长得一样，对比没有分辨力。

做法：让策略在给定速度下走，逐步记录每个关节的
  * 执行器实际输出力矩 tau
  * 关节速度 dq、加速度 ddq
  * 这一回合实际抽到的 armature / 粘滞 / 库伦摩擦
然后算三项物理效应的大小，和 tau 比：

  惯量项   I_a · ddq        （armature，高速摆腿时最大）
  粘滞项   b · dq           （随速度线性增长）
  库伦项   F · sign(dq)     （换向时最关键）

判读：某一项占 tau 的比例 >10% 就说明这个参数被充分激励，辨识得准不准会影响策略；
<1% 说明这个任务测不出它，跟落地实验一样。

    conda activate bpx_rma
    python scripts/sensitivity.py --headless --vx 1.5
    python scripts/sensitivity.py --headless --vx 0.5     # 对照：慢速下是不是真的测不出
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="辨识参数在走路任务里的可观测性")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--load-run", dest="load_run", type=str, default=".*_student")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--warmup", type=int, default=100, help="前多少步不统计（起步加速阶段）")
parser.add_argument("--vx", type=float, default=1.5)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401
from bpx_rma import spec  # noqa: E402
from bpx_rma.mdp.state import get_state  # noqa: E402

GROUPS = {"hip_roll": [0, 3, 6, 9], "hip_pitch": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
    # 关掉外力扰动，免得污染力矩统计
    env_cfg.events.push_robot = None

    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    is_distill = agent_cfg.class_name == "DistillationRunner"
    log_root = (PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    ckpt = get_checkpoint_path(str(log_root), args_cli.load_run, "model_.*.pt")
    print(f"[bpx] checkpoint：{ckpt}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = DistillationRunner if is_distill else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt, load_cfg={"student": True, "teacher": True, "iteration": False} if is_distill else None)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    st = get_state(base)
    st.resolve_joint_ids(robot)
    jids = st.joint_ids_sdk

    acc: dict[str, list[torch.Tensor]] = {k: [] for k in ("tau", "inertia", "viscous", "coulomb", "dq")}
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(args_cli.steps):
            obs, _, _, _ = env.step(policy(obs))
            if step < args_cli.warmup:
                continue
            tau = robot.data.applied_torque[:, jids]
            dq = robot.data.joint_vel[:, jids]
            ddq = robot.data.joint_acc[:, jids]
            acc["tau"].append(tau.abs())
            acc["inertia"].append((st.armature * ddq).abs())
            acc["viscous"].append((st.viscous * dq).abs())
            acc["coulomb"].append(st.friction.abs())        # 库伦摩擦与速度无关，只看幅值
            acc["dq"].append(dq.abs())

    data = {k: torch.stack(v) for k, v in acc.items()}      # (T, N, 12)
    n = data["tau"].shape[0]

    print(f"\n{'='*78}")
    print(f"指令 vx={args_cli.vx} vy={args_cli.vy} wz={args_cli.wz}  "
          f"| {args_cli.num_envs} 环境 × {n} 步（已跳过前 {args_cli.warmup} 步）")
    print(f"{'='*78}")
    print(f"{'关节组':<11} {'|tau| 均值':>10} {'峰值速度':>9} "
          f"{'惯量占比':>9} {'粘滞占比':>9} {'库伦占比':>9}   判读")
    print("-" * 78)

    for name, cols in GROUPS.items():
        tau = data["tau"][:, :, cols]
        tau_mean = tau.mean().item()
        dq_peak = data["dq"][:, :, cols].quantile(0.99).item()
        fracs = {}
        for key in ("inertia", "viscous", "coulomb"):
            # 每个样本先算比例再取中位数，避免 tau 接近 0 的点把均值拉爆
            r = data[key][:, :, cols] / data["tau"][:, :, cols].clamp(min=0.5)
            fracs[key] = r.median().item()
        worst = max(fracs.values())
        verdict = "可分辨" if worst > 0.10 else ("偏弱" if worst > 0.03 else "测不出")
        print(f"{name:<11} {tau_mean:>9.2f}N·m {dq_peak:>7.2f}r/s "
              f"{fracs['inertia']:>8.1%} {fracs['viscous']:>8.1%} {fracs['coulomb']:>8.1%}   {verdict}")

    print("-" * 78)
    print("判读标准：某项 >10% = 该参数被充分激励，辨识准不准会影响策略；")
    print("          3–10% = 偏弱；<3% = 这个任务测不出它（和落地实验慢动作时一样）。")
    print("库伦项是摩擦力矩的绝对幅值与驱动力矩之比，它在换向瞬间最关键，和速度无关。")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
