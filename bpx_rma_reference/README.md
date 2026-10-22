# BPX-Pro RMA：用策略验证执行器辨识结果

先在 IsaacLab 里用**辨识出来的执行器参数**训一个 RMA（师生蒸馏）走路策略，导出成板子
能直接吃的 ONNX，然后在真机上和厂家策略同条件对比。

这比 sim2sim 强在：板子上现在跑的厂家策略**本身就是 RMA**，接口是写死的
（`robot_cat/bpx_dual_pad_flush.py` 的 `make_policy_session()` 逐项校验 ONNX）。
只要按同一个契约导出，就能原地替换、同条件对比 —— 是真机验证，不是仿真自说自话。

## 环境

conda 环境 **`bpx_rma`**（克隆自 `dolang_barrier`）。不碰 `env_pace` / `env_isaaclab`。

| | |
|---|---|
| IsaacLab | 2.3.2（`/home/wangyun/IsaacLab`，isaaclab_rl 0.5.1）|
| rsl-rl-lib | 5.0.1（`student`/`teacher` 两个 model cfg 的新 API）|
| Isaac Sim | `~/isaac_sim/isaac-sim`，由 `etc/conda/activate.d/setenv.sh` 挂入 |

必须 `conda activate bpx_rma`，不能直接调 `envs/bpx_rma/bin/python`（那样缺 `pxr`）。
本项目**没有** `pip install -e`，脚本自己往 `sys.path` 里挂项目目录。

机器人资产走 `BPX_PRO_ASSET_DIR` / `BPX_PRO_USD_PATH`，默认找
`~/intern/ZGYD/pace/BPX-Pro/usd/bpxPro.usd`（只读）。

## 跑

```bash
conda activate bpx_rma
cd ~/intern/ZGYD/bpx

python scripts/check_contract.py --headless                     # 先验契约，必过
python scripts/train.py --headless --num_envs 4096              # 阶段一：特权 PPO = teacher
python scripts/train.py --headless --num_envs 4096 \
    --agent rsl_rl_distillation_cfg_entry_point                 # 阶段二：蒸馏 = student（RMA）
python scripts/play.py --headless --steps 1000 --report         # 看指标
python scripts/export_onnx.py --headless                        # 导出 policy.onnx + deploy.yaml
```

阶段二自动从 `logs/rsl_rl/bpx_pro_flat/` 下最新的 `*_teacher` run 取 checkpoint 当 teacher。

## 部署契约（改之前先读）

全部来自板子上的 `bpx_dual_pad.py`，写死在 `bpx_rma/spec.py` 里，`check_contract.py`
每一条都验：

| 项 | 值 |
|---|---|
| 关节顺序 | `fl, fr, hl, hr × (hip_roll, hip_pitch, knee)`，= bpx_sdk `JointIndex` |
| 45 维帧 | `q-q0 (12), dq (12), ω (3), 投影重力 (3), last_action (12), cmd (3)` |
| 历史 | 25 帧，最旧在前，**term-major** 展平 = 1125 |
| ONNX | 输入 `policy` [1,45] + `adaptation_history` [1,1125]，输出 `actions` [1,12] |
| 动作 | `target = q0 + 0.1875 · clip(raw, ±4)`，`q0 = [0, 0.7, −1.4] × 4` |
| PD | kp 40 / kd 1，力矩限幅 **30 N·m**，50 Hz 策略 / 500 Hz 内环 |

IsaacLab 内部关节顺序是 `[0,4,8,1,5,9,...]`（交错的），和 SDK 顺序完全不同 ——
所有按关节的项都必须 `preserve_order=True`，否则 12 个关节会接错。

观测**默认不加噪声**。板子上 `history[-1]` 就是同一帧 `policy`，逐位相等；IsaacLab 里
`policy` 和 `history` 是两个独立的组，各自采噪声就会破坏这个恒等关系。要加噪声得先做
一个"每步只采一次、两组共享"的缓存，不要只把 `enable_corruption` 打开。

## 辨识参数怎么进训练

`data/bpx_pro_training_params.yaml`，和 PACE 的 `CMAESOptimizer.update_simulator`
写的是同一组量，保证"辨识出来的数"和"训练用的数"走同一条路：

| 量 | 进仿真的方式 |
|---|---|
| armature | `write_joint_armature_to_sim` |
| 粘滞阻尼 | 关节 viscous friction（**不是** PD 的 kd） |
| 库伦摩擦 | 关节 static = dynamic friction |
| 编码器零偏 | `PaceDCMotor.update_encoder_bias` |
| 力矩延迟 | `PaceDCMotor.update_time_lags` |
| 整机质量 | 只改 torso，惯量同比例缩放 |

膝摩擦用的是**承重值 2.5 N·m**，不是悬空辨识的零负载 0.56–0.96。两种做法：

* **方案 A** 每回合抽常数，1.5–3.5 N·m
* **方案 B**（默认，`knee_friction.load_model.enabled: true`）每个控制步按负载更新
  `F = f0 + mu·L`，规则和 `mujoco_floor.set_load_friction` / `isaac_friction_check.LoadFriction` 一致

方案 B 的 `interval` 事件在物理步进**之后**触发，所以摩擦下一个控制步（20 ms）才生效；
负载估计本身是 τ=0.2 s 的低通，这点滞后可忽略。

## 架构：真 RMA，不是普通师生蒸馏

照 Kumar et al. (RSS 2021) 的结构，**不是**两张独立的网回归动作：

```
阶段一 (PPO)    z  = μ(特权参数 65)        a = π(本体感知 45, z)      π、μ 一起训
阶段二 (蒸馏)   ẑ = φ(25 帧历史 1125)      a = π(本体感知 45, ẑ)      π 从阶段一复制过来冻住，只训 φ
```

`z` 是 8 维瓶颈。**π 是同一张网** —— 所以 ẑ 和 z 活在同一个空间里，可以直接比。
这正是"用策略验证辨识结果"的证据所在：普通的动作蒸馏里 student 自己重学一张控制网，
内部表示和 teacher 没有对应关系，拿不出任何可以和辨识值对照的量。

冻结是在 `models.py:RmaDistillation.load` 里做的，日志会打印可训练参数占比
（典型 741,780 / 935,200 —— 差值正好是 π 的参数量）。

| 观测组 | 维度 | 去哪 |
|---|---|---|
| `policy` | 45 | base 策略 π（teacher 和 student 都是） |
| `privileged` | 65 | teacher 的编码器 μ |
| `history` | 1125 | student 的 adaptation module φ |

约定：观测集里**第一个组进 π，其余的组进编码器**，所以 teacher 和 student 能共用
`RmaModel` 这一个类。

`privileged` 装的是辨识产出的真值（armature / 粘滞 / 摩擦 / 零偏 / 延迟 / 质量 /
膝负载比值）+ 机身线速度 + 触地。

## 怎么拿到定量结论

`export_onnx.py` 除了部署用的 `policy.onnx`，还会导一个 `adaptation.onnx`
（输入历史 1125，输出 ẑ 8 维，**不要放板子上**）。做法：

1. 真机跑这个策略，板子本来就会记 `rec_*_policy50.bin`（里面有 obs00..obs44）
2. 离线用那串 obs 重建 25 帧历史 → `adaptation.onnx` → 真机上的 ẑ(t)
3. 仿真里同时有 ẑ 和 z（以及产生 z 的那 65 维真值），训一个小探针 ẑ → 参数值
4. 把探针用到真机的 ẑ 上，读出"策略认为机器人的参数是多少"，和辨识值比

辨识参数那组和厂家参数那组各做一遍，看哪组的 ẑ 更接近真机的。这是定量的，
比"策略能走"强得多。

## 已训好的四个策略

| 实验名 | 参数 | vx 上限 | ONNX |
|---|---|---|---|
| `bpx_pro_flat` | 辨识 | 1.5 | `logs/rsl_rl/bpx_pro_flat/*_student/exported/` |
| `bpx_pro_flat_baseline` | 厂家 | 1.5 | `logs/rsl_rl/bpx_pro_flat_baseline/*_student/exported/` |
| `bpx_pro_fast` | 辨识 | 3.0 | `logs/rsl_rl/bpx_pro_fast/*_student/exported/` |
| `bpx_pro_fast_baseline` | 厂家 | 3.0 | `logs/rsl_rl/bpx_pro_fast_baseline/*_student/exported/` |

四组**完全同预算**：teacher 1200 迭代、student 1500 迭代、4096 环境、奖励配置一字未改。

### 主要结论：辨识的价值随速度增长

MuJoCo corrected 模型里直线行走，厂家策略误差 / 辨识策略误差：

| 指令速度 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|
| 线速度误差比 | 1.32 | **1.07** | 1.14 | 1.27 | **1.40** |
| 角速度误差比 | 0.88 | 1.30 | **1.78** | 1.67 | 1.45 |

1.5 m/s 下两组基本打平（比值 1.07）—— adaptation module 把参数误差在线吸收掉了，这正是
RMA 的设计目的，但对"验证辨识"是坏消息。速度升到 3.0 时任务逼近电机极限，
惯量 / 阻尼在力矩平衡里的占比大幅上升（膝惯量 0.9% @0.5m/s → 9.4% @1.5m/s），
在线适应补不满，辨识的价值才显出来。

**转向段在低速下也能分辨**（1.5 m/s 的 mixed 预设里差 3.7 倍 / 6.7 倍）。

**保留意见**：评估用的 corrected 模型就是辨识结果本身，辨识组占主场。只做了 A/C
没做 2×2，主场优势占多少未测。决定性证据仍然是真机。

## 对照实验

换一份参数 yaml 就能跑"厂家参数 vs 辨识参数"：

```bash
BPX_PRO_PARAMS_YAML=data/bpx_pro_baseline_params.yaml \
    python scripts/train.py --headless --run-name teacher_baseline
```

每次训练会把当次用的参数存到 `logs/.../params/identification.yaml`，事后能查是哪一版。

## 上真机前

`export_onnx.py` 产出的 `deploy.yaml` 里 `hardware_joint_convention_confirmed` 固定是
**false**。那是板子的安全闸门（门 2）：必须人工在真机上确认关节正负方向、零位、顺序和
训练一致之后才能改 true。脚本不会替你确认。

## 目录

```
bpx_rma/spec.py             部署契约 + 机器人规格（纯 Python，不依赖 isaaclab）
bpx_rma/params.py           读辨识参数 yaml
bpx_rma/actuator.py         PaceDCMotor（从 pace-sim2real 搬来，Apache-2.0 © ETH Zurich）
bpx_rma/robot_cfg.py        ArticulationCfg
bpx_rma/models.py           RmaModel（μ/φ 编码器 + 共享 base）、RmaDistillation（冻结 base）
bpx_rma/mdp/state.py        逐 env 的域随机化状态
bpx_rma/mdp/events.py       参数/质量随机化 + 方案 B 膝摩擦
bpx_rma/mdp/observations.py 特权观测
bpx_rma/velocity_env_cfg.py 环境配置
bpx_rma/agents/rsl_rl_cfg.py PPO(teacher) + Distillation(student)
scripts/check_contract.py   契约核对（训练前后各跑一次）
scripts/train.py            两阶段训练
scripts/play.py             回放 + 指标
scripts/export_onnx.py      导出部署 ONNX
```
