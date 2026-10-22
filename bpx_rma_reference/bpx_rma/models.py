"""RMA 的网络结构：编码器 + 共享 base 策略。

和 rsl_rl 自带的师生蒸馏（两张互不相干的网、回归动作）不同，这里照 Kumar et al.
(RSS 2021) 的结构来，**因为我们要的那个证据藏在隐变量里**：

    阶段一（PPO）     z  = μ(特权参数)          a = π(本体感知, z)
    阶段二（蒸馏）     ẑ = φ(25 帧历史)          a = π(本体感知, ẑ)   ← π 从阶段一复制过来并冻住

关键点：**π 是同一张网**，阶段二只训 φ。所以 ẑ 和 z 活在同一个空间里，可以直接比。
如果辨识是对的，真机上 φ 估出来的 ẑ 应该和"把辨识参数喂给 μ"得到的 z 落在一起 ——
这是定量结论，比"策略能走"强得多。

阶段二的损失仍然是动作 MSE（rsl_rl 的 `Distillation` 就是这么算的），但因为 π 冻住了，
梯度只能从动作误差穿过 π 回流到 φ，等价于在逼近 z。这是 RMA 原文"直接回归 z"的常见变体，
好处是不用改 rsl_rl 的算法主循环。

**观测组的约定**：观测集里的**第一个组进 base 策略，其余的组进编码器**。于是同一个类
两边复用：

    teacher: ["policy", "privileged"]  -> base 吃 policy(45)，μ 吃 privileged(65)
    student: ["policy", "history"]     -> base 吃 policy(45)，φ 吃 history(1125)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState


class RmaModel(MLPModel):
    """μ / φ 编码器 + 共享 base 策略。teacher 和 student 用的是同一个类。"""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        latent_dim: int = 8,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        encoder_activation: str = "elu",
        **kwargs: Any,
    ) -> None:
        groups = obs_groups[obs_set]
        if len(groups) < 2:
            raise ValueError(
                f"观测集 '{obs_set}' 只有 {groups}；RMA 需要至少两个组"
                "（第一个进 base 策略，其余进编码器）"
            )
        # 这几个要在 super().__init__ 之前定好 —— 基类建 self.mlp 时会调 _get_latent_dim()。
        # nn.Module 只拦截 Parameter / Module 的提前赋值，普通 int / list 没问题。
        self._base_group = groups[0]
        self._enc_groups = list(groups[1:])
        self._base_dim = int(obs[self._base_group].shape[-1])
        self._enc_dim = int(sum(obs[g].shape[-1] for g in self._enc_groups))
        self._latent_dim = int(latent_dim)

        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

        self.encoder = MLP(self._enc_dim, self._latent_dim, list(encoder_hidden_dims), encoder_activation)
        if self.obs_normalization:
            self.base_normalizer: nn.Module = EmpiricalNormalization(self._base_dim)
            self.enc_normalizer: nn.Module = EmpiricalNormalization(self._enc_dim)
        else:
            self.base_normalizer = nn.Identity()
            self.enc_normalizer = nn.Identity()
        # 基类那个覆盖整个观测集的归一化器用不上了，换成恒等免得被误用
        self.obs_normalizer = nn.Identity()
        #: base 冻住以后 base_normalizer 的统计量也不能再动（阶段二）
        self.base_frozen = False

        print(
            f"RmaModel[{obs_set}]：base {self._base_group}({self._base_dim}) + z({self._latent_dim})"
            f" <- 编码器 {self._enc_groups}({self._enc_dim})"
        )

    # -- 维度 ---------------------------------------------------------------
    def _get_latent_dim(self) -> int:
        """base 策略的输入 = 本体感知 + 隐变量。"""
        return self._base_dim + self._latent_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # -- 前向 ---------------------------------------------------------------
    def encode(self, obs: TensorDict) -> torch.Tensor:
        """μ(特权参数) 或 φ(历史) -> z，(N, latent_dim)。"""
        x = torch.cat([obs[g] for g in self._enc_groups], dim=-1)
        return self.encoder(self.enc_normalizer(x))

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        return torch.cat([self.base_normalizer(obs[self._base_group]), self.encode(obs)], dim=-1)

    # -- 归一化 -------------------------------------------------------------
    def update_normalization(self, obs: TensorDict) -> None:
        if not self.obs_normalization:
            return
        if not self.base_frozen:
            self.base_normalizer.update(obs[self._base_group])  # type: ignore[operator]
        self.enc_normalizer.update(torch.cat([obs[g] for g in self._enc_groups], dim=-1))  # type: ignore[operator]

    # -- 阶段二：冻住 base --------------------------------------------------
    def adopt_base_from(self, teacher: RmaModel) -> None:
        """把 teacher 的 base 策略（和它的归一化统计量）搬过来并冻住。

        这是 RMA 阶段二的定义性操作：π 不再变，只训 φ。
        """
        if self._base_dim != teacher._base_dim or self._latent_dim != teacher._latent_dim:
            raise ValueError(
                f"base/隐变量维度对不上：student({self._base_dim}, {self._latent_dim}) "
                f"vs teacher({teacher._base_dim}, {teacher._latent_dim})"
            )
        self.mlp.load_state_dict(teacher.mlp.state_dict())
        if self.obs_normalization:
            self.base_normalizer.load_state_dict(teacher.base_normalizer.state_dict())
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        self.base_normalizer.eval()
        self.base_frozen = True

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        if self.base_frozen:
            self.base_normalizer.eval()
        return self


class RmaDistillation(Distillation):
    """在 rsl_rl 的蒸馏上加一步：student 复用并冻住 teacher 的 base 策略。

    不加这一步就退化成普通的师生动作蒸馏 —— student 自己重新学一张控制网，
    它内部的 ẑ 和 teacher 的 z 没有任何对应关系，也就拿不出可以和辨识值对照的量。
    """

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resumed = super().load(loaded_dict, load_cfg, strict)
        if not self.teacher_loaded:
            return resumed
        if not isinstance(self.student, RmaModel) or not isinstance(self.teacher, RmaModel):
            raise TypeError("RmaDistillation 要求 student / teacher 都是 RmaModel")
        self.student.adopt_base_from(self.teacher)
        n_train = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.student.parameters())
        print(
            f"[RMA] student 已复用并冻住 teacher 的 base 策略："
            f"可训练 {n_train:,} / 共 {n_total:,} 参数（只剩 φ 和输出噪声）"
        )
        return resumed
