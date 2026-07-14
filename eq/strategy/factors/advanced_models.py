"""机构级量化交易前沿模型与优化器。

论文来源：
- DeepLOB: "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books" (Zhang et al., 2019)
- TFT: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (Lim et al., 2019)
- SAM: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (Foret et al., 2021)
- Lookahead: "Lookahead Optimizer: k steps forward, 1 step back" (Zhang et al., 2019)
- Lion: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

包含：
1. DeepLOB — CNN + BiLSTM + Attention（高频订单簿微观结构建模）
2. Temporal Fusion Transformer（中低频多时间跨度预测）
3. SAM / Lookahead / Lion 优化器
4. 可微夏普比率损失函数
5. 特征正交化（去 Beta）
6. 对抗训练（FGSM）
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 第 1 部分：DeepLOB — CNN + BiLSTM + Attention
# =============================================================================
# 论文: Zhang, Zihao, Stefan Zohren, and Stephen Roberts.
#       "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"
#       IEEE TNNLS 2021 / JMLR 2019
#
# 架构: Input(100×40) → Conv3×2(16,16,16) → BiLSTM(64) → Attention → FC(1)
# 输入: 100 步 × 40 维（买卖 10 档价量 * 2 侧 = 40）
# 对日线选用：Input(60×26) 或 Input(120×26) 适配 Alpha158 特征重塑

class DeepLOB(nn.Module):
    """DeepLOB: CNN + BiLSTM + Attention 用于金融时序预测。

    输入形状: (batch, seq_len, input_dim)
    默认 seq_len=120, input_dim=26（适配 Alpha158 的 6×26 时序重塑 x 20 天窗口）

    Args:
        seq_len: 输入时间步（默认 120，对应 120 个交易日 ≈ 半年）
        input_dim: 每步特征维度（默认 26）
        conv_filters: 卷积层 filter 数列表 [16, 16, 16]
        lstm_hidden: BiLSTM 隐藏层维度（默认 64）
        dropout: Dropout 率（默认 0.3，金融数据建议 0.3-0.5）
        use_attention: 是否使用注意力机制（默认 True）
    """
    def __init__(
        self,
        seq_len: int = 120,
        input_dim: int = 26,
        conv_filters: list[int] | None = None,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
        raw_input_dim: int = 0,  # 原始输入维度，>0 时自动添加投影层
    ):
        super().__init__()
        if conv_filters is None:
            conv_filters = [16, 16, 16]

        # 输入投影层（处理非标准输入，如 Alpha158 的 158 维）
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.proj = None
        if raw_input_dim > 0:
            target = seq_len * input_dim
            self.proj = nn.Sequential(
                nn.Linear(raw_input_dim, target),
                nn.LayerNorm(target),
                nn.ReLU(),
            )
            self.raw_input_dim = raw_input_dim
        else:
            self.raw_input_dim = 0

        # CNN 模块：1×2 卷积核（同档位买卖价差捕捉）
        # 输入: (B, 1, seq_len, input_dim)  →  Conv2d 处理
        in_channels = 1
        conv_layers = []
        for i, out_c in enumerate(conv_filters):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_c, kernel_size=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ])
            in_channels = out_c
        self.conv_block = nn.Sequential(*conv_layers)
        # 经过 Conv2d(1,2) 后 input_dim → input_dim+1（padding=1）
        # 多层后维度变化，取最后 conv_filters[-1] 个通道
        conv_out_dim = input_dim + len(conv_filters)  # 每层 (1,2) padding 加 1 列

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=conv_filters[-1] * conv_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            attn_dim = lstm_hidden * 2  # BiLSTM 双向
            self.attention = nn.Sequential(
                nn.Linear(attn_dim, attn_dim // 2),
                nn.Tanh(),
                nn.Linear(attn_dim // 2, 1),
            )

        # 输出层
        lstm_out_dim = lstm_hidden * 2  # BiLSTM
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, input_dim) 或 (batch, raw_input_dim)
        Returns:
            (batch,) 预测分数
        """
        # 如果有投影层，先用投影层映射到目标维度
        if self.proj is not None and x.dim() == 2:
            x = self.proj(x)  # (batch, seq_len * input_dim)
            batch = x.size(0)
            x = x.view(batch, self.seq_len, -1)

        batch, seq_len, input_dim = x.shape

        # CNN: (B, 1, seq_len, input_dim)
        x = x.unsqueeze(1)
        x = self.conv_block(x)  # (B, C, seq_len, conv_out_dim)
        # 重塑为 (B, seq_len, C * conv_out_dim) 喂 LSTM
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (B, seq_len, lstm_hidden * 2)
        lstm_out = self.lstm_dropout(lstm_out)

        if self.use_attention:
            # 注意力加权: (B, seq_len, attn_dim) → (B, seq_len, 1)
            attn_weights = self.attention(lstm_out)  # (B, seq_len, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            # 加权和: (B, seq_len, attn_dim) * (B, seq_len, 1) → (B, attn_dim)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            # 取最后一步
            context = lstm_out[:, -1, :]

        return self.head(context).squeeze(-1)


# =============================================================================
# 第 2 部分：Temporal Fusion Transformer (TFT)
# =============================================================================
# 论文: Lim, Bryan, et al.
#       "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
#       International Journal of Forecasting, 2021
#
# 核心组件：
# - GRN (Gated Residual Network)
# - Variable Selection Network (VSN)
# - Multi-Head Attention
# - 可解释性输出

class GRN(nn.Module):
    """门控残差网络 (Gated Residual Network)。

    GRN(x) = LayerNorm(x + GLU(ELU(W2 * ReLU(W1 * x + b1) + b2)))
    """
    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, input_dim)
        self.glu = nn.Linear(input_dim, input_dim * 2)
        self.ln = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = F.elu(x)
        x = self.w2(x)
        x = self.dropout(x)
        # GLU 门控
        gate = self.glu(residual)
        gate_a, gate_b = gate.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate_a) + residual  # 残差连接
        return self.ln(x)


class InterpretableMultiHeadAttention(nn.Module):
    """可解释多头注意力（TFT 使用共享 value 矩阵实现可解释性）。"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = query.shape
        Q = self.q_linear(query).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_linear(key).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(value).view(B, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, L, -1)
        return self.out_linear(out)


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer — 多时间跨度预测。

    论文超参数：
    - Hidden Size: 128-256（特征 100+ 时取大值）
    - Attention Heads: 4
    - Dropout: 0.3-0.4（金融数据信噪比极低，强正则化）
    - Max Encoder Steps: 60-252（对应 3 个月 ~ 1 年交易日）

    Args:
        input_dim: 输入特征维度（默认 158，Alpha158 全量）
        hidden_dim: 隐藏层维度（默认 256，TFT 推荐）
        num_heads: 注意力头数（默认 4）
        dropout: Dropout 率（默认 0.3，金融建议 0.3-0.4）
        max_seq_len: 最大序列长度（默认 252，一年交易日）
        num_layers: LSTM 编码器层数（默认 2）
    """
    def __init__(
        self,
        input_dim: int = 158,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        max_seq_len: int = 252,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM 编码器（捕捉时序依赖）
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )

        # 门控残差网络（特征选择 + 非线性变换）
        self.grn = GRN(hidden_dim, hidden_dim, dropout=dropout)

        # 多头注意力
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout,
        )

        # 位置编码（可学习）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.1)

        # 输出层
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (batch, seq_len, input_dim) 或 (batch, seq_len * input_dim)
        Returns:
            (batch,) 预测分数
        """
        # 如果输入是展平的，尝试重塑
        if x.dim() == 2:
            batch = x.size(0)
            # 尝试自动推断 seq_len
            total = x.size(1)
            if total % self.input_dim == 0:
                seq_len = total // self.input_dim
                x = x.view(batch, seq_len, self.input_dim)
            else:
                # 取前 max_seq_len * input_dim 个元素
                n_elements = self.max_seq_len * self.input_dim
                if total >= n_elements:
                    x = x[:, :n_elements].view(batch, self.max_seq_len, self.input_dim)
                else:
                    # 填充
                    pad = torch.zeros(batch, n_elements - total, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=1).view(batch, self.max_seq_len, self.input_dim)

        batch, seq_len, _ = x.shape

        # 输入投影
        x = self.input_proj(x)  # (B, S, H)

        # 位置编码
        x = x + self.pos_encoding[:, :seq_len, :]

        # LSTM 编码
        lstm_out, _ = self.encoder_lstm(x)  # (B, S, H)

        # GRN 门控残差
        grn_out = self.grn(lstm_out)  # (B, S, H)

        # 多头注意力（自注意力）
        attn_out = self.attention(grn_out, grn_out, grn_out)  # (B, S, H)

        # 残差连接 + LayerNorm
        out = self.output_norm(grn_out + attn_out)

        # 取最后一步输出
        last_out = out[:, -1, :]

        return self.output_proj(last_out).squeeze(-1)


# =============================================================================
# 第 3 部分：高级优化器
# =============================================================================

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (SAM)。

    论文: Foret et al., "Sharpness-Aware Minimization for Efficiently
          Improving Generalization", ICLR 2021

    SAM 不直接找损失最低点，而是找周围平坦的区域（Flat Minima），
    对金融数据的概念漂移（Concept Drift）有天然抵抗力。

    用法:
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        optimizer = SAM(model.parameters(), base_opt, rho=0.05)
        for ...:
            loss = criterion(model(x), y)
            optimizer.first_step(zero_grad=True)
            loss = criterion(model(x), y)
            optimizer.second_step(zero_grad=True)

    超参数:
        rho: 邻域半径（默认 0.05，太大不收敛，太小失去平滑作用）
        base_optimizer: 内部优化器（推荐 AdamW 或 SGD）
    """
    def __init__(self, params, base_optimizer: torch.optim.Optimizer, rho: float = 0.05, **kwargs):
        assert rho >= 0.0, f"rho 必须 >= 0, got {rho}"
        # 不调 super().__init__，直接用 base_optimizer 的 param_groups
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        self.state = {}  # SAM 独立 state，存储 e_w（不共享 base_optimizer.state）
        self._rho = rho

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """第一步：计算对抗性扰动，更新权重为 w + ρ * grad/||grad||。"""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", self._rho) / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # 临时上移
                self.state[p] = e_w  # 用参数 id 作为 key
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """第二步：在扰动后的位置取梯度，回退到原始位置，用 base_optimizer 更新。"""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state:
                    continue
                p.sub_(self.state[p])  # 回退
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        """计算梯度范数。"""
        norm = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norm += p.grad.norm().item() ** 2
        return norm ** 0.5

    def step(self, closure=None):
        """默认的 step 执行标准优化器更新（与 SAM 两步模式互斥）。"""
        self.base_optimizer.step(closure)


class Lookahead(torch.optim.Optimizer):
    """Lookahead 元优化器 — k 步前看，1 步后收。

    论文: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back"
          NeurIPS 2019

    维护两组权重：快速权重（探索）和慢速权重（稳定）。
    每 k 步将慢速权重向快速权重插值，再将快速权重重置回慢速权重。

    超参数:
        k: 同步周期（默认 5，快速权重更新 5 次后同步）
        alpha: 慢速步长（默认 0.5，慢速权重的学习率）
        base_optimizer: 内部优化器（推荐 AdamW）
    """
    def __init__(self, base_optimizer: torch.optim.Optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state

        self.k = k
        self.alpha = alpha
        self._step_count = 0
        # 备份慢速权重
        self._slow_weights = {}
        self._backup_every_k()

    def _backup_every_k(self):
        """备份当前权重作为慢速权重。"""
        for group in self.param_groups:
            for p in group["params"]:
                self._slow_weights[p] = p.data.clone()

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        self._step_count += 1

        if self._step_count % self.k == 0:
            # 慢速权重向快速权重插值
            for group in self.param_groups:
                for p in group["params"]:
                    if p in self._slow_weights:
                        # slow = slow + alpha * (fast - slow)
                        p.data = self._slow_weights[p] + self.alpha * (p.data - self._slow_weights[p])
                        self._slow_weights[p] = p.data.clone()
        return loss


class Lion(torch.optim.Optimizer):
    """Lion 优化器 (EvoLved Sign Momentum) — Google Brain 通过进化搜索发现。

    论文: Chen et al., "Symbolic Discovery of Optimization Algorithms"
          NeurIPS 2023

    只保留梯度的符号（Sign）进行更新，天然免疫极端异常值。
    在量化数据中，闪崩等极端行情会产生巨大梯度，Lion 只看方向不看幅度。

    超参数:
        lr: 学习率（通常比 Adam 小 3-10 倍，建议 3e-5 起步）
        betas: (β1, β2) 动量衰减（默认 0.9, 0.99）
        weight_decay: 权重衰减（默认 0.01，量化建议 0.01-0.1）

    更新规则:
        update = sign(β1 * m + (1-β1) * g)
        w = w * (1 - lr * λ) - lr * update
        m = β2 * m + (1-β2) * g
    """
    def __init__(
        self,
        params,
        lr: float = 3e-5,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)

                m = state["momentum"]
                # 更新动量
                m.lerp_(g, 1 - beta1)  # m = β1 * m + (1-β1) * g
                # 符号更新
                update = m.sign()
                # 权重衰减
                if wd > 0:
                    p.mul_(1 - lr * wd)
                # 参数更新
                p.add_(update, alpha=-lr)
                # 更新二阶动量（用于下一轮）
                # 注意：Lion 的"momentum"实际上是一阶动量以 β2 衰减
                # 这里用 g 的符号来更新，保持与传统实现一致
                m.lerp_(g, 1 - beta2)  # m = β2 * m + (1-β2) * g

        return loss


# =============================================================================
# 第 4 部分：可微夏普比率损失函数
# =============================================================================
# 绝对不要用 MSE 或 Cross-Entropy 优化量化模型。
# 你需要优化的是风险调整后的收益，而不是预测准确率。
#
# Loss = -E[R_t] / sqrt(Var[R_t] + epsilon)
# 其中 R_t 是模型预测排序后的投资组合收益率序列

class DifferentiableSharpeRatio(nn.Module):
    """可微夏普比率损失函数。

    直接优化组合的风险调整收益，而不是 MSE。

    Args:
        epsilon: 防止除零的小量（默认 1e-8）
        annual_factor: 年化因子（日线 = sqrt(252)，分钟线 = sqrt(252*240)）
    """
    def __init__(self, epsilon: float = 1e-8, annual_factor: float = math.sqrt(252)):
        super().__init__()
        self.epsilon = epsilon
        self.annual_factor = annual_factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算负夏普比率（取负值作为损失，最小化）。

        Args:
            pred: 模型预测分数 (batch,)
            target: 未来真实收益 (batch,)

        Returns:
            标量损失（负夏普比率）
        """
        # 将预测分数归一化为仓位权重（多头+空头）
        weights = pred - pred.mean()
        weights = weights / (weights.abs().sum() + self.epsilon)

        # 组合收益率
        portfolio_returns = weights * target

        # 期望收益
        mean_return = portfolio_returns.mean()

        # 收益方差
        variance = portfolio_returns.var() + self.epsilon

        # 负夏普比率（最小化）
        sharpe = mean_return / (variance.sqrt() + self.epsilon)
        return -sharpe * self.annual_factor


def sharpe_ratio_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """函数式可微夏普比率损失（简洁接口）。"""
    weights = pred - pred.mean()
    weights = weights / (weights.abs().sum() + 1e-8)
    port_ret = weights * target
    return -port_ret.mean() / (port_ret.var() + 1e-8).sqrt()


# =============================================================================
# 第 5 部分：特征正交化（去 Beta / 去市场因子）
# =============================================================================
# 深度模型容易直接拟合市场大盘趋势（Market Beta）。
# 正交化确保模型学习的是纯粹的 Alpha，而不是 Beta 波动。

def feature_orthogonalize(
    features: pd.DataFrame | np.ndarray,
    market_factor: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """对特征进行正交化，去除市场因子（Beta）的影响。

    对每个特征列：
        f_alpha = f - beta * market  (beta = cov(f, market) / var(market))

    Args:
        features: 特征矩阵 (n_samples, n_features)
        market_factor: 市场因子序列 (n_samples,)，默认用等权平均

    Returns:
        正交化后的特征矩阵 (n_samples, n_features)
    """
    if isinstance(features, pd.DataFrame):
        cols = features.columns
        idx = features.index
        arr = features.values.astype(np.float64)
    else:
        arr = np.asarray(features, dtype=np.float64)
        cols = None
        idx = None

    if market_factor is None:
        market_factor = np.nanmean(arr, axis=1)  # 等权平均作为市场因子
    else:
        market_factor = np.asarray(market_factor, dtype=np.float64)

    # 去均值
    market_dm = market_factor - market_factor.mean()
    market_var = np.var(market_dm) + 1e-12

    # 逐列正交化
    result = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        f = arr[:, j]
        if not np.all(np.isfinite(f)):
            result[:, j] = 0.0
            continue
        beta = np.cov(f, market_factor, ddof=0)[0, 1] / market_var
        result[:, j] = f - beta * market_dm

    if cols is not None:
        return pd.DataFrame(result, columns=cols, index=idx)
    return result


def feature_orthogonalize_tensor(
    x: torch.Tensor,
    market_factor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tensor 版特征正交化（可微分，可用于训练中）。"""
    if market_factor is None:
        market_factor = x.mean(dim=1, keepdim=True)

    mf = market_factor - market_factor.mean(dim=0, keepdim=True)
    mf_var = mf.var(dim=0, unbiased=False) + 1e-12

    x_centered = x - x.mean(dim=0, keepdim=True)
    beta = (x_centered * mf).mean(dim=0, keepdim=True) / mf_var

    return x - beta * mf


# =============================================================================
# 第 6 部分：对抗训练（FGSM）
# =============================================================================
# 金融市场充满噪音、假突破和恶意骗线。FGSM 对抗训练强制模型忽略微小价格波动。

def adversarial_fgsm_perturb(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    epsilon: float = 0.01,
    clamp_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """FGSM (Fast Gradient Sign Method) 对抗性扰动。

    生成对抗样本:
        x_adv = x + epsilon * sign(grad_x(loss(model(x), y)))

    Args:
        model: 模型
        x: 输入 (batch, ...)
        y: 标签 (batch,)
        loss_fn: 损失函数
        epsilon: 扰动幅度（默认 0.01，金融数据建议 0.001-0.05）
        clamp_range: 裁剪范围 [min, max]

    Returns:
        对抗样本 x_adv
    """
    x_adv = x.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = loss_fn(pred, y)
    model.zero_grad()
    loss.backward()

    # FGSM 扰动
    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + epsilon * grad_sign

    if clamp_range is not None:
        x_adv = torch.clamp(x_adv, clamp_range[0], clamp_range[1])

    return x_adv.detach()


def adversarial_train_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    epsilon: float = 0.01,
    alpha: float = 0.5,
) -> torch.Tensor:
    """对抗训练一步（标准训练 + FGSM 对抗样本混合训练）。

    Loss = α * L(model(x), y) + (1-α) * L(model(x_adv), y)

    Args:
        model: 模型
        x: 原始输入
        y: 标签
        loss_fn: 损失函数（如 SharpeRatioLoss 或 MSELoss）
        optimizer: 优化器
        epsilon: FGSM 扰动幅度
        alpha: 原始损失权重（默认 0.5，50% 原始 + 50% 对抗）

    Returns:
        总损失值
    """
    model.train()
    optimizer.zero_grad()

    # 原始损失
    pred = model(x)
    loss_clean = loss_fn(pred, y)

    # 生成对抗样本
    x_adv = adversarial_fgsm_perturb(model, x, y, loss_fn, epsilon=epsilon)

    # 对抗损失
    pred_adv = model(x_adv)
    loss_adv = loss_fn(pred_adv, y)

    # 混合损失
    loss = alpha * loss_clean + (1 - alpha) * loss_adv
    loss.backward()
    optimizer.step()

    return loss.detach()


# =============================================================================
# 第 7 部分：高级训练器（集成所有技术）
# =============================================================================

class AdvancedTrainer:
    """高级训练器 — 整合 DeepLOB/TFT + 高级优化器 + 可微夏普比率 + 对抗训练。

    用法:
        trainer = AdvancedTrainer(
            model=model,
            optimizer_type="adamw",  # "adamw" | "sam" | "lookahead" | "lion"
            loss_type="sharpe",      # "sharpe" | "mse" | "ic"
            use_adversarial=True,
            orthogonalize=True,
            device="cuda",
        )
        result = trainer.fit(x_train, y_train, x_valid, y_valid)
    """

    OPTIMIZER_MAP = {
        "adamw": ("AdamW", {"lr": 1e-4, "weight_decay": 0.01}),
        "sam": ("SAM", {"lr": 1e-4, "rho": 0.05}),
        "lookahead": ("Lookahead", {"lr": 1e-4, "k": 5, "alpha": 0.5}),
        "lion": ("Lion", {"lr": 3e-5, "weight_decay": 0.01}),
    }

    def __init__(
        self,
        model: nn.Module,
        optimizer_type: str = "adamw",
        optimizer_kwargs: dict[str, Any] | None = None,
        loss_type: str = "sharpe",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_steps: int = 300,
        batch_size: int = 1024,
        early_stop: int = 30,
        use_adversarial: bool = False,
        adversarial_epsilon: float = 0.01,
        adversarial_alpha: float = 0.5,
        orthogonalize: bool = False,
        use_scheduler: bool = True,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.use_adversarial = use_adversarial
        self.adversarial_epsilon = adversarial_epsilon
        self.adversarial_alpha = adversarial_alpha
        self.orthogonalize = orthogonalize
        self.use_scheduler = use_scheduler
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        self.model.to(self.device)

        # 损失函数
        if loss_type == "sharpe":
            self.loss_fn = DifferentiableSharpeRatio()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "ic":
            self.loss_fn = self._ic_loss
        else:
            raise ValueError(f"未知损失类型: {loss_type}，可选 sharpe/mse/ic")
        self.loss_type = loss_type

        # 优化器
        self.optimizer = self._build_optimizer(optimizer_kwargs)

        # 学习率调度器
        self.scheduler = None
        if use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self._get_base_optimizer(), mode="max", factor=0.5,
                patience=8, min_lr=1e-7,
            )

        # 训练状态
        self.best_score = -float("inf")
        self.best_state = None
        self.best_step = 0

    def _get_base_optimizer(self):
        """获取基础优化器（SAM/Lookahead 内部包装的优化器）。"""
        if hasattr(self.optimizer, "base_optimizer"):
            return self.optimizer.base_optimizer
        return self.optimizer

    def _build_optimizer(self, kwargs_override: dict | None = None) -> torch.optim.Optimizer:
        kwargs = {"lr": self.learning_rate, "weight_decay": self.weight_decay}
        if kwargs_override:
            kwargs.update(kwargs_override)

        base_opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=kwargs.pop("lr", self.learning_rate),
            weight_decay=kwargs.pop("weight_decay", self.weight_decay),
        )

        if self.optimizer_type == "adamw":
            return base_opt
        elif self.optimizer_type == "sam":
            rho = kwargs.pop("rho", 0.05)
            return SAM(self.model.parameters(), base_opt, rho=rho)
        elif self.optimizer_type == "lookahead":
            k = kwargs.pop("k", 5)
            alpha = kwargs.pop("alpha", 0.5)
            return Lookahead(base_opt, k=k, alpha=alpha)
        elif self.optimizer_type == "lion":
            return Lion(
                self.model.parameters(),
                lr=kwargs.get("lr", 3e-5),
                weight_decay=kwargs.get("weight_decay", 0.01),
            )
        else:
            raise ValueError(f"未知优化器: {self.optimizer_type}，可选: adamw/sam/lookahead/lion")

    def _ic_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """IC 损失（负 IC，最大化 IC）。"""
        if pred.std() < 1e-8 or target.std() < 1e-8:
            return torch.tensor(1.0, device=pred.device)
        cov = torch.cov(torch.stack([pred, target]))[0, 1]
        ic = cov / (pred.std() * target.std() + 1e-8)
        return -ic  # 最小化负 IC = 最大化 IC

    def _to_tensor(self, data) -> torch.Tensor:
        if hasattr(data, "values"):
            return torch.from_numpy(data.values.astype(np.float32))
        return torch.from_numpy(np.asarray(data, dtype=np.float32))

    def fit(
        self,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_train_orth: pd.DataFrame | None = None,
        x_valid_orth: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """训练模型。

        Args:
            x_train: 训练特征
            y_train: 训练标签
            x_valid: 验证特征
            y_valid: 验证标签
            x_train_orth: 用于正交化的训练市场因子（可选）
            x_valid_orth: 用于正交化的验证市场因子（可选）

        Returns:
            {"best_ic": float, "best_step": int, "model": self.model}
        """
        xt = self._to_tensor(x_train).to(self.device)
        yt = self._to_tensor(y_train).squeeze(-1).to(self.device)
        xv = self._to_tensor(x_valid).to(self.device)
        yv = self._to_tensor(y_valid).squeeze(-1).to(self.device)

        # 可选：输入展开 → 重塑为 (batch, seq_len, input_dim)
        # 模型内部会处理展平输入

        stop = 0
        n_samples = len(xt)

        for step in range(self.max_steps):
            self.model.train()
            idx = torch.randperm(n_samples, device=self.device)
            epoch_losses = []

            for i in range(0, n_samples, self.batch_size):
                b = idx[i:i + self.batch_size]
                if len(b) < 2:
                    continue

                xb = xt[b]
                yb = yt[b]

                # 可选：特征正交化（训练中在线做）
                if self.orthogonalize:
                    xb = feature_orthogonalize_tensor(xb)

                if self.use_adversarial:
                    # 对抗训练一步
                    loss = adversarial_train_step(
                        self.model, xb, yb, self.loss_fn,
                        self.optimizer,
                        epsilon=self.adversarial_epsilon,
                        alpha=self.adversarial_alpha,
                    )
                elif self.optimizer_type == "sam":
                    # SAM 两步
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    pred2 = self.model(xb)
                    self.loss_fn(pred2, yb).backward()
                    self.optimizer.second_step(zero_grad=True)
                else:
                    # 标准训练
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                epoch_losses.append(loss.item())

            # 验证
            self.model.eval()
            with torch.no_grad():
                xv_input = xv
                if self.orthogonalize:
                    xv_input = feature_orthogonalize_tensor(xv)
                vp = self.model(xv_input)
                if len(vp) >= 2 and vp.std().item() > 0 and yv.std().item() > 0:
                    cov = torch.cov(torch.stack([vp, yv]))[0, 1]
                    score = cov / (vp.std().item() * yv.std().item())
                else:
                    score = -float("inf")

            # 调度器
            if self.scheduler is not None and score != -float("inf"):
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()

            # 进度
            mem_mb = torch.cuda.memory_allocated() / 1e6 if self.device.type == "cuda" else 0.0
            cur_lr = self._get_base_optimizer().param_groups[0]["lr"]
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

            if step % 5 == 0 or score > self.best_score:
                best_mark = "✓" if score > self.best_score else " "
                if self.verbose:
                    opt_name = self.optimizer_type.upper()
                    loss_name = self.loss_type.upper()
                    print(f"  [{opt_name}+{loss_name} step {step:3d}] "
                          f"loss={avg_loss:.6f} IC={score:+.4f} {best_mark} "
                          f"best={self.best_score:+.4f}@{self.best_step} "
                          f"lr={cur_lr:.2e} mem={mem_mb:.0f}MB", flush=True)

            if score > self.best_score:
                self.best_score = score
                self.best_step = step
                self.best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                stop = 0
            else:
                stop += 1
                if stop >= self.early_stop:
                    break

        # 恢复最佳权重
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        if self.verbose:
            print(f"  [训练完成] best IC={self.best_score:+.4f} @step {self.best_step} "
                  f"(early_stop={stop}/{self.early_stop})", flush=True)

        return {
            "best_ic": self.best_score,
            "best_step": self.best_step,
            "model": self.model,
        }

    def predict(self, x) -> np.ndarray:
        """预测。"""
        self.model.eval()
        with torch.no_grad():
            xt = self._to_tensor(x).to(self.device)
            if self.orthogonalize:
                xt = feature_orthogonalize_tensor(xt)
            pred = self.model(xt)
            return pred.cpu().numpy()