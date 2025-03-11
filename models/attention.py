#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class AdaptiveWeighting(nn.Module):
    """自适应权重分配模块"""
    def __init__(self, d_model: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 2),  # 输出AR和MA的权重
            nn.Softmax(dim=-1)  # 确保权重和为1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算自适应权重
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            权重 [batch_size, seq_len, 2]
        """
        # 全局上下文信息
        global_context = torch.mean(x, dim=1, keepdim=True)
        # 计算权重
        weights = self.weight_net(global_context)
        return weights

class ARMAAttention(nn.Module):
    """
    增强的ARMA (AutoRegressive Moving Average) Attention Implementation
    
    结构组成:
    - AR分支：使用门控因果空洞卷积获取长期依赖
    - MA分支：使用动态深度可分离卷积处理局部模式
    - 自适应权重：动态调整AR和MA分支的权重
    - 注意力正则化：防止过拟合
    输出: Output = α(x) * AR(x) + (1-α(x)) * MA(x) + x
    """
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int = 8,
                 layer_depth: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        
        # AR分支参数
        self.ar_kernel_size = 3
        self.dilation = 2 ** layer_depth  # 空洞率随层深指数增长
        
        # MA分支参数
        self.ma_kernel_size = 5
        
        # AR门控因果空洞卷积
        self.ar_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=2 * d_model,  # 双倍通道用于门控机制
            kernel_size=self.ar_kernel_size,
            padding=0,  # 移除默认填充
            dilation=self.dilation,
            groups=n_heads
        )
        
        # 增强AR分支
        self.ar_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # MA动态卷积参数生成器
        self.ma_weight_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.ma_kernel_size * d_model)
        )
        
        # 增强MA分支
        self.ma_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 自适应权重分配
        self.adaptive_weighting = AdaptiveWeighting(d_model)
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 注意力正则化参数
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_scale = nn.Parameter(torch.tensor(1.0))
    
    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """确保因果性的填充"""
        pad_size = (self.ar_kernel_size - 1) * self.dilation
        return F.pad(x, (pad_size, 0))
    
    def _ar_branch(self, x: torch.Tensor) -> torch.Tensor:
        """增强的AR分支：门控因果空洞卷积"""
        # [batch, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # 应用因果填充
        x = self._causal_padding(x)
        
        # 门控卷积
        conv_out = self.ar_conv(x)
        gates, values = conv_out.chunk(2, dim=1)
        gates = torch.sigmoid(gates)
        
        # 门控机制
        ar_out = gates * values
        
        # 确保输出序列长度正确
        orig_len = x.size(-1) - (self.ar_kernel_size - 1) * self.dilation
        ar_out = ar_out[:, :, :orig_len]
        
        # 转回 [batch, seq_len, d_model]
        ar_out = ar_out.transpose(1, 2)
        
        # 应用增强
        ar_out = self.ar_enhancement(ar_out)
        
        return ar_out
    
    def _ma_branch(self, x: torch.Tensor) -> torch.Tensor:
        """增强的MA分支：动态深度可分离卷积"""
        B, L, D = x.shape
        
        # 生成动态卷积权重
        dynamic_weights = self.ma_weight_generator(x)
        dynamic_weights = dynamic_weights.view(
            B, L, self.ma_kernel_size, D).permute(0, 3, 2, 1)
        
        # 准备输入
        x_unf = F.unfold(
            x.transpose(1, 2).unsqueeze(-1),
            kernel_size=(self.ma_kernel_size, 1),
            padding=(self.ma_kernel_size // 2, 0)
        )
        x_unf = x_unf.view(B, D, self.ma_kernel_size, L)
        
        # 动态卷积
        out = (dynamic_weights * x_unf).sum(dim=2)
        
        # 转回 [batch, seq_len, d_model]
        out = out.transpose(1, 2)
        
        # 应用增强
        out = self.ma_enhancement(out)
        
        return out
    
    def _apply_attention_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """应用注意力正则化"""
        # 缩放输入
        x = x * self.attention_scale
        
        # 应用dropout
        x = self.attention_dropout(x)
        
        return x
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 保存残差连接
        residual = x
        
        # Layer Norm
        x = self.layer_norm(x)
        
        # AR分支
        ar_out = self._ar_branch(x)
        
        # MA分支
        ma_out = self._ma_branch(x)
        
        # 计算自适应权重
        weights = self.adaptive_weighting(x)
        ar_weight = weights[:, :, 0].unsqueeze(-1)
        ma_weight = weights[:, :, 1].unsqueeze(-1)
        
        # 合并AR和MA输出
        output = ar_weight * ar_out + ma_weight * ma_out
        
        # 应用注意力正则化
        output = self._apply_attention_regularization(output)
        
        # 输出投影
        output = self.output_projection(output)
        
        # 残差连接
        output = output + residual
        
        # 应用mask（如果提供）
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        
        return output
    
    def train(self, mode: bool = True):
        """切换训练/推理模式"""
        super().train(mode)
        return self

# # 创建ARMA注意力层
# arma = ARMAAttention(
#     d_model=256,      # 模型维度
#     n_heads=8,        # 注意力头数
#     layer_depth=2,    # 层深度（影响空洞率）
#     dropout=0.1       # dropout率
# )

# # 输入张量 [batch_size, seq_len, d_model]
# x = torch.randn(32, 100, 256)
# mask = None  # 可选的掩码

# # 前向传播
# output = arma(x, mask)  # 输出维度与输入相同
