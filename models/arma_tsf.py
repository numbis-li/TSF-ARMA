#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# 获取项目根目录并添加到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple
from models.attention import ARMAAttention
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码到输入张量"""
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

class ARMATransformerLayer(nn.Module):
    """增强的Pre-LN结构的ARMA Transformer层"""
    def __init__(self, 
                d_model: int = 512,
                n_heads: int = 8,
                d_ff: int = 2048,
                layer_idx: int = 0,
                dropout: float = 0.1):
        super().__init__()
        # Pre-LN结构的Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # 额外的规范化层
        
        # ARMA注意力层
        self.attention = ARMAAttention(
            d_model=d_model,
            n_heads=n_heads,
            layer_depth=layer_idx,  # 使用层索引作为深度
            dropout=dropout
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 残差连接缩放因子
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, prev_layer_output: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            prev_layer_output: 上一层的输出 [batch_size, seq_len, d_model]
            mask: 可选的掩码张量
        Returns:
            输出张量 [batch_size, seq_len, d_model]
            注意力输出 [batch_size, seq_len, d_model]
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # Pre-LN结构：先规范化，再注意力
        norm_x = self.norm1(x)
        
        # 如果有上一层输出，融合进当前层
        if prev_layer_output is not None:
            # 确保prev_layer_output的形状与norm_x匹配
            if prev_layer_output.shape != norm_x.shape:
                # 如果形状不匹配，调整prev_layer_output的形状
                prev_layer_output = F.interpolate(
                    prev_layer_output.transpose(1, 2), 
                    size=norm_x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
                
                # 如果通道数不匹配，使用线性投影
                if prev_layer_output.size(2) != norm_x.size(2):
                    prev_layer_output = nn.Linear(
                        prev_layer_output.size(2), norm_x.size(2)
                    ).to(norm_x.device)(prev_layer_output)
            
            # 计算门控权重
            gate_weights = self.gate(torch.cat([norm_x, prev_layer_output], dim=-1))
            # 融合上一层输出
            norm_x = gate_weights * norm_x + (1 - gate_weights) * prev_layer_output
        
        # 应用注意力
        attn_out = self.attention(norm_x, mask)
        
        # 第一个残差连接
        x = residual + self.res_scale * self.dropout(attn_out)
        
        # 保存注意力输出用于返回
        attn_output = x
        
        # 第二个残差路径
        residual2 = x
        
        # Pre-LN结构：先规范化，再前馈
        ff_out = self.feed_forward(self.norm2(x))
        
        # 第二个残差连接
        x = residual2 + self.res_scale * self.dropout(ff_out)
        
        # 最终规范化和投影
        x = self.norm3(x)
        x = self.output_projection(x)
        
        return x, attn_output

class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, d_model: int, scales: List[int] = [1, 2, 4, 8]):
        super().__init__()
        self.d_model = d_model
        self.scales = scales
        
        # 为每个尺度创建卷积层 - 修复维度不匹配问题
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,  # 保持与输入相同的通道数
                kernel_size=scale,
                padding=(scale // 2),
                stride=1,
                groups=d_model // min(d_model, 16)  # 确保分组数是通道数的因子
            ) for scale in scales
        ])
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # 注意力权重
        self.attention_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度特征融合
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            融合后的特征 [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 转换为卷积格式 [batch_size, d_model, seq_len]
        x_conv = x.transpose(1, 2)
        
        # 应用不同尺度的卷积
        multi_scale_features = []
        for i, conv in enumerate(self.conv_layers):
            # 确保序列长度一致
            scale_feature = conv(x_conv)
            if scale_feature.size(2) != seq_len:
                # 使用插值调整长度
                scale_feature = F.interpolate(
                    scale_feature, 
                    size=seq_len, 
                    mode='linear', 
                    align_corners=False
                )
            multi_scale_features.append(scale_feature)
        
        # 获取注意力权重
        attn_weights = self.softmax(self.attention_weights)
        
        # 初始化融合特征张量
        fused_features = torch.zeros_like(x_conv)
        
        # 加权融合不同尺度的特征
        for i, feature in enumerate(multi_scale_features):
            # 确保维度匹配
            fused_features = fused_features + attn_weights[i] * feature
        
        # 转回原始格式 [batch_size, seq_len, d_model]
        fused_features = fused_features.transpose(1, 2)
        
        # 应用融合层
        output = self.fusion(fused_features)
        
        return output

class MultiScaleHead(nn.Module):
    """增强的多尺度预测头"""
    def __init__(self, 
                d_model: int,
                n_features: int,
                scale_factors: List[int] = [1, 6, 12, 24]):
        super().__init__()
        self.scale_factors = scale_factors
        
        # 多尺度特征提取
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in scale_factors
        ])
        
        # 预测头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_features)
            ) for _ in scale_factors
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        多尺度预测
        Args:
            x: [batch_size, pred_len, d_model]
        Returns:
            List of [batch_size, pred_len, n_features]
        """
        predictions = []
        
        # 确保输入是float类型
        if x.dtype != torch.float32:
            x = x.float()
        
        for i, (extractor, head) in enumerate(zip(self.feature_extractors, self.heads)):
            # 提取特征
            features = extractor(x)  # [batch_size, pred_len, d_model]
            # 应用预测头
            pred = head(features)    # [batch_size, pred_len, n_features]
            # 确保预测张量是连续的
            if self.training and pred.requires_grad:
                pred = pred.contiguous()
            predictions.append(pred)
        
        return predictions

class ProbabilisticPredictionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.mean_proj = nn.Linear(d_model, output_dim)
        self.var_proj = nn.Linear(d_model, output_dim)  # 预测方差
        
    def forward(self, x):
        mean = self.mean_proj(x)
        log_var = self.var_proj(x)  # 预测对数方差
        return mean, log_var

class ARMATransformer(nn.Module):
    """增强的Decoder-only ARMA Transformer"""
    def __init__(self,
                n_features: int,
                d_model: int = 512,
                n_heads: int = 8,
                n_layers: int = 6,
                d_ff: int = 2048,
                dropout: float = 0.1,
                max_seq_len: int = 5000,
                pred_len: int = 12,
                fusion_scales: List[int] = [1, 2, 4, 8],
                use_layer_fusion: bool = True,
                use_adaptive_weighting: bool = True,
                use_residual_scaling: bool = True,
                residual_scale_init: float = 1.0):
        super().__init__()
        
        self.pred_len = pred_len
        self.use_layer_fusion = use_layer_fusion
        
        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)
        
        # 可学习的位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFeatureFusion(d_model, scales=fusion_scales)
        
        # Transformer层
        self.layers = nn.ModuleList([
            ARMATransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                layer_idx=i,
                dropout=dropout
            ) for i in range(n_layers)
        ])
        
        # 如果使用残差缩放，设置初始值
        if use_residual_scaling:
            for layer in self.layers:
                layer.res_scale.data.fill_(residual_scale_init)
        
        # 层间连接融合
        if use_layer_fusion:
            self.layer_fusion = nn.Linear(d_model * 2, d_model)
        
        # 多尺度预测头
        self.multi_scale_head = MultiScaleHead(
            d_model=d_model,
            n_features=1,  # 只预测Global_active_power
            scale_factors=[1, 6, 12, 24]  # 多个预测尺度
        )
        
        # 最终层规范化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            p.requires_grad = True
        
    def forward(self, 
               x: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, n_features]
            mask: 可选的掩码张量
        Returns:
            不同时间尺度的预测列表，每个元素形状为 [batch_size, pred_len, 1]
        """
        # 确保输入是浮点类型并且保持连续信息
        if x.dtype != torch.float32:
            x = x.float()
        
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 应用多尺度特征融合
        fusion_output = self.multi_scale_fusion(x)
        x = fusion_output + x  # 添加残差连接
        
        # Transformer层
        layer_outputs = []
        prev_layer_output = None
        
        for i, layer in enumerate(self.layers):
            # 应用层，传入上一层的输出
            x, attn_output = layer(x, prev_layer_output, mask)
            
            # 保存每层的输出用于深度监督
            if i >= len(self.layers) // 2:  # 只保存后半部分层的输出
                layer_outputs.append(x)
            
            # 更新上一层输出
            if self.use_layer_fusion and i > 0 and hasattr(self, 'layer_fusion'):
                # 融合当前层和上一层的输出
                if prev_layer_output is not None:
                    prev_layer_output = self.layer_fusion(
                        torch.cat([x, prev_layer_output], dim=-1)
                    )
                else:
                    prev_layer_output = x
            else:
                prev_layer_output = x
            
            # 确保中间状态保持连续信息
            if x.requires_grad:
                x = x.contiguous()
                if prev_layer_output is not None:
                    prev_layer_output = prev_layer_output.contiguous()
        
        # 最终规范化
        x = self.final_norm(x)  # [batch_size, seq_len, d_model]
        
        # 只使用最后pred_len个时间步进行预测
        x = x[:, -self.pred_len:, :]  # [batch_size, pred_len, d_model]
        
        # 多尺度预测
        predictions = self.multi_scale_head(x)  # List of [batch_size, pred_len, 1]
        
        # 确保所有预测都保持连续信息
        if self.training:
            predictions = [p if not p.requires_grad else p.contiguous() for p in predictions]
        
        return predictions
    
    def generate(self, 
                x: torch.Tensor, 
                target_len: int,
                scale_idx: int = 0) -> torch.Tensor:
        """
        自回归生成
        Args:
            x: 输入序列 [batch_size, seq_len, n_features]
            target_len: 目标生成长度
            scale_idx: 使用哪个尺度的预测头(0:1h, 1:6h, 2:24h)
        Returns:
            生成的序列 [batch_size, target_len, 1]
        """
        self.eval()
        device = x.device
        batch_size = x.size(0)
        generated = []
        
        # 初始输入序列
        current_input = x
        
        # 自回归生成
        with torch.no_grad():
            for _ in range(target_len):
                # 前向传播
                predictions = self.forward(current_input)
                next_step = predictions[scale_idx][:, -1:]  # 取最后一个时间步 [batch_size, 1, 1]
                
                # 添加到生成序列
                generated.append(next_step)
                
                # 更新输入序列，只更新第一个特征（Global_active_power）
                next_input = current_input[:, 1:].clone()  # 移除第一个时间步
                next_input = torch.cat([next_input, current_input[:, -1:]], dim=1)  # 添加最后一个时间步的副本
                next_input[:, -1:, 0:1] = next_step  # 只更新Global_active_power
                current_input = next_input
        
        return torch.cat(generated, dim=1)
    

    def get_ar_output(self, x):
        """获取AR分支的输出"""
        return self.attention._ar_branch(x)

    def get_ma_output(self, x):
        """获取MA分支的输出"""
        return self.attention._ma_branch(x)

# # 使用示例
# if __name__ == "__main__":
#     # 模型参数
#     n_features = 7  # 输入特征维度
#     d_model = 512   # 模型维度
#     n_heads = 8     # 注意力头数
#     n_layers = 6    # 层数
#     d_ff = 2048     # 前馈网络维度
    
#     # 创建模型
#     model = ARMATransformer(
#         n_features=n_features,
#         d_model=d_model,
#         n_heads=n_heads,
#         n_layers=n_layers,
#         d_ff=d_ff
#     )

#    # 测试输入
#     batch_size = 32
#     seq_len = 100
#     x = torch.randn(batch_size, seq_len, n_features)
    
#     # 前向传播测试
#     predictions = model(x)
#     print("\n多尺度预测输出")
