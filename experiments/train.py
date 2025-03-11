import sys
import gc
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from models.arma_tsf import ARMATransformer
from data.small_data.electricity_dataloader import ElectricityDataLoader
from configs.config import Config
from utils.advanced_training import (
    CurriculumScheduler, 
    WarmupCosineScheduler, 
    DynamicWeightedSampler,
    AdvancedMetrics,
    visualize_advanced_metrics
)

class QuantileLoss(nn.Module):
    """
    分位数损失函数
    用于预测区间估计
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        # 检查输入是否有效
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("警告: QuantileLoss - 预测值包含NaN或Inf")
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("警告: QuantileLoss - 目标值包含NaN或Inf")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        losses = []
        
        # 如果预测维度不匹配分位数数量，使用相同的预测值
        if preds.size(-1) < len(self.quantiles):
            preds = preds.expand(-1, -1, len(self.quantiles))
        
        try:
            for i, q in enumerate(self.quantiles):
                # 确保索引有效
                if i >= preds.size(-1):
                    break
                    
                # 提取当前分位数的预测
                pred_q = preds[..., min(i, preds.size(-1)-1):min(i+1, preds.size(-1))]
                
                # 计算误差
                errors = target - pred_q
                
                # 计算分位数损失
                q_loss = torch.max((q-1) * errors, q * errors)
                
                # 检查损失是否有效
                if torch.isnan(q_loss).any() or torch.isinf(q_loss).any():
                    print(f"警告: 分位数 {q} 的损失包含NaN或Inf")
                    continue
                
                losses.append(q_loss.mean())
        except Exception as e:
            print(f"计算分位数损失时出错: {e}")
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
        # 如果没有有效的损失，返回零张量
        if not losses:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)
        
        # 计算平均损失
        try:
            mean_loss = torch.stack(losses).mean()
            
            # 最终检查
            if torch.isnan(mean_loss) or torch.isinf(mean_loss):
                print("警告: 平均分位数损失为NaN或Inf")
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
                
            return mean_loss
        except Exception as e:
            print(f"计算平均分位数损失时出错: {e}")
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

class TemporalCoherenceLoss(nn.Module):
    """
    时间一致性损失函数
    用于保持预测的时间连续性
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, preds, target):
        # 检查输入是否有效
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("警告: TemporalCoherenceLoss - 预测值包含NaN或Inf")
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("警告: TemporalCoherenceLoss - 目标值包含NaN或Inf")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # 序列长度检查
            seq_len = preds.size(1)
            if seq_len <= 1:
                # 序列太短，无法计算差分
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
            # 计算一阶差分
            pred_diff = preds[:, 1:] - preds[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            
            # 检查差分是否有效
            if torch.isnan(pred_diff).any() or torch.isinf(pred_diff).any():
                print("警告: 预测差分包含NaN或Inf")
                pred_diff = torch.nan_to_num(pred_diff, nan=0.0, posinf=1.0, neginf=-1.0)
                
            if torch.isnan(target_diff).any() or torch.isinf(target_diff).any():
                print("警告: 目标差分包含NaN或Inf")
                target_diff = torch.nan_to_num(target_diff, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 计算差分的MSE损失
            diff_loss = F.mse_loss(pred_diff, target_diff)
            
            # 检查损失是否有效
            if torch.isnan(diff_loss) or torch.isinf(diff_loss):
                print("警告: 差分损失为NaN或Inf")
                return torch.tensor(0.0, device=preds.device, requires_grad=True)
            
            return self.alpha * diff_loss
        except Exception as e:
            print(f"计算时间一致性损失时出错: {e}")
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

class MultiScaleLoss(nn.Module):
    """
    多尺度损失函数
    用于捕捉不同时间尺度的模式
    """
    def __init__(self, scales=[1, 2, 3, 4], weights=None):
        super().__init__()
        self.scales = scales
        self.weights = weights if weights is not None else [1.0] * len(scales)
        
    def forward(self, preds, target):
        # 检查输入是否有效
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("警告: MultiScaleLoss - 预测值包含NaN或Inf")
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=-1.0)
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("警告: MultiScaleLoss - 目标值包含NaN或Inf")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        losses = []
        seq_len = preds.size(1)
        
        # 动态调整尺度，确保不超过序列长度的一半
        valid_scales = []
        valid_weights = []
        
        for s, w in zip(self.scales, self.weights):
            if s <= seq_len // 2 and s > 0:  # 确保尺度为正且不超过序列长度的一半
                valid_scales.append(s)
                valid_weights.append(w)
        
        # 如果没有有效的尺度，直接返回MSE损失
        if not valid_scales:
            return F.mse_loss(preds, target)
        
        # 对每个有效尺度计算损失
        for scale, weight in zip(valid_scales, valid_weights):
            try:
                if scale == 1:
                    # 原始尺度的MSE损失
                    scaled_loss = F.mse_loss(preds, target)
                else:
                    # 使用平均池化进行下采样
                    # 确保kernel_size不大于序列长度
                    kernel_size = min(scale, seq_len)
                    
                    # 安全检查：确保序列长度大于kernel_size
                    if seq_len <= kernel_size:
                        continue
                    
                    # 转置并添加维度以适应avg_pool1d
                    preds_t = preds.transpose(1, 2)  # [batch, features, seq_len]
                    target_t = target.transpose(1, 2)  # [batch, features, seq_len]
                    
                    # 计算填充，确保输出长度合理
                    padding = 0
                    output_size = (seq_len - kernel_size) // kernel_size + 1
                    if output_size < 1:
                        continue
                    
                    # 应用平均池化
                    scaled_pred = F.avg_pool1d(
                        preds_t, 
                        kernel_size=kernel_size, 
                        stride=kernel_size,
                        padding=padding
                    )
                    
                    scaled_target = F.avg_pool1d(
                        target_t, 
                        kernel_size=kernel_size, 
                        stride=kernel_size,
                        padding=padding
                    )
                    
                    # 检查输出是否有效
                    if scaled_pred.size(2) == 0 or scaled_target.size(2) == 0:
                        continue
                    
                    # 转置回原始格式
                    scaled_pred = scaled_pred.transpose(1, 2)  # [batch, seq_len/scale, features]
                    scaled_target = scaled_target.transpose(1, 2)  # [batch, seq_len/scale, features]
                    
                    # 计算MSE损失
                    scaled_loss = F.mse_loss(scaled_pred, scaled_target)
                
                # 检查损失是否有效
                if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                    print(f"警告: 尺度 {scale} 的损失为NaN或Inf")
                    continue
                
                losses.append(weight * scaled_loss)
            except Exception as e:
                print(f"计算尺度 {scale} 的损失时出错: {e}")
                continue
        
        # 如果没有有效的损失，返回基本MSE损失
        if not losses:
            return F.mse_loss(preds, target)
        
        # 计算加权平均损失
        total_weight = sum(valid_weights[:len(losses)])
        if total_weight == 0:
            return F.mse_loss(preds, target)
            
        return sum(losses) / total_weight  # 归一化损失

class CombinedLoss(nn.Module):
    """
    组合损失函数
    结合多个损失函数以提高预测性能
    """
    def __init__(self, config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.quantile_loss = QuantileLoss(quantiles=config.train_params.get('quantiles', [0.5]))
        self.temporal_coherence_loss = TemporalCoherenceLoss(alpha=0.1)
        self.multi_scale_loss = MultiScaleLoss(
            scales=config.train_params.get('multiscale_scales', [1, 2]),
            weights=config.train_params.get('multiscale_weights', [1.0, 0.5])
        )
        
        # 损失权重
        self.lambda_q = config.train_params.get('lambda_quantile', 0.1)
        self.lambda_t = config.train_params.get('lambda_temporal', 0.05)
        self.lambda_m = config.train_params.get('lambda_multiscale', 0.05)
        
        # 调试模式
        self.debug = False
        
    def forward(self, preds, target):
        # 确保预测是列表
        if not isinstance(preds, list):
            preds = [preds]
        
        # 使用第一个预测结果
        pred = preds[0]
        
        # 检查数值是否有效
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("警告: 预测值包含NaN或Inf")
            # 替换NaN和Inf为0
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("警告: 目标值包含NaN或Inf")
            # 替换NaN和Inf为0
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 基础MSE损失
        mse = self.mse_loss(pred, target)
        
        # 检查MSE损失是否有效
        if torch.isnan(mse) or torch.isinf(mse):
            print("警告: MSE损失为NaN或Inf")
            return torch.tensor(0.1, device=pred.device, requires_grad=True)
        
        # 分位数损失（用于不确定性估计）
        try:
            quantile = self.quantile_loss(pred, target)
            # 检查分位数损失是否有效
            if torch.isnan(quantile) or torch.isinf(quantile):
                print("警告: 分位数损失为NaN或Inf")
                quantile = torch.tensor(0.0, device=pred.device)
        except Exception as e:
            print(f"计算分位数损失时出错: {e}")
            quantile = torch.tensor(0.0, device=pred.device)
        
        # 时间一致性损失
        try:
            temporal = self.temporal_coherence_loss(pred, target)
            # 检查时间一致性损失是否有效
            if torch.isnan(temporal) or torch.isinf(temporal):
                print("警告: 时间一致性损失为NaN或Inf")
                temporal = torch.tensor(0.0, device=pred.device)
        except Exception as e:
            print(f"计算时间一致性损失时出错: {e}")
            temporal = torch.tensor(0.0, device=pred.device)
        
        # 多尺度损失
        try:
            multiscale = self.multi_scale_loss(pred, target)
            # 检查多尺度损失是否有效
            if torch.isnan(multiscale) or torch.isinf(multiscale):
                print("警告: 多尺度损失为NaN或Inf")
                multiscale = torch.tensor(0.0, device=pred.device)
        except Exception as e:
            print(f"计算多尺度损失时出错: {e}")
            multiscale = torch.tensor(0.0, device=pred.device)
        
        # 组合所有损失
        total_loss = mse + self.lambda_q * quantile + \
                    self.lambda_t * temporal + self.lambda_m * multiscale
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告: 总损失为NaN或Inf，使用MSE损失代替")
            return mse
        
        if self.debug:
            print(f"MSE: {mse.item():.6f}, Quantile: {quantile.item():.6f}, "
                  f"Temporal: {temporal.item():.6f}, MultiScale: {multiscale.item():.6f}, "
                  f"Total: {total_loss.item():.6f}")
        
        return total_loss

class Trainer:
    """模型训练器"""
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.setup_advanced_training()
        
    def setup_logging(self) -> None:
        """配置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self) -> None:
        """配置计算设备（仅CPU）"""
        self.device = torch.device('cpu')
        # 设置线程数以优化CPU性能
        torch.set_num_threads(psutil.cpu_count(logical=False))
        
    def setup_data(self) -> None:
        """准备数据加载器"""
        data_loader = ElectricityDataLoader(
            data_path=self.config.data_params['data_path'],
            seq_len=self.config.data_params['seq_len'],
            pred_len=self.config.data_params['pred_len'],
            batch_size=self.config.data_params['batch_size']
        )
        
        self.train_loader, self.val_loader, _ = data_loader.get_data_loaders()
        
    def setup_model(self) -> None:
        """初始化模型"""
        # 添加pred_len到模型参数
        model_params = self.config.model_params.copy()
        model_params['pred_len'] = self.config.data_params['pred_len']
        
        # 添加新的配置参数
        if 'fusion_scales' in self.config.model_params:
            model_params['fusion_scales'] = self.config.model_params['fusion_scales']
        
        if 'use_layer_fusion' in self.config.model_params:
            model_params['use_layer_fusion'] = self.config.model_params['use_layer_fusion']
        
        if 'use_adaptive_weighting' in self.config.model_params:
            model_params['use_adaptive_weighting'] = self.config.model_params['use_adaptive_weighting']
        
        if 'use_residual_scaling' in self.config.train_params:
            model_params['use_residual_scaling'] = self.config.train_params['use_residual_scaling']
            model_params['residual_scale_init'] = self.config.train_params['residual_scale_init']
        
        self.model = ARMATransformer(**model_params)
        self.model = self.model.to(self.device)
        
        # 确保所有参数都需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 计算总参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
    def setup_training(self) -> None:
        """配置训练组件"""
        # 设置内存优化
        torch.backends.cudnn.benchmark = False  # 禁用cudnn基准测试
        torch.backends.cudnn.deterministic = True  # 确保结果可复现
        
        # 设置CPU线程数
        torch.set_num_threads(4)  # 限制CPU线程数
        
        # 启用内存分配器
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.train_params['learning_rate'],
            weight_decay=self.config.train_params['weight_decay'],
            betas=self.config.train_params['betas'],
            eps=self.config.train_params['eps']
        )
        
        # 学习率调度器 - 兼容新旧配置
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.config.train_params['epochs']
        
        # 检查是否使用新的预热调度器
        if self.config.train_params.get('use_warmup_scheduler', False):
            # 新的预热调度器在setup_advanced_training中设置
            self.lr_scheduler = None
        else:
            # 兼容旧配置，如果没有warmup_steps_ratio，使用默认值0.3
            pct_start = self.config.train_params.get('warmup_steps_ratio', 0.3)
            
            self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.train_params['learning_rate'],
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=25.0,  # 初始学习率 = max_lr / div_factor
            final_div_factor=10000.0,  # 最终学习率 = max_lr / (div_factor * final_div_factor)
            anneal_strategy='cos'
        )
        
        # 损失函数
        self.criterion = CombinedLoss(self.config)
        
        # 最佳验证损失
        self.best_val_loss = float('inf')
        
        # 早停计数器
        self.early_stop_counter = 0
        
    def setup_advanced_training(self) -> None:
        """配置高级训练组件"""
        # 初始化最佳验证损失和早停计数器
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        
        # 设置课程学习调度器
        if self.config.train_params.get('use_curriculum_learning', False):
            self.curriculum_scheduler = CurriculumScheduler(
                total_epochs=self.config.train_params['epochs'],
                seq_len_schedule=self.config.train_params.get('seq_len_schedule'),
                pred_len_schedule=self.config.train_params.get('pred_len_schedule'),
                feature_schedule=self.config.train_params.get('feature_schedule'),
                difficulty_metric=self.config.train_params.get('difficulty_metric', 'variance')
            )
            self.logger.info("启用课程学习")
        else:
            self.curriculum_scheduler = None
            
        # 设置动态采样
        if self.config.train_params.get('use_dynamic_sampling', False):
            # 定义难度计算函数
            def calculate_difficulty(sample, target):
                if self.curriculum_scheduler:
                    return self.curriculum_scheduler.calculate_sample_difficulty(sample, target)
                else:
                    # 默认使用方差作为难度指标
                    return torch.var(sample).item()
            
            # 定义重要性计算函数
            def calculate_importance(sample, target, model):
                # 使用当前模型的损失作为重要性指标
                model.eval()
                with torch.no_grad():
                    sample = sample.unsqueeze(0).to(self.device)
                    target = target.unsqueeze(0).to(self.device)
                    pred = model(sample)
                    loss = self.criterion(pred, target)
                return loss.item()
            
            # 初始化动态采样器
            self.dynamic_sampler = DynamicWeightedSampler(
                dataset=self.train_loader.dataset,
                difficulty_fn=calculate_difficulty,
                importance_fn=calculate_importance,
                alpha=self.config.train_params.get('sampling_alpha', 0.5),
                update_interval=self.config.train_params.get('sampling_update_interval', 10)
            )
            self.logger.info("启用动态采样")
        else:
            self.dynamic_sampler = None
            
        # 设置学习率调度器
        if self.config.train_params.get('use_warmup_scheduler', False):
            self.lr_scheduler = WarmupCosineScheduler(
                optimizer=self.optimizer,
                warmup_epochs=self.config.train_params.get('warmup_epochs', 5),
                total_epochs=self.config.train_params['epochs'],
                min_lr_ratio=self.config.train_params.get('min_lr_ratio', 0.1)
            )
            self.logger.info("启用预热余弦学习率调度")
        else:
            # 使用默认的OneCycleLR
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.config.train_params['learning_rate'],
                epochs=self.config.train_params['epochs'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=10000.0
            )
            
        # 记录每个样本的损失历史
        self.sample_loss_history = {}
        
    def update_dataloader_for_epoch(self, epoch: int) -> None:
        """根据课程学习更新数据加载器"""
        if not self.curriculum_scheduler:
            return
            
        # 获取当前轮次的参数
        params = self.curriculum_scheduler.get_params_for_epoch(epoch)
        
        # 如果参数有变化，重新创建数据加载器
        if params:
            self.logger.info(f"更新数据加载器参数: {params}")
            
            # 更新数据加载器参数
            data_params = self.config.data_params.copy()
            data_params.update(params)
            
            # 重新创建数据加载器
            data_loader = ElectricityDataLoader(
                data_path=data_params['data_path'],
                seq_len=data_params.get('seq_len', self.config.data_params['seq_len']),
                pred_len=data_params.get('pred_len', self.config.data_params['pred_len']),
                batch_size=self.config.data_params['batch_size']
            )
            
            self.train_loader, self.val_loader, _ = data_loader.get_data_loaders()
            
            # 如果使用动态采样，更新采样器
            if self.dynamic_sampler:
                self.dynamic_sampler = DynamicWeightedSampler(
                    dataset=self.train_loader.dataset,
                    difficulty_fn=self.dynamic_sampler.difficulty_fn,
                    importance_fn=self.dynamic_sampler.importance_fn,
                    alpha=self.dynamic_sampler.alpha,
                    update_interval=self.dynamic_sampler.update_interval
                )
    
    def update_sampler(self) -> None:
        """更新动态采样器"""
        if not self.dynamic_sampler:
            return
            
        self.dynamic_sampler.update_weights(
            model=self.model,
            loss_history=self.sample_loss_history
        )
        
        # 使用更新后的采样器重新创建数据加载器
        self.train_loader = DataLoader(
            dataset=self.train_loader.dataset,
            batch_size=self.config.data_params['batch_size'],
            sampler=self.dynamic_sampler.get_sampler(),
            num_workers=0,
            pin_memory=False
        )
    
    def train_epoch(self, epoch: int) -> float:
        """训练一个轮次"""
        # 更新数据加载器
        self.update_dataloader_for_epoch(epoch)
        
        # 设置为训练模式
        self.model.train()
        
        # 初始化损失
        train_loss = 0.0
        valid_batches = 0
        
        # 进度条
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        # 遍历批次
        for batch_idx, (x_batch, y_batch) in enumerate(progress_bar):
            try:
                # 清除梯度
                self.optimizer.zero_grad()
                
                # 移动数据到设备
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 前向传播
                predictions = self.model(x_batch)
                
                # 计算损失
                loss = self.compute_loss(predictions, y_batch)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config.train_params.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train_params['grad_clip']
                    )
                
                # 更新参数
                self.optimizer.step()
                
                # 更新学习率
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                # 累加损失
                train_loss += loss.item()
                valid_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'avg_loss': f"{train_loss / valid_batches:.6f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # 记录样本损失历史
                if self.dynamic_sampler:
                    batch_start_idx = batch_idx * self.config.data_params['batch_size']
                    for i in range(len(x_batch)):
                        sample_idx = batch_start_idx + i
                        if sample_idx < len(self.train_loader.dataset):
                            self.sample_loss_history[sample_idx] = loss.item()
                
                # 定期清理缓存
                if batch_idx % 10 == 0:
                    self.clear_cache()
                    
            except Exception as e:
                self.logger.error(f"处理训练批次 {batch_idx} 时出错: {e}")
                continue
        
        # 更新采样器
        self.update_sampler()
        
        # 计算平均损失
        avg_train_loss = train_loss / valid_batches if valid_batches > 0 else float('inf')
        
        return avg_train_loss
    
    def validate(self) -> float:
        """在验证集上评估模型"""
        # 设置为评估模式
        self.model.eval()
        
        # 初始化损失
        val_loss = 0.0
        valid_batches = 0
        
        # 收集预测和真实值用于高级指标计算
        all_preds = []
        all_trues = []
        
        # 禁用梯度计算
        with torch.no_grad():
            # 遍历验证集
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm(self.val_loader, desc="Validating")):
                try:
                    # 移动数据到设备
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # 前向传播
                    predictions = self.model(x_batch)
                    
                    # 计算损失
                    loss = self.compute_loss(predictions, y_batch)
                    
                    # 累加损失
                    val_loss += loss.item()
                    valid_batches += 1
                    
                    # 收集预测和真实值
                    if isinstance(predictions, list):
                        pred = predictions[0]  # 使用第一个尺度的预测
                    else:
                        pred = predictions
                        
                    all_preds.append(pred.detach().cpu().numpy())
                    all_trues.append(y_batch.detach().cpu().numpy())
                    
                except Exception as e:
                    self.logger.error(f"处理验证批次 {batch_idx} 时出错: {e}")
                    continue
        
        # 计算平均损失
        avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
        self.logger.info(f"Validation loss: {avg_val_loss}")
        
        # 计算高级指标
        if all_preds and all_trues:
            try:
                # 合并所有批次的结果
                all_preds = np.concatenate(all_preds, axis=0)
                all_trues = np.concatenate(all_trues, axis=0)
                
                # 计算高级指标
                advanced_metrics = AdvancedMetrics.calculate_all_metrics(all_trues, all_preds)
                
                # 记录高级指标
                self.logger.info("高级评估指标:")
                for metric, value in advanced_metrics.items():
                    self.logger.info(f"{metric}: {value:.6f}")
                    
                # 计算多步预测指标
                multi_step_metrics = AdvancedMetrics.calculate_all_metrics(
                    all_trues, all_preds, multi_step=True)
                    
                # 可视化多步预测指标
                try:
                    vis_dir = Path(self.config.system_params['results_dir']) / "visualizations"
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    
                    visualize_advanced_metrics(
                        multi_step_metrics,
                        title=f"Epoch {self.current_epoch} 预测性能随步长的变化",
                        save_path=str(vis_dir / f"epoch_{self.current_epoch}_metrics.png")
                    )
                except Exception as e:
                    self.logger.warning(f"可视化高级指标时出错: {e}")
                    self.logger.warning("继续训练过程，但可视化结果将不可用")
            except Exception as e:
                self.logger.error(f"计算高级指标时出错: {e}")
        
        return avg_val_loss
    
    def train(self) -> None:
        """训练模型"""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.train_params['epochs']} epochs")
        self.logger.info(f"Training on device: {self.device}")
        
        # 记录初始内存使用
        self.log_memory_usage()
        
        # 训练循环
        for epoch in range(1, self.config.train_params['epochs'] + 1):
            # 记录当前轮次
            self.current_epoch = epoch
            
            # 训练一个轮次
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            
            # 在验证集上评估
            val_loss = self.validate()
            
            # 检查是否是最佳模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 记录轮次信息
            epoch_time = time.time() - start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            self.logger.info(f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            self.log_memory_usage()
            
            # 检查早停
            if self.early_stop_counter >= self.config.train_params['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

    def clear_cache(self) -> None:
        """清理内存缓存"""
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def log_memory_usage(self) -> None:
        """记录内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
        
    def compute_loss(self, predictions: Union[torch.Tensor, List[torch.Tensor]], 
                     targets: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        计算损失值
        
        Args:
            predictions: 模型预测值，可能是张量或张量列表
            targets: 目标值，可能是张量或张量列表
            
        Returns:
            torch.Tensor: 损失值
        """
        return self.criterion(predictions, targets)
    
    def save_checkpoint(self, 
                       epoch: int, 
                       val_loss: float, 
                       is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            val_loss: 验证损失
            is_best: 是否是最佳模型
        """
        checkpoint_dir = Path(self.config.system_params['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备检查点数据
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 添加学习率调度器状态（如果存在）
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()
        
        # 保存常规检查点
        if epoch % self.config.train_params.get('save_interval', 1) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        # 保存最佳模型
        if is_best:
            best_model_path = checkpoint_dir / "model_best.pth"
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Saved best model to {best_model_path}")

def main():
    """主函数"""
    try:
        # 加载配置
        config = Config()
        
        # 创建训练器
        trainer = Trainer(config)
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
