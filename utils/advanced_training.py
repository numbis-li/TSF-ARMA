#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Union, Callable
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path

class CurriculumScheduler:
    """
    课程学习调度器
    根据训练进度逐步增加任务难度
    """
    def __init__(self, 
                 total_epochs: int,
                 seq_len_schedule: List[int] = None,
                 pred_len_schedule: List[int] = None,
                 feature_schedule: List[List[str]] = None,
                 difficulty_metric: str = 'variance'):
        """
        初始化课程学习调度器
        
        Args:
            total_epochs: 总训练轮数
            seq_len_schedule: 输入序列长度调度 [初始长度, 最终长度]
            pred_len_schedule: 预测序列长度调度 [初始长度, 最终长度]
            feature_schedule: 特征使用调度 [初始特征列表, 最终特征列表]
            difficulty_metric: 难度度量方式 ('variance', 'entropy', 'gradient')
        """
        self.total_epochs = total_epochs
        self.seq_len_schedule = seq_len_schedule
        self.pred_len_schedule = pred_len_schedule
        self.feature_schedule = feature_schedule
        self.difficulty_metric = difficulty_metric
        self.logger = logging.getLogger(__name__)
        
    def get_params_for_epoch(self, epoch: int) -> Dict:
        """
        获取当前轮次的参数设置
        
        Args:
            epoch: 当前训练轮次
            
        Returns:
            Dict: 包含当前轮次的参数设置
        """
        # 计算进度比例 (0-1)
        progress = min(1.0, epoch / self.total_epochs)
        
        params = {}
        
        # 更新序列长度
        if self.seq_len_schedule:
            start_len, end_len = self.seq_len_schedule
            current_seq_len = int(start_len + progress * (end_len - start_len))
            params['seq_len'] = current_seq_len
            
        # 更新预测长度
        if self.pred_len_schedule:
            start_len, end_len = self.pred_len_schedule
            current_pred_len = int(start_len + progress * (end_len - start_len))
            params['pred_len'] = current_pred_len
            
        # 更新特征使用
        if self.feature_schedule and len(self.feature_schedule) >= 2:
            start_features = self.feature_schedule[0]
            end_features = self.feature_schedule[-1]
            
            # 如果有中间阶段，根据进度选择相应阶段
            if len(self.feature_schedule) > 2:
                stage = int(progress * (len(self.feature_schedule) - 1))
                current_features = self.feature_schedule[stage]
            else:
                # 线性增加特征数量
                num_features = int(len(start_features) + progress * (len(end_features) - len(start_features)))
                current_features = end_features[:num_features]
                
            params['features'] = current_features
            
        self.logger.info(f"Epoch {epoch}: Curriculum params: {params}")
        return params
    
    def calculate_sample_difficulty(self, 
                                   sample: torch.Tensor, 
                                   target: torch.Tensor = None) -> float:
        """
        计算样本难度
        
        Args:
            sample: 输入样本
            target: 目标值
            
        Returns:
            float: 样本难度分数
        """
        if self.difficulty_metric == 'variance':
            # 使用样本方差作为难度指标
            return torch.var(sample).item()
            
        elif self.difficulty_metric == 'entropy':
            # 使用样本熵作为难度指标
            # 归一化样本
            normalized = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample) + 1e-8)
            # 计算直方图
            hist = torch.histc(normalized, bins=10, min=0, max=1)
            # 计算概率
            probs = hist / torch.sum(hist)
            # 移除零概率
            probs = probs[probs > 0]
            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs))
            return entropy.item()
            
        elif self.difficulty_metric == 'gradient':
            # 使用梯度大小作为难度指标
            if target is not None:
                # 计算一阶差分
                diffs = torch.abs(sample[1:] - sample[:-1])
                return torch.mean(diffs).item()
            
        # 默认返回随机难度
        return np.random.random()

class WarmupCosineScheduler(_LRScheduler):
    """
    带预热的余弦学习率调度器
    """
    def __init__(self, 
                 optimizer, 
                 warmup_epochs: int,
                 total_epochs: int,
                 min_lr_ratio: float = 0.1,
                 last_epoch: int = -1):
        """
        初始化学习率调度器
        
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热轮数
            total_epochs: 总训练轮数
            min_lr_ratio: 最小学习率比例
            last_epoch: 上一轮次索引
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        计算当前学习率
        """
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            decayed_lr_ratio = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decayed_lr_ratio for base_lr in self.base_lrs]

class DynamicWeightedSampler:
    """
    动态加权采样器
    根据样本难度和重要性动态调整采样权重
    """
    def __init__(self, 
                 dataset: Dataset,
                 difficulty_fn: Callable,
                 importance_fn: Optional[Callable] = None,
                 alpha: float = 0.5,
                 update_interval: int = 10):
        """
        初始化动态加权采样器
        
        Args:
            dataset: 数据集
            difficulty_fn: 计算样本难度的函数
            importance_fn: 计算样本重要性的函数
            alpha: 难度与重要性的平衡系数 (0-1)
            update_interval: 权重更新间隔
        """
        self.dataset = dataset
        self.difficulty_fn = difficulty_fn
        self.importance_fn = importance_fn
        self.alpha = alpha
        self.update_interval = update_interval
        self.weights = torch.ones(len(dataset))
        self.steps = 0
        self.logger = logging.getLogger(__name__)
        
    def update_weights(self, 
                      model: nn.Module = None, 
                      loss_history: Dict[int, float] = None):
        """
        更新采样权重
        
        Args:
            model: 当前模型
            loss_history: 每个样本的损失历史
        """
        self.steps += 1
        
        # 检查是否需要更新
        if self.steps % self.update_interval != 0:
            return
            
        new_weights = []
        
        for i in range(len(self.dataset)):
            sample, target = self.dataset[i]
            
            # 计算难度分数
            difficulty = self.difficulty_fn(sample, target)
            
            # 计算重要性分数
            importance = 1.0
            if self.importance_fn and model is not None:
                importance = self.importance_fn(sample, target, model)
            
            # 如果有损失历史，使用它来调整重要性
            if loss_history and i in loss_history:
                importance *= (1.0 + loss_history[i])
                
            # 计算最终权重
            weight = self.alpha * difficulty + (1 - self.alpha) * importance
            new_weights.append(weight)
            
        # 归一化权重
        new_weights = torch.tensor(new_weights)
        new_weights = new_weights / (torch.sum(new_weights) + 1e-8)
        
        # 平滑更新
        self.weights = 0.7 * self.weights + 0.3 * new_weights
        
        # 记录权重统计信息
        self.logger.info(f"Updated sampling weights: min={torch.min(self.weights).item():.4f}, "
                         f"max={torch.max(self.weights).item():.4f}, "
                         f"mean={torch.mean(self.weights).item():.4f}")
        
    def get_sampler(self) -> WeightedRandomSampler:
        """
        获取加权随机采样器
        
        Returns:
            WeightedRandomSampler: PyTorch加权随机采样器
        """
        return WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.weights),
            replacement=True
        )

class AdvancedMetrics:
    """
    高级评估指标
    包括时间序列特有的评估指标
    """
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        方向准确率 (预测趋势方向的准确率)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            float: 方向准确率 (0-1)
        """
        # 计算真实值和预测值的差分
        true_diff = np.diff(y_true.flatten())
        pred_diff = np.diff(y_pred.flatten())
        
        # 计算方向是否一致
        correct_dir = (true_diff * pred_diff) > 0
        
        # 计算准确率
        return np.mean(correct_dir)
    
    @staticmethod
    def peak_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.2) -> float:
        """
        峰值准确率 (预测峰值的准确率)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            threshold: 峰值阈值
            
        Returns:
            float: 峰值准确率 (0-1)
        """
        # 展平数组
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 找出真实峰值
        true_peaks = []
        for i in range(1, len(y_true) - 1):
            if y_true[i] > y_true[i-1] and y_true[i] > y_true[i+1]:
                true_peaks.append(i)
                
        # 找出预测峰值
        pred_peaks = []
        for i in range(1, len(y_pred) - 1):
            if y_pred[i] > y_pred[i-1] and y_pred[i] > y_pred[i+1]:
                pred_peaks.append(i)
                
        # 如果没有峰值，返回0
        if not true_peaks:
            return 0.0
            
        # 计算匹配的峰值
        matches = 0
        for tp in true_peaks:
            for pp in pred_peaks:
                # 如果预测峰值在真实峰值附近，视为匹配
                if abs(tp - pp) <= threshold * len(y_true):
                    matches += 1
                    break
                    
        return matches / len(true_peaks)
    
    @staticmethod
    def correlation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        相关性指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict: 包含各种相关性指标
        """
        # 展平数组
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 计算Pearson相关系数
        pearson_corr, _ = pearsonr(y_true, y_pred)
        
        # 计算Spearman等级相关系数
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }
    
    @staticmethod
    def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        预测偏差 (平均预测误差)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            float: 预测偏差
        """
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def theil_u_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Theil's U统计量 (预测准确度的相对度量)
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            float: Theil's U统计量
        """
        # 展平数组
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # 计算预测模型的MSE
        mse_pred = np.mean(np.square(y_pred - y_true))
        
        # 计算朴素模型的MSE (使用前一个值作为预测)
        naive_pred = np.roll(y_true, 1)
        naive_pred[0] = naive_pred[1]  # 处理第一个值
        mse_naive = np.mean(np.square(naive_pred - y_true))
        
        # 计算Theil's U
        if mse_naive == 0:
            return 1.0
        return np.sqrt(mse_pred / mse_naive)
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             multi_step: bool = False) -> Dict[str, Union[float, List[float]]]:
        """
        计算所有高级指标
        
        Args:
            y_true: 真实值 [batch_size, pred_len, features]
            y_pred: 预测值 [batch_size, pred_len, features]
            multi_step: 是否计算多步预测指标
            
        Returns:
            Dict: 包含所有计算的指标
        """
        # 确保输入是numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
            
        # 初始化结果字典
        results = {}
        
        if not multi_step:
            # 计算整体指标
            # 基本指标
            results['mae'] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            results['rmse'] = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
            
            # 高级指标
            results['directional_accuracy'] = AdvancedMetrics.directional_accuracy(y_true, y_pred)
            results['peak_accuracy'] = AdvancedMetrics.peak_accuracy(y_true, y_pred)
            results['forecast_bias'] = AdvancedMetrics.forecast_bias(y_true, y_pred)
            results['theil_u'] = AdvancedMetrics.theil_u_statistic(y_true, y_pred)
            
            # 相关性指标
            corr_metrics = AdvancedMetrics.correlation_metrics(y_true, y_pred)
            results.update(corr_metrics)
        else:
            # 计算多步预测指标
            pred_len = y_true.shape[1]
            
            # 初始化多步指标
            multi_step_results = {
                'mae': [],
                'rmse': [],
                'directional_accuracy': [],
                'peak_accuracy': [],
                'forecast_bias': [],
                'theil_u': [],
                'pearson_correlation': [],
                'spearman_correlation': []
            }
            
            # 对每个预测步长计算指标
            for i in range(pred_len):
                step_true = y_true[:, i, :]
                step_pred = y_pred[:, i, :]
                
                # 基本指标
                multi_step_results['mae'].append(
                    mean_absolute_error(step_true.flatten(), step_pred.flatten()))
                multi_step_results['rmse'].append(
                    np.sqrt(mean_squared_error(step_true.flatten(), step_pred.flatten())))
                
                # 高级指标
                multi_step_results['directional_accuracy'].append(
                    AdvancedMetrics.directional_accuracy(step_true, step_pred))
                multi_step_results['peak_accuracy'].append(
                    AdvancedMetrics.peak_accuracy(step_true, step_pred))
                multi_step_results['forecast_bias'].append(
                    AdvancedMetrics.forecast_bias(step_true, step_pred))
                multi_step_results['theil_u'].append(
                    AdvancedMetrics.theil_u_statistic(step_true, step_pred))
                
                # 相关性指标
                corr_metrics = AdvancedMetrics.correlation_metrics(step_true, step_pred)
                multi_step_results['pearson_correlation'].append(corr_metrics['pearson_correlation'])
                multi_step_results['spearman_correlation'].append(corr_metrics['spearman_correlation'])
                
            results = multi_step_results
            
        return results

def visualize_advanced_metrics(metrics: Dict[str, List[float]], 
                              title: str = "高级指标随预测步长的变化",
                              save_path: Optional[str] = None):
    """
    可视化高级指标
    
    Args:
        metrics: 多步预测指标
        title: 图表标题
        save_path: 保存路径
    """
    # 检查指标是否为空
    if not metrics or not all(len(v) > 0 for v in metrics.values()):
        print("警告: 没有有效的指标数据可视化")
        return
    
    # 设置图表样式
    try:
        # 尝试使用新版本的seaborn样式
        plt.style.use('seaborn-whitegrid')
    except:
        try:
            # 尝试使用旧版本的seaborn样式
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            # 如果都不可用，使用默认样式
            print("警告: seaborn样式不可用，使用默认样式")
            plt.style.use('default')
    
    try:
        plt.figure(figsize=(15, 10))
        
        # 获取步长
        steps = range(1, len(list(metrics.values())[0]) + 1)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # 绘制误差指标
        ax1 = axes[0, 0]
        if 'mae' in metrics:
            ax1.plot(steps, metrics['mae'], 'o-', label='MAE')
        if 'rmse' in metrics:
            ax1.plot(steps, metrics['rmse'], 's-', label='RMSE')
        ax1.set_title('误差指标')
        ax1.set_xlabel('预测步长')
        ax1.set_ylabel('误差值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制方向和峰值准确率
        ax2 = axes[0, 1]
        if 'directional_accuracy' in metrics:
            ax2.plot(steps, metrics['directional_accuracy'], 'o-', label='方向准确率')
        if 'peak_accuracy' in metrics:
            ax2.plot(steps, metrics['peak_accuracy'], 's-', label='峰值准确率')
        ax2.set_title('准确率指标')
        ax2.set_xlabel('预测步长')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制相关性指标
        ax3 = axes[1, 0]
        if 'pearson_correlation' in metrics:
            ax3.plot(steps, metrics['pearson_correlation'], 'o-', label='Pearson相关系数')
        if 'spearman_correlation' in metrics:
            ax3.plot(steps, metrics['spearman_correlation'], 's-', label='Spearman相关系数')
        ax3.set_title('相关性指标')
        ax3.set_xlabel('预测步长')
        ax3.set_ylabel('相关系数')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 绘制其他指标
        ax4 = axes[1, 1]
        if 'forecast_bias' in metrics:
            ax4.plot(steps, metrics['forecast_bias'], 'o-', label='预测偏差')
        if 'theil_u' in metrics:
            ax4.plot(steps, metrics['theil_u'], 's-', label="Theil's U")
        ax4.set_title('其他指标')
        ax4.set_xlabel('预测步长')
        ax4.set_ylabel('指标值')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # 保存或显示图表
        if save_path:
            try:
                # 确保目录存在
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存至: {save_path}")
            except Exception as e:
                print(f"保存图表时出错: {e}")
            finally:
                plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"可视化指标时出错: {e}")
        # 确保关闭所有图表，避免内存泄漏
        plt.close('all') 