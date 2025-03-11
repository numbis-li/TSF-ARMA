#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from models.arma_tsf import ARMATransformer
from data.small_data.electricity_dataloader import ElectricityDataLoader
from configs.config import Config
from utils.metrics import calculate_metrics, evaluate_multi_step, calculate_uncertainty
from utils.visualization import (
    plot_prediction_vs_actual, 
    plot_error_distribution, 
    plot_metrics_by_horizon, 
    plot_model_comparison,
    plot_uncertainty,
    create_evaluation_report
)
from utils.advanced_training import (
    AdvancedMetrics,
    visualize_advanced_metrics
)

class Evaluator:
    """模型评估器"""
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.setup_logging()
        self.setup_device()
        self.setup_data()
        self.setup_model()
        
    def setup_logging(self) -> None:
        """配置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "evaluation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_device(self) -> None:
        """配置计算设备（仅CPU）"""
        self.device = torch.device('cpu')
        self.logger.info(f"使用设备: {self.device}")
        
    def setup_data(self) -> None:
        """准备数据加载器"""
        data_loader = ElectricityDataLoader(
            data_path=self.config.data_params['data_path'],
            seq_len=self.config.data_params['seq_len'],
            pred_len=self.config.data_params['pred_len'],
            batch_size=self.config.data_params['batch_size']
        )
        
        _, _, self.test_loader = data_loader.get_data_loaders()
        self.logger.info(f"测试集批次数: {len(self.test_loader)}")
        
    def setup_model(self) -> None:
        """加载模型"""
        # 初始化模型
        self.model = ARMATransformer(**self.config.model_params)
        
        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"模型已从检查点加载: {self.checkpoint_path}")
        self.logger.info(f"检查点信息: 轮次 {checkpoint.get('epoch', 'unknown')}, 验证损失 {checkpoint.get('val_loss', 'unknown')}")
        
    def evaluate(self) -> Dict[str, float]:
        """评估模型性能"""
        self.logger.info("开始评估模型...")
        
        # 准备存储预测和真实值的列表
        all_preds = []
        all_trues = []
        
        # 在测试集上进行评估
        with torch.no_grad():
            for x_batch, y_batch in tqdm(self.test_loader, desc="评估进度"):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # 使用generate方法进行预测
                pred = self.model.generate(
                    x_batch,
                    target_len=self.config.data_params['pred_len'],
                    scale_idx=0  # 使用第一个尺度的预测头
                )
                
                # 只保留第一个特征的预测
                pred = pred[:, :, 0:1]
                
                # 收集预测和真实值
                all_preds.append(pred.detach().cpu().numpy())
                all_trues.append(y_batch.detach().cpu().numpy())
        
        # 合并所有批次的结果
        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        
        # 使用相同的指标列表
        metrics_list = self.config.eval_params['metrics'] + ['extreme']
        
        # 计算评估指标
        metrics = calculate_metrics(
            all_trues, 
            all_preds, 
            metrics_list=metrics_list,
            extreme_quantile=self.config.eval_params['extreme_quantile']
        )
        
        # 计算多步预测性能，使用相同的指标列表（但不包括extreme，因为它不适用于每个时间步）
        multi_step_metrics = evaluate_multi_step(
            all_trues, 
            all_preds, 
            metrics_list=self.config.eval_params['metrics']  # 不包括extreme
        )
        
        # 计算高级评估指标
        if self.config.eval_params.get('advanced_metrics'):
            self.logger.info("计算高级评估指标...")
            advanced_metrics = AdvancedMetrics.calculate_all_metrics(all_trues, all_preds)
            
            # 添加到基本指标中
            metrics.update(advanced_metrics)
            
            # 计算多步预测的高级指标
            advanced_multi_step_metrics = AdvancedMetrics.calculate_all_metrics(
                all_trues, all_preds, multi_step=True)
                
            # 合并多步指标
            multi_step_metrics.update(advanced_multi_step_metrics)
            
            # 可视化高级指标
            if self.config.eval_params.get('visualize_metrics', False):
                vis_dir = Path(self.config.system_params['results_dir']) / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                
                visualize_advanced_metrics(
                    advanced_multi_step_metrics,
                    title=f"{Path(self.checkpoint_path).stem} 高级指标随预测步长的变化",
                    save_path=str(vis_dir / f"{Path(self.checkpoint_path).stem}_advanced_metrics.png")
                )
        
        # 记录评估结果
        self.logger.info("评估指标:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.6f}")
        
        # 保存评估结果
        self.save_results(all_trues, all_preds, metrics, multi_step_metrics)
        
        return metrics
    
    def monte_carlo_dropout_evaluation(self, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """使用Monte Carlo Dropout进行不确定性估计"""
        self.logger.info(f"开始Monte Carlo Dropout评估 (样本数: {n_samples})...")
        
        # 启用dropout
        self.model.train()
        
        # 选择一个批次进行评估
        x_batch, y_batch = next(iter(self.test_loader))
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # 收集多次预测结果
        predictions = []
        
        with torch.no_grad():
            for _ in tqdm(range(n_samples), desc="MC Dropout采样"):
                # 前向传播
                pred_batch = self.model(x_batch)
                pred = pred_batch[0]  # 使用第一个尺度的预测
                predictions.append(pred.detach().cpu().numpy())
        
        # 计算不确定性
        uncertainty_results = calculate_uncertainty(predictions)
        
        # 保存不确定性分析结果
        self.save_uncertainty_results(y_batch.detach().cpu().numpy(), uncertainty_results)
        
        return uncertainty_results
    
    def save_results(self, 
                    y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    metrics: Dict[str, float],
                    multi_step_metrics: Dict[str, List[float]]) -> None:
        """保存评估结果"""
        # 创建结果目录
        results_dir = Path(self.config.system_params['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型名称
        model_name = Path(self.checkpoint_path).stem
        
        # 保存评估指标
        metrics_df = pd.DataFrame([metrics])
        metrics_path = results_dir / f"{model_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        self.logger.info(f"评估指标已保存至: {metrics_path}")
        
        # 保存多步预测指标
        # 确保所有指标列表长度一致
        pred_len = y_true.shape[1]
        for metric in list(multi_step_metrics.keys()):
            if len(multi_step_metrics[metric]) != pred_len:
                self.logger.warning(f"指标 {metric} 的长度 ({len(multi_step_metrics[metric])}) 与预测长度 ({pred_len}) 不匹配，将被移除")
                del multi_step_metrics[metric]
        
        if multi_step_metrics:
            multi_step_df = pd.DataFrame(multi_step_metrics)
            multi_step_df.insert(0, 'step', range(1, len(multi_step_df) + 1))
            multi_step_path = results_dir / f"{model_name}_multi_step_metrics.csv"
            multi_step_df.to_csv(multi_step_path, index=False)
            self.logger.info(f"多步预测指标已保存至: {multi_step_path}")
        else:
            self.logger.warning("没有有效的多步预测指标可保存")
        
        # 保存预测结果
        if self.config.eval_params.get('save_predictions', False):
            # 创建预测结果目录
            pred_dir = results_dir / "predictions"
            pred_dir.mkdir(exist_ok=True)
            
            # 保存预测和真实值
            np.save(str(pred_dir / f"{model_name}_predictions.npy"), y_pred)
            np.save(str(pred_dir / f"{model_name}_ground_truth.npy"), y_true)
            self.logger.info(f"预测结果已保存至: {pred_dir}")
            
            # 保存为CSV格式（更易于分析）
            # 展平数组并创建DataFrame
            pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
            true_flat = y_true.reshape(-1, y_true.shape[-1])
            
            # 创建索引
            sample_indices = np.repeat(np.arange(y_pred.shape[0]), y_pred.shape[1])
            step_indices = np.tile(np.arange(1, y_pred.shape[1] + 1), y_pred.shape[0])
            
            # 创建DataFrame
            results_df = pd.DataFrame({
                'sample_idx': sample_indices,
                'step': step_indices,
                'prediction': pred_flat.flatten(),
                'ground_truth': true_flat.flatten()
            })
            
            # 保存CSV
            csv_path = pred_dir / f"{model_name}_predictions.csv"
            results_df.to_csv(csv_path, index=False)
            self.logger.info(f"预测结果CSV已保存至: {csv_path}")
        
        # 创建可视化结果
        # 1. 预测结果与真实值对比图
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        plot_prediction_vs_actual(
            y_true, 
            y_pred, 
            feature_idx=0,  # 使用第一个特征
            sample_idx=0,   # 使用第一个样本
            title=f"{model_name} 预测结果与真实值对比",
            save_path=str(vis_dir / f"{model_name}_prediction_vs_actual.png")
        )
        
        # 2. 误差分布图
        plot_error_distribution(
            y_true, 
            y_pred, 
            feature_idx=0,
            title=f"{model_name} 预测误差分布",
            save_path=str(vis_dir / f"{model_name}_error_distribution.png")
        )
        
        # 3. 多步预测性能图
        if multi_step_metrics:
            plot_metrics_by_horizon(
                multi_step_metrics,
                title=f"{model_name} 预测性能随预测步长的变化",
                save_path=str(vis_dir / f"{model_name}_metrics_by_horizon.png")
            )
        
        # 创建评估报告
        create_evaluation_report(
            model_name=model_name,
            metrics=metrics,
            multi_step_metrics=multi_step_metrics,
            save_dir=str(results_dir)
        )
        
        self.logger.info(f"评估结果可视化已保存至: {vis_dir}")
    
    def save_uncertainty_results(self, 
                               y_true: np.ndarray, 
                               uncertainty_results: Dict[str, np.ndarray]) -> None:
        """保存不确定性分析结果"""
        # 创建结果目录
        results_dir = Path(self.config.system_params['results_dir'])
        vis_dir = results_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型名称
        model_name = Path(self.checkpoint_path).stem
        
        # 绘制不确定性分析图
        plot_uncertainty(
            y_true,
            uncertainty_results,
            feature_idx=0,
            sample_idx=0,
            title=f"{model_name} 预测不确定性分析",
            save_path=str(vis_dir / f"{model_name}_uncertainty.png")
        )
        
        self.logger.info(f"不确定性分析结果已保存至: {vis_dir}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="评估ARMA Transformer模型性能",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 在帮助信息中显示默认值
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/model_best.pth",  # 添加默认值
        help="模型检查点路径，默认使用'checkpoints/model_best.pth'"
    )
    parser.add_argument(
        "--mc_samples", 
        type=int, 
        default=100,
        help="Monte Carlo Dropout采样数量"
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=0,
        help="要评估的特征索引，默认为0（第一个特征）"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="结果保存目录"
    )
    args = parser.parse_args()
    
    try:
        # 检查检查点文件是否存在
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            checkpoints_dir = Path("checkpoints")
            if checkpoints_dir.exists():
                available_checkpoints = [f for f in checkpoints_dir.glob("*.pth")]
                print(f"\n错误: 找不到检查点文件 '{args.checkpoint}'")
                print("\n可用的检查点文件:")
                for ckpt in available_checkpoints:
                    print(f"  - {ckpt}")
            return
        
        # 加载配置
        config = Config()
        
        # 创建评估器
        evaluator = Evaluator(config, str(checkpoint_path))
        
        # 评估模型
        metrics = evaluator.evaluate()
        
        # 进行Monte Carlo Dropout评估
        if args.mc_samples > 0:
            uncertainty_results = evaluator.monte_carlo_dropout_evaluation(args.mc_samples)
            
    except Exception as e:
        logging.error(f"评估失败: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
