import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Union
import seaborn as sns
from datetime import datetime, timedelta

def set_plot_style():
    """设置绘图样式"""
    plt.style.use('seaborn')  # 使用更简单的seaborn样式
    
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    except:
        pass  # 如果字体不可用，使用默认字体
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18

def plot_prediction_vs_actual(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             feature_idx: int = 0,
                             sample_idx: int = 0,
                             timestamps: Optional[List[datetime]] = None,
                             title: str = "预测结果与真实值对比",
                             save_path: Optional[str] = None):
    """
    绘制预测结果与真实值的对比图
    
    Args:
        y_true: 真实值 [batch_size, pred_len, features]
        y_pred: 预测值 [batch_size, pred_len, features]
        feature_idx: 要绘制的特征索引
        sample_idx: 要绘制的样本索引
        timestamps: 时间戳列表，用于x轴
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    set_plot_style()
    
    # 提取指定样本和特征的数据
    true_values = y_true[sample_idx, :, feature_idx]
    pred_values = y_pred[sample_idx, :, feature_idx]
    
    # 创建x轴
    if timestamps is None:
        x = np.arange(len(true_values))
        x_label = "预测步数"
    else:
        x = timestamps[:len(true_values)]
        x_label = "时间"
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(x, true_values, 'b-', label='真实值', linewidth=2)
    plt.plot(x, pred_values, 'r--', label='预测值', linewidth=2)
    
    # 添加误差区域
    error = np.abs(true_values - pred_values)
    plt.fill_between(x, pred_values - error, pred_values + error, 
                    color='r', alpha=0.2, label='误差范围')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 如果是时间戳，设置x轴格式
    if timestamps is not None:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_error_distribution(y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           feature_idx: int = 0,
                           title: str = "预测误差分布",
                           save_path: Optional[str] = None):
    """
    绘制预测误差的分布图
    
    Args:
        y_true: 真实值 [batch_size, pred_len, features]
        y_pred: 预测值 [batch_size, pred_len, features]
        feature_idx: 要绘制的特征索引
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    set_plot_style()
    
    # 提取指定特征的数据
    true_values = y_true[:, :, feature_idx].flatten()
    pred_values = y_pred[:, :, feature_idx].flatten()
    
    # 计算误差
    errors = pred_values - true_values
    
    # 绘制误差分布
    plt.figure(figsize=(12, 6))
    
    # 左侧：误差直方图
    plt.subplot(1, 2, 1)
    sns.histplot(errors, kde=True)
    plt.title("误差分布直方图")
    plt.xlabel("误差")
    plt.ylabel("频率")
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # 右侧：QQ图
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title("误差QQ图")
    
    # 设置整体标题
    plt.suptitle(title, fontsize=16)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_metrics_by_horizon(metrics_by_step: Dict[str, List[float]],
                           title: str = "预测性能随预测步长的变化",
                           save_path: Optional[str] = None):
    """
    绘制多步预测性能随预测步长的变化
    
    Args:
        metrics_by_step: 每个步长的评估指标，格式为 {metric_name: [step1_value, step2_value, ...]}
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    set_plot_style()
    
    # 创建x轴
    steps = np.arange(1, len(list(metrics_by_step.values())[0]) + 1)
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    
    # 为每个指标绘制一条线
    for metric_name, values in metrics_by_step.items():
        plt.plot(steps, values, 'o-', linewidth=2, label=metric_name.upper())
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel("预测步长")
    plt.ylabel("指标值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(steps)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_model_comparison(models_metrics: Dict[str, Dict[str, float]],
                         metric_names: List[str] = ['mae', 'rmse', 'mape'],
                         title: str = "不同模型性能对比",
                         save_path: Optional[str] = None):
    """
    绘制不同模型性能的对比图
    
    Args:
        models_metrics: 不同模型的评估指标，格式为 {model_name: {metric_name: value, ...}}
        metric_names: 要绘制的指标名称列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    set_plot_style()
    
    # 准备数据
    model_names = list(models_metrics.keys())
    metrics_data = {metric: [models_metrics[model].get(metric, 0) for model in model_names] 
                   for metric in metric_names}
    
    # 设置图表
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 6))
    if len(metric_names) == 1:
        axes = [axes]
    
    # 为每个指标绘制条形图
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        bars = ax.bar(model_names, metrics_data[metric], alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        ax.set_title(f"{metric.upper()}")
        ax.set_ylabel("指标值")
        ax.grid(True, alpha=0.3, axis='y')
    
    # 设置整体标题
    plt.suptitle(title, fontsize=16)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_uncertainty(y_true: np.ndarray,
                    uncertainty_results: Dict[str, np.ndarray],
                    feature_idx: int = 0,
                    sample_idx: int = 0,
                    timestamps: Optional[List[datetime]] = None,
                    title: str = "预测不确定性分析",
                    save_path: Optional[str] = None):
    """
    绘制预测不确定性分析图
    
    Args:
        y_true: 真实值 [batch_size, pred_len, features]
        uncertainty_results: 不确定性分析结果，包含mean, std, lower_bound, upper_bound
        feature_idx: 要绘制的特征索引
        sample_idx: 要绘制的样本索引
        timestamps: 时间戳列表，用于x轴
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    set_plot_style()
    
    # 提取数据
    true_values = y_true[sample_idx, :, feature_idx]
    mean_pred = uncertainty_results["mean"][sample_idx, :, feature_idx]
    lower_bound = uncertainty_results["lower_bound"][sample_idx, :, feature_idx]
    upper_bound = uncertainty_results["upper_bound"][sample_idx, :, feature_idx]
    
    # 创建x轴
    x = np.arange(len(true_values))
    
    # 确保所有数组长度一致
    min_len = min(len(x), len(mean_pred))
    x = x[:min_len]
    true_values = true_values[:min_len]
    mean_pred = mean_pred[:min_len]
    lower_bound = lower_bound[:min_len]
    upper_bound = upper_bound[:min_len]
    
    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(x, true_values, 'b-', label='真实值', linewidth=2)
    plt.plot(x, mean_pred, 'r--', label='平均预测值', linewidth=2)
    
    # 添加置信区间
    plt.fill_between(x, lower_bound, upper_bound, 
                    color='r', alpha=0.2, label='95%置信区间')
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel("预测步数")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def create_evaluation_report(model_name: str,
                            metrics: Dict[str, float],
                            multi_step_metrics: Optional[Dict[str, List[float]]] = None,
                            save_dir: str = "results"):
    """
    创建评估报告
    
    Args:
        model_name: 模型名称
        metrics: 评估指标
        multi_step_metrics: 多步预测评估指标
        save_dir: 保存目录
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建报告文件
    report_path = os.path.join(save_dir, f"{model_name}_evaluation_report.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write(f"# {model_name} 模型评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 写入整体评估指标
        f.write("## 整体评估指标\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|------|\n")
        for metric, value in metrics.items():
            f.write(f"| {metric.upper()} | {value:.6f} |\n")
        f.write("\n")
        
        # 如果有多步预测指标，写入多步预测评估
        if multi_step_metrics and len(multi_step_metrics) > 0:
            # 确保所有指标列表长度一致
            first_metric = list(multi_step_metrics.values())[0]
            if all(len(values) == len(first_metric) for values in multi_step_metrics.values()):
                f.write("## 多步预测评估\n\n")
                
                # 获取步数
                steps = range(1, len(first_metric) + 1)
                
                # 创建表头
                f.write("| 步数 | " + " | ".join([m.upper() for m in multi_step_metrics.keys()]) + " |\n")
                f.write("|" + "------|" * (len(multi_step_metrics) + 1) + "\n")
                
                # 写入每一步的指标
                for i, step in enumerate(steps):
                    row = f"| {step} |"
                    for metric, values in multi_step_metrics.items():
                        row += f" {values[i]:.6f} |"
                    f.write(row + "\n")
                
                f.write("\n")
            else:
                f.write("## 多步预测评估\n\n")
                f.write("多步预测指标长度不一致，无法生成表格。\n\n")
        else:
            f.write("## 多步预测评估\n\n")
            f.write("没有可用的多步预测指标。\n\n")
        
        # 写入结论
        f.write("## 结论\n\n")
        
        # 根据MAE指标给出简单结论
        mae_value = metrics.get('mae', 0)
        if mae_value < 0.1:
            conclusion = "模型表现优秀，预测误差很小。"
        elif mae_value < 0.3:
            conclusion = "模型表现良好，预测误差在可接受范围内。"
        else:
            conclusion = "模型表现一般，预测误差较大，可能需要进一步优化。"
        
        f.write(conclusion + "\n")
    
    print(f"评估报告已保存至: {report_path}")
    return report_path
