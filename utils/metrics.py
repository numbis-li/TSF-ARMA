import numpy as np
import torch
from typing import Dict, List, Union, Tuple

def mae(y_true: Union[np.ndarray, torch.Tensor], 
        y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算平均绝对误差 (Mean Absolute Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        float: MAE值
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算均方根误差 (Root Mean Squared Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        float: RMSE值
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def mape(y_true: Union[np.ndarray, torch.Tensor], 
         y_pred: Union[np.ndarray, torch.Tensor],
         epsilon: float = 1e-8) -> float:
    """
    计算平均绝对百分比误差 (Mean Absolute Percentage Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 小值，防止除零错误
    
    Returns:
        float: MAPE值 (百分比)
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 避免除以零
    mask = np.abs(y_true) > epsilon
    return 100.0 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon)))

def smape(y_true: Union[np.ndarray, torch.Tensor], 
          y_pred: Union[np.ndarray, torch.Tensor],
          epsilon: float = 1e-8) -> float:
    """
    计算对称平均绝对百分比误差 (Symmetric Mean Absolute Percentage Error)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 小值，防止除零错误
    
    Returns:
        float: SMAPE值 (百分比)
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def extreme_error(y_true: Union[np.ndarray, torch.Tensor], 
                  y_pred: Union[np.ndarray, torch.Tensor],
                  quantile: float = 0.95) -> Dict[str, float]:
    """
    计算极端情况下的误差（高于指定分位数的误差）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        quantile: 分位数阈值
    
    Returns:
        Dict: 包含极端情况下的MAE和RMSE
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 计算绝对误差
    abs_errors = np.abs(y_pred - y_true)
    
    # 确定阈值
    threshold = np.quantile(abs_errors, quantile)
    
    # 筛选极端误差
    extreme_mask = abs_errors >= threshold
    extreme_true = y_true[extreme_mask]
    extreme_pred = y_pred[extreme_mask]
    
    # 如果没有极端值，返回NaN
    if len(extreme_true) == 0:
        return {"extreme_mae": np.nan, "extreme_rmse": np.nan}
    
    # 计算极端情况下的指标
    extreme_mae_val = np.mean(np.abs(extreme_pred - extreme_true))
    extreme_rmse_val = np.sqrt(np.mean(np.square(extreme_pred - extreme_true)))
    
    return {
        "extreme_mae": extreme_mae_val,
        "extreme_rmse": extreme_rmse_val
    }

def calculate_metrics(y_true: Union[np.ndarray, torch.Tensor], 
                      y_pred: Union[np.ndarray, torch.Tensor],
                      metrics_list: List[str] = ['mae', 'rmse', 'mape'],
                      extreme_quantile: float = 0.95) -> Dict[str, float]:
    """
    计算多个评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics_list: 要计算的指标列表
        extreme_quantile: 极端值分位数
    
    Returns:
        Dict: 包含所有计算的指标
    """
    results = {}
    
    # 计算基本指标
    metrics_funcs = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape
    }
    
    for metric_name in metrics_list:
        if metric_name in metrics_funcs:
            results[metric_name] = metrics_funcs[metric_name](y_true, y_pred)
    
    # 计算极端情况下的指标
    if 'extreme' in metrics_list:
        extreme_results = extreme_error(y_true, y_pred, extreme_quantile)
        results.update(extreme_results)
    
    return results

def calculate_uncertainty(predictions: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    计算预测的不确定性
    
    Args:
        predictions: 多次预测结果的列表 [n_samples, ...]
    
    Returns:
        Dict: 包含均值、标准差、置信区间等
    """
    # 将预测结果堆叠
    stacked_preds = np.stack(predictions, axis=0)
    
    # 计算均值和标准差
    mean_pred = np.mean(stacked_preds, axis=0)
    std_pred = np.std(stacked_preds, axis=0)
    
    # 计算95%置信区间
    lower_bound = np.percentile(stacked_preds, 2.5, axis=0)
    upper_bound = np.percentile(stacked_preds, 97.5, axis=0)
    
    return {
        "mean": mean_pred,
        "std": std_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }

def evaluate_multi_step(y_true: Union[np.ndarray, torch.Tensor], 
                        y_pred: Union[np.ndarray, torch.Tensor],
                        metrics_list: List[str] = ['mae', 'rmse', 'mape']) -> Dict[str, List[float]]:
    """
    评估多步预测性能
    
    Args:
        y_true: 真实值 [batch_size, pred_len, features]
        y_pred: 预测值 [batch_size, pred_len, features]
        metrics_list: 要计算的指标列表
    
    Returns:
        Dict: 每个时间步的评估指标
    """
    # 确保输入是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    pred_len = y_true.shape[1]
    results = {metric: [] for metric in metrics_list}
    
    # 对每个时间步计算指标
    for i in range(pred_len):
        step_metrics = calculate_metrics(
            y_true[:, i, :], 
            y_pred[:, i, :], 
            metrics_list
        )
        
        for metric, value in step_metrics.items():
            results[metric].append(value)
    
    return results
