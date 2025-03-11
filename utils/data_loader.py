#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

class TimeSeriesDataset(Dataset):
    """
    通用时间序列数据集基类
    """
    def __init__(self, seq_len, pred_len):
        self.seq_len = seq_len  # 输入序列长度
        self.pred_len = pred_len  # 预测序列长度
        self.scaler = StandardScaler()
        
    def _load_data(self, data_path):
        """加载数据（由子类实现）"""
        raise NotImplementedError
        
    def _preprocess(self, data):
        """预处理数据（由子类实现）"""
        raise NotImplementedError
        
    def __len__(self):
        """返回数据集长度"""
        raise NotImplementedError
        
    def __getitem__(self, idx):
        """返回单个样本"""
        raise NotImplementedError

class BaseDataLoader:
    """
    通用数据加载器基类
    """
    def __init__(self, batch_size, train_ratio=0.7, val_ratio=0.1):
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
    def _split_data(self, dataset):
        """
        将数据集分割为训练集、验证集和测试集
        """
        total_len = len(dataset)
        train_len = int(total_len * self.train_ratio)
        val_len = int(total_len * self.val_ratio)
        test_len = total_len - train_len - val_len
        
        return torch.utils.data.random_split(
            dataset, 
            [train_len, val_len, test_len]
        )
        
    def get_data_loaders(self):
        """
        返回训练、验证和测试数据加载器（由子类实现）
        """
        raise NotImplementedError

def create_small_dataset(source_path, target_path, num_rows=300):
    """
    从原始数据集中提取指定行数创建小数据集
    
    Args:
        source_path: 原始数据集路径
        target_path: 目标数据集保存路径
        num_rows: 需要提取的行数
    """
    try:
        # 读取原始数据
        with open(source_path, 'r') as f:
            lines = f.readlines()
        
        # 确保提取的行数不超过文件总行数
        num_rows = min(num_rows, len(lines))
        
        # 提取指定行数的数据
        selected_lines = lines[:num_rows]
        
        # 创建目标文件夹（如果不存在）
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 保存到新文件
        with open(target_path, 'w') as f:
            f.writelines(selected_lines)
            
        print(f"Successfully created small dataset with {num_rows} rows at {target_path}")
        
    except Exception as e:
        print(f"Error creating small dataset: {str(e)}")

def create_data_loaders(dataset, batch_size, train_ratio=0.7, val_ratio=0.15):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

class TimeSeriesDataLoader:
    def __init__(self, config):
        self.config = config
        
    def get_data_loaders(self):
        """返回训练、验证和测试数据加载器"""
        raise NotImplementedError 
