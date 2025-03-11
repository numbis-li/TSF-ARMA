#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings

# 获取项目根目录并添加到Python路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data_loader import TimeSeriesDataset, BaseDataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

class ElectricityDataset(TimeSeriesDataset):
    """
    Electricity数据集处理类，增强版
    """
    def __init__(self, data_path, seq_len, pred_len):
        super().__init__(seq_len, pred_len)
        self.data_path = data_path
        # 使用相对路径指定processed目录
        self.processed_dir = Path(project_root) / "data" / "small_data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_and_preprocess_data()
        
    def _detect_and_handle_outliers(self, df, columns, n_sigma=3):
        """
        检测和处理异常值
        Args:
            df: 数据框
            columns: 要处理的列
            n_sigma: 标准差倍数阈值
        Returns:
            处理后的数据框
        """
        df_clean = df.copy()
        for col in columns:
            # 使用MAD方法检测异常值
            median = df[col].median()
            mad = stats.median_abs_deviation(df[col])
            lower_bound = median - n_sigma * mad
            upper_bound = median + n_sigma * mad
            
            # 记录异常值数量
            outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                warnings.warn(f"检测到{len(outliers)}个异常值在{col}列")
            
            # 使用中位数填充异常值
            df_clean.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = median
            
        return df_clean
    
    def _add_time_features(self, df):
        """
        添加丰富的时间特征
        Args:
            df: 带有日期索引的数据框
        Returns:
            添加时间特征后的数据框
        """
        # 基础时间特征
        df['hour'] = df.index.hour
        df['day'] = df.index.day
        df['weekday'] = df.index.weekday
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # 是否为工作日
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # 是否为节假日（简化版）
        df['is_holiday'] = ((df['month'] == 12) & (df['day'] == 25)).astype(int)  # 圣诞节
        
        # 一天中的时段
        df['time_period'] = pd.cut(df['hour'], 
                                 bins=[-1, 6, 12, 18, 23], 
                                 labels=['night', 'morning', 'afternoon', 'evening'])
        df = pd.get_dummies(df, columns=['time_period'])
        
        return df
    
    def _add_statistical_features(self, df, target_col='Global_active_power'):
        """
        添加统计特征
        Args:
            df: 数据框
            target_col: 目标列名
        Returns:
            添加统计特征后的数据框
        """
        # 滑动窗口统计
        windows = [6, 12, 24]  # 6小时、12小时、24小时
        for window in windows:
            # 移动平均
            df[f'rolling_mean_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1).mean()
            # 移动标准差
            df[f'rolling_std_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1).std()
            # 移动最大值
            df[f'rolling_max_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1).max()
            # 移动最小值
            df[f'rolling_min_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1).min()
        
        # 差分特征
        df['diff_1h'] = df[target_col].diff()
        df['diff_24h'] = df[target_col].diff(24)
        
        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _add_interaction_features(self, df):
        """
        添加交互特征
        Args:
            df: 数据框
        Returns:
            添加交互特征后的数据框
        """
        # 功率密度
        df['power_intensity'] = df['Global_active_power'] / df['Global_intensity']
        
        # 电压效率
        df['voltage_efficiency'] = df['Global_active_power'] / df['Voltage']
        
        # 总功耗比例
        df['sub_metering_sum'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
        df['other_consumption'] = df['Global_active_power'] * 1000/60 - df['sub_metering_sum']
        df['consumption_ratio'] = df['sub_metering_sum'] / (df['Global_active_power'] * 1000/60)
        
        return df
    
    def _load_and_preprocess_data(self):
        """加载并预处理数据"""
        # 读取数据
        df = pd.read_csv(
            self.data_path, 
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            dayfirst=True  # 适应欧洲日期格式
        )
        
        # 确保datetime列被正确解析为datetime类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 设置时间索引并确保其类型
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)  # 显式转换索引为datetime类型
        
        # 检测和处理异常值
        numeric_columns = ['Global_active_power', 'Global_reactive_power', 
                          'Voltage', 'Global_intensity', 
                          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        df = self._detect_and_handle_outliers(df, numeric_columns)
        
        # 添加时间特征
        df = self._add_time_features(df)
        
        # 添加统计特征
        df = self._add_statistical_features(df)
        
        # 添加交互特征
        df = self._add_interaction_features(df)
        
        # 选择输入特征
        input_features = [col for col in df.columns if col != 'Global_active_power']
        print(f"\n特征工程后的特征数量：{len(input_features)}")
        print("特征列表：")
        for i, feat in enumerate(input_features, 1):
            print(f"{i}. {feat}")
        
        # 使用RobustScaler进行特征标准化（对异常值更稳健）
        input_scaler = RobustScaler()
        input_scaled = input_scaler.fit_transform(df[input_features])
        
        # 保存输入特征的标准化器
        np.save(self.processed_dir / 'input_scaler.npy', input_scaler)
        
        # 标准化输出特征
        output_scaler = RobustScaler()
        output_data = df['Global_active_power'].values.reshape(-1, 1)
        output_scaled = output_scaler.fit_transform(output_data)
        
        # 保存输出特征的标准化器
        np.save(self.processed_dir / 'output_scaler.npy', output_scaler)
        
        # 合并输入和输出特征
        processed_data = np.concatenate([input_scaled, output_scaled], axis=1)
        
        # 分割并保存数据集
        self._split_and_save_data(processed_data)
        
        return processed_data
    
    def _split_and_save_data(self, data):
        """按70%，15%，15%的比例分割并保存数据集"""
        total_len = len(data)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)
        
        # 分割数据
        train_data = data[:train_len]
        val_data = data[train_len:train_len+val_len]
        test_data = data[train_len+val_len:]
        
        # 保存分割后的数据集
        np.save(self.processed_dir / 'train_data.npy', train_data)
        np.save(self.processed_dir / 'val_data.npy', val_data)
        np.save(self.processed_dir / 'test_data.npy', test_data)
        
        print("\n数据集分割完成：")
        print(f"训练集大小：{len(train_data)} (70%)")
        print(f"验证集大小：{len(val_data)} (15%)")
        print(f"测试集大小：{len(test_data)} (15%)")
        print(f"数据已保存至：{self.processed_dir}")
    
    def __len__(self):
        """返回数据集长度"""
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            (x, y): 输入序列和目标序列
        """
        # 获取输入序列
        x_start = idx
        x_end = idx + self.seq_len
        x = self.data[x_start:x_end, :-1]  # 除最后一列外的所有特征
        
        # 获取目标序列
        y_start = x_end
        y_end = y_start + self.pred_len
        y = self.data[y_start:y_end, -1:]  # 最后一列作为预测目标
        
        return torch.FloatTensor(x), torch.FloatTensor(y)

class ElectricityDataLoader(BaseDataLoader):
    """
    Electricity数据集加载器
    """
    def __init__(self, data_path, seq_len, pred_len, batch_size, 
                 train_ratio=0.7, val_ratio=0.15):
        super().__init__(batch_size, train_ratio, val_ratio)
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def get_data_loaders(self):
        """返回训练、验证和测试数据加载器"""
        # 创建数据集
        dataset = ElectricityDataset(
            data_path=self.data_path,
            seq_len=self.seq_len,
            pred_len=self.pred_len
        )
        
        # 分割数据集
        train_dataset, val_dataset, test_dataset = self._split_data(dataset)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader, test_loader

def test_electricity_dataloader():
    # 设置参数，使用相对路径
    data_path = Path(project_root) / "data" / "small_data" / "raw" / "electricity.txt"
    seq_len = 96      # 更新为与新配置一致
    pred_len = 12     # 保持不变
    batch_size = 32   # 更新为与新配置一致
    
    # 创建数据加载器
    data_loader = ElectricityDataLoader(
        data_path=str(data_path),
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size
    )
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    # 打印信息
    print(f"训练集批次数：{len(train_loader)}")
    print(f"验证集批次数：{len(val_loader)}")
    print(f"测试集批次数：{len(test_loader)}")
    
    # 获取一个批次的数据
    x_batch, y_batch = next(iter(train_loader))
    print(f"输入形状：{x_batch.shape}")
    print(f"目标形状：{y_batch.shape}")

if __name__ == "__main__":
    test_electricity_dataloader()
