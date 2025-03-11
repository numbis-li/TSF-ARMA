#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 获取项目根目录
current_file = Path(__file__).resolve()
project_root = str(current_file.parent.parent.parent)
sys.path.insert(0, project_root)

def create_small_dataset(
    input_file: str,
    output_file: str,
    sample_ratio: float = 0.1,  # 采样比例
    random_seed: int = 42
) -> None:
    """
    创建较小的数据集
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_ratio: 采样比例
        random_seed: 随机种子
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 读取原始数据
    print(f"读取数据文件: {input_file}")
    df = pd.read_csv(
        input_file,
        sep=';',
        parse_dates=['Date'],
        date_format='%d/%m/%Y'
    )
    
    # 获取原始数据大小
    original_size = len(df)
    print(f"原始数据集大小: {original_size} 条记录")
    
    # 计算采样大小
    sample_size = int(original_size * sample_ratio)
    print(f"目标采样大小: {sample_size} 条记录")
    
    # 确保采样的连续性（按时间顺序）
    # 我们选择连续的时间段而不是随机采样，以保持时间序列的连续性
    start_idx = np.random.randint(0, original_size - sample_size)
    sampled_df = df.iloc[start_idx:start_idx + sample_size].copy()
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存采样后的数据
    sampled_df.to_csv(output_file, sep=';', index=False)
    print(f"采样后的数据已保存至: {output_file}")
    print(f"采样数据集大小: {len(sampled_df)} 条记录")

def main():
    # 设置文件路径
    input_file = os.path.join(project_root, "data", "Electricity", "raw", "electricity.txt")
    output_file = os.path.join(project_root, "data", "small_data", "raw", "electricity.txt")
    
    # 创建小数据集
    create_small_dataset(
        input_file=input_file,
        output_file=output_file,
        sample_ratio=0.02  # 使用2%的数据
    )

if __name__ == "__main__":
    main() 