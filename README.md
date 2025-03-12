# ARMA Transformer 时间序列预测项目

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/badge/Last%20Updated-March%202024-green.svg)](https://github.com/yourusername/TSF-ARMA)

## 目录

- [项目简介](#项目简介)
- [最新实验结果](#最新实验结果)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [使用说明](#使用说明)
- [实验结果复现](#实验结果复现)
- [注意事项](#注意事项)
- [常见问题](#常见问题)
- [维护者](#维护者)
- [许可证](#许可证)

## 项目简介
本项目实现了一个基于 ARMA Transformer 的时间序列预测模型，特别适用于电力消耗预测等时间序列数据。该模型结合了 ARMA（自回归移动平均）和 Transformer 架构的优点，能够有效捕捉时间序列数据的长期依赖关系和局部模式。

## 最新实验结果
- MAE: 0.2508
- RMSE: 0.4661
- 峰值准确率: 100%
- 相关系数: 0.7659

详细结果请参见 [RESULTS.md](RESULTS.md)

## 环境要求
- Python 3.7-3.10
- PyTorch 1.8+
- CUDA（可选，支持CPU训练）
- 内存：≥8GB
- 存储空间：≥1GB

完整依赖列表请参见 [requirements.txt](requirements.txt)

## 快速开始

1. 克隆仓库
```bash
git clone https://github.com/yourusername/TSF-ARMA.git
cd TSF-ARMA
```

2. 环境配置
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. 数据准备
```python
python data/electricity_data/preprocess.py
```

4. 训练模型
```python
python experiments/train.py
```

5. 评估模型
```python
python experiments/evaluate.py --checkpoint checkpoints/model_best.pth
```

## 项目结构
```
TSF-ARMA/
├── configs/            # 配置文件
│   └── config.py      # 模型和训练配置
├── data/              # 数据目录
│   └── electricity_data/
├── models/            # 模型实现
│   ├── attention.py   # ARMA注意力机制
│   └── arma_tsf.py    # 主模型架构
├── experiments/       # 实验脚本
│   ├── train.py      # 训练脚本
│   └── evaluate.py    # 评估脚本
├── utils/            # 工具函数
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   └── advanced_training.py
├── results/          # 实验结果
├── checkpoints/      # 模型检查点
└── logs/            # 训练日志
```

## 模型架构

### ARMA注意力机制
- AR分支：使用门控因果空洞卷积捕捉长期依赖
- MA分支：使用动态深度可分离卷积处理局部模式
- 自适应权重分配：动态平衡AR和MA分支的贡献

### 优化特点
1. CPU环境优化
   - 批处理大小：16
   - 模型维度：32
   - 注意力头数：2

2. 训练策略
   - 课程学习
   - 动态采样
   - 早停机制

## 使用说明

### 配置修改
修改 `configs/config.py` 中的参数：
```python
model_params = {
    'd_model': 32,
    'n_heads': 2,
    'n_layers': 2,
    'd_ff': 128
}
```

### 自定义数据
1. 准备数据文件（CSV格式）
2. 修改 `data_loader.py` 中的数据加载逻辑
3. 更新配置文件中的数据参数

### 模型训练
```python
python experiments/train.py --config configs/config.py
```

### 模型评估
```python
python experiments/evaluate.py --checkpoint checkpoints/model_best.pth
```

## 实验结果复现
1. 使用提供的配置文件
2. 确保数据预处理步骤一致
3. 使用相同的随机种子（42）
4. 训练环境：
   - CPU：≥4核
   - 内存：≥8GB
   - 操作系统：Windows/Linux/MacOS
   - Python版本：3.7-3.10

## 注意事项
1. 数据预处理
   - 确保数据标准化
   - 处理缺失值
   - 检查异常值

2. 训练过程
   - 监控内存使用
   - 观察损失曲线
   - 注意早停条件

3. 模型评估
   - 使用多个指标
   - 考虑预测区间
   - 分析误差分布

## 常见问题

### 1. 内存不足
- 减小批处理大小
- 降低模型维度
- 使用梯度累积

### 2. 训练不稳定
- 调整学习率
- 检查梯度裁剪
- 增加预热轮数

### 3. 预测偏差
- 检查数据分布
- 调整损失权重
- 优化采样策略

## 维护者
- 项目作者：李宇彤
- 联系方式：338959685@qq.com
- 项目主页：https://github.com/numbis-li/TSF-ARMA

## 许可证
本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 更新日志
- 2024-03-11: 完成基础模型训练和评估
- 2024-03-10: 实现 ARMA 注意力机制
- 2024-03-09: 项目初始化 
