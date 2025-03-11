#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import json

class Config:
    """配置类"""
    def __init__(self):
        # 获取项目根目录
        self.project_root = Path(__file__).parent.parent.resolve()
        
        # 模型参数
        self.model_params = {
            'n_features': 44,     
            'd_model': 256,       # 平衡维度
            'n_heads': 8,         # 保持标准头数
            'n_layers': 3,        # 保持层数
            'd_ff': 512,          # 平衡前馈维度
            'dropout': 0.1,       
            'max_seq_len': 500,   
            'fusion_scales': [1, 2, 4],  # 保持一定的多尺度能力
            'use_layer_fusion': True,
            'use_adaptive_weighting': True
        }
        
        # 数据参数
        self.data_params = {
            'data_path': str(Path(self.project_root) / "data" / "small_data" / "raw" / "electricity.txt"),
            'seq_len': 72,        # 平衡序列长度
            'pred_len': 12,       
            'batch_size': 32,     # 标准批次大小
            'num_workers': 0,     
            'pin_memory': False   
        }
        
        # 训练参数
        self.train_params = {
            'epochs': 25,         # 最小收敛轮数
            'learning_rate': 2e-4, # 平衡学习率
            'weight_decay': 1e-2,  
            'betas': (0.9, 0.999),
            'eps': 1e-8,          
            'grad_clip': 1.0,     
            'early_stopping_patience': 5, # 适当的早停
            'save_interval': 5,    
            # 损失函数参数
            'lambda_quantile': 0.2,
            'lambda_temporal': 0.1,
            'lambda_multiscale': 0.1,
            'quantiles': [0.1, 0.5, 0.9],  # 保持完整分位数
            'multiscale_scales': [1, 2, 4],  # 平衡多尺度
            'multiscale_weights': [1.0, 0.5, 0.25],
            'use_residual_scaling': True,
            'residual_scale_init': 1.0,
            
            # 渐进式训练
            'use_curriculum_learning': True,
            'seq_len_schedule': [48, 72],    # 平衡序列长度范围
            'pred_len_schedule': [12, 12],   
            'feature_schedule': [             
                ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity'],  
                ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
                 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
            ],
            'difficulty_metric': 'variance',
            
            # 动态采样
            'use_dynamic_sampling': True,
            'sampling_alpha': 0.6,
            'sampling_update_interval': 3,    
            
            # 学习率调度
            'use_warmup_scheduler': True,
            'warmup_epochs': 3,              
            'min_lr_ratio': 0.1
        }
        
        # 评估参数
        self.eval_params = {
            'metrics': ['mae', 'mse', 'rmse', 'mape'],  # 保持完整评估指标
            'test_steps': [1, 6, 12],  
            'mc_dropout_samples': 20,   # 平衡采样数
            'extreme_quantile': 0.95,
            
            # 保持关键的高级评估指标
            'advanced_metrics': [
                'directional_accuracy',
                'peak_accuracy',
                'forecast_bias',
                'theil_u'
            ],
            'visualize_metrics': True,
            'save_predictions': True
        }
        
        # 系统参数
        self.system_params = {
            'seed': 42,           # 随机种子
            'log_dir': 'logs',    # 日志目录
            'checkpoint_dir': 'checkpoints',  # 检查点目录
            'results_dir': 'results',  # 结果目录
            'use_amp': False,     # 是否使用自动混合精度
            'profiler': None,     # 性能分析器
            'debug': False        # 是否为调试模式
        }
        
        # 验证配置参数的合法性
        self._validate_config()
        
        # 创建必要的目录
        self._create_directories()
    
    def _validate_config(self):
        """验证配置参数的合法性"""
        assert self.model_params['d_model'] > 0, "d_model must be positive"
        assert self.model_params['n_heads'] > 0, "n_heads must be positive"
        assert self.model_params['n_layers'] > 0, "n_layers must be positive"
        assert 0 <= self.model_params['dropout'] <= 1, "dropout must be between 0 and 1"
        
        assert self.data_params['seq_len'] > 0, "seq_len must be positive"
        assert self.data_params['pred_len'] > 0, "pred_len must be positive"
        assert self.data_params['batch_size'] > 0, "batch_size must be positive"
        
        assert self.train_params['epochs'] > 0, "epochs must be positive"
        assert self.train_params['learning_rate'] > 0, "learning_rate must be positive"
        assert len(self.train_params['quantiles']) > 0, "quantiles must not be empty"
        assert len(self.train_params['multiscale_scales']) > 0, "multiscale_scales must not be empty"
        assert len(self.train_params['multiscale_scales']) == len(self.train_params['multiscale_weights']), \
            "multiscale_scales and multiscale_weights must have the same length"
        
        assert all(q > 0 and q < 1 for q in self.train_params['quantiles']), \
            "all quantiles must be between 0 and 1"
        assert all(s > 0 for s in self.train_params['multiscale_scales']), \
            "all scales must be positive"
        assert all(w > 0 for w in self.train_params['multiscale_weights']), \
            "all weights must be positive"
            
        # 验证课程学习参数
        if self.train_params.get('use_curriculum_learning', False):
            if 'seq_len_schedule' in self.train_params:
                assert len(self.train_params['seq_len_schedule']) == 2, \
                    "seq_len_schedule must have exactly 2 elements"
                assert all(s > 0 for s in self.train_params['seq_len_schedule']), \
                    "all seq_len_schedule values must be positive"
                    
            if 'pred_len_schedule' in self.train_params:
                assert len(self.train_params['pred_len_schedule']) == 2, \
                    "pred_len_schedule must have exactly 2 elements"
                assert all(p > 0 for p in self.train_params['pred_len_schedule']), \
                    "all pred_len_schedule values must be positive"
                    
            if 'feature_schedule' in self.train_params:
                assert len(self.train_params['feature_schedule']) >= 2, \
                    "feature_schedule must have at least 2 elements"
                    
        # 验证动态采样参数
        if self.train_params.get('use_dynamic_sampling', False):
            assert 0 <= self.train_params.get('sampling_alpha', 0.5) <= 1, \
                "sampling_alpha must be between 0 and 1"
            assert self.train_params.get('sampling_update_interval', 10) > 0, \
                "sampling_update_interval must be positive"
                
        # 验证学习率调度参数
        if self.train_params.get('use_warmup_scheduler', False):
            assert self.train_params.get('warmup_epochs', 5) > 0, \
                "warmup_epochs must be positive"
            assert 0 < self.train_params.get('min_lr_ratio', 0.1) <= 1, \
                "min_lr_ratio must be between 0 and 1"
    
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [
            self.system_params['log_dir'],
            self.system_params['checkpoint_dir'],
            self.system_params['results_dir']
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def update(self, updates: dict):
        """
        更新配置参数
        
        Args:
            updates: 包含要更新的参数的字典
        """
        for key, value in updates.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, dict):
                    current_value.update(value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config key: {key}")
        
        # 重新验证配置
        self._validate_config()
    
    def save(self, path: str):
        """
        保存配置到文件
        
        Args:
            path: 保存路径
        """
        config_dict = {
            'model_params': self.model_params,
            'data_params': self.data_params,
            'train_params': self.train_params,
            'eval_params': self.eval_params,
            'system_params': self.system_params
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """
        从文件加载配置
        
        Args:
            path: 配置文件路径
            
        Returns:
            Config: 配置对象
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.update(config_dict)
        return config

if __name__ == "__main__":
    # 测试配置
    config = Config()
    print(config) 
