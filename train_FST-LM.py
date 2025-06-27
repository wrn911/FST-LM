#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
联邦学习实验运行脚本
提供多种实验配置的快速运行
"""

import subprocess
import sys
import os
from typing import List, Dict


class ExperimentRunner:
    """实验运行器"""

    def __init__(self):
        self.base_args = [
            "--dataset_dir", "dataset",
            "--file_path", "milano.h5",
            "--save_dir", "results",
            "--device", "auto",
            "--log_level", "INFO"
        ]

    def run_single_experiment(self, config: Dict[str, str], experiment_name: str = None):
        """运行单个实验"""
        cmd = ["python", "main.py"] + self.base_args

        # 添加配置参数
        for key, value in config.items():
            cmd.extend([f"--{key}", str(value)])

        if experiment_name:
            print(f"\n{'=' * 60}")
            print(f"开始实验: {experiment_name}")
            print(f"{'=' * 60}")
            print(f"命令: {' '.join(cmd)}")
            print()

        # 运行实验
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"实验 {experiment_name} 成功完成!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"实验 {experiment_name} 失败: {e}")
            return False
        except KeyboardInterrupt:
            print(f"实验 {experiment_name} 被用户中断")
            return False

    def run_multiple_experiments(self, experiments: List[Dict]):
        """运行多个实验"""
        successful = []
        failed = []

        for i, exp in enumerate(experiments):
            config = exp.get('config', {})
            name = exp.get('name', f'实验_{i + 1}')

            print(f"\n进度: {i + 1}/{len(experiments)}")

            if self.run_single_experiment(config, name):
                successful.append(name)
            else:
                failed.append(name)

        # 打印总结
        print(f"\n{'=' * 60}")
        print("实验批次总结")
        print(f"{'=' * 60}")
        print(f"成功: {len(successful)} 个")
        for name in successful:
            print(f"  ✓ {name}")

        if failed:
            print(f"失败: {len(failed)} 个")
            for name in failed:
                print(f"  ✗ {name}")

        return successful, failed


def get_quick_experiments():
    """获取快速实验配置（用于测试）"""
    return [
        {
            'name': '快速测试_FedAvg',
            'config': {
                'num_clients': 10,
                'rounds': 20,
                'local_epochs': 2,
                'frac': 0.5,
                'aggregation': 'fedavg',
                'eval_every': 5,
                'seq_len': 48,
                'pred_len': 12,
                'd_model': 32,
                'n_layers': 1
            }
        },
        {
            'name': '快速测试_加权聚合',
            'config': {
                'num_clients': 10,
                'rounds': 20,
                'local_epochs': 2,
                'frac': 0.5,
                'aggregation': 'weighted',
                'eval_every': 5,
                'seq_len': 48,
                'pred_len': 12,
                'd_model': 32,
                'n_layers': 1
            }
        }
    ]


def get_full_experiments():
    """获取完整实验配置"""
    return [
        {
            'name': '基线_FedAvg_Transformer',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'eval_every': 10,
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 2,
                'lr': 0.001
            }
        },
        {
            'name': '改进_加权聚合_Transformer',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.3,
                'aggregation': 'weighted',
                'use_coordinates': True,
                'eval_every': 10,
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 2,
                'lr': 0.001
            }
        },
        {
            'name': '大模型_FedAvg',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'eval_every': 10,
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 128,
                'n_heads': 8,
                'n_layers': 3,
                'lr': 0.0005,
                'patience': 15
            }
        },
        {
            'name': '不同参与比例_20%',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.2,
                'aggregation': 'fedavg',
                'eval_every': 10,
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 2,
                'lr': 0.001
            }
        },
        {
            'name': '不同参与比例_50%',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.5,
                'aggregation': 'fedavg',
                'eval_every': 10,
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_heads': 8,
                'n_layers': 2,
                'lr': 0.001
            }
        }
    ]


def get_ablation_experiments():
    """获取消融实验配置"""
    return [
        {
            'name': '消融_序列长度_48h',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'seq_len': 48,
                'pred_len': 24,
                'd_model': 64,
                'n_layers': 2
            }
        },
        {
            'name': '消融_序列长度_144h',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 3,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'seq_len': 144,
                'pred_len': 24,
                'd_model': 64,
                'n_layers': 2
            }
        },
        {
            'name': '消融_本地训练轮数_1',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 1,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_layers': 2
            }
        },
        {
            'name': '消融_本地训练轮数_5',
            'config': {
                'num_clients': 50,
                'rounds': 100,
                'local_epochs': 5,
                'frac': 0.3,
                'aggregation': 'fedavg',
                'seq_len': 96,
                'pred_len': 24,
                'd_model': 64,
                'n_layers': 2
            }
        }
    ]


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run_experiment.py quick     # 运行快速测试实验")
        print("  python run_experiment.py full      # 运行完整实验")
        print("  python run_experiment.py ablation  # 运行消融实验")
        print("  python run_experiment.py single    # 运行单个自定义实验")
        sys.exit(1)

    experiment_type = sys.argv[1].lower()
    runner = ExperimentRunner()

    if experiment_type == 'quick':
        print("运行快速测试实验...")
        experiments = get_quick_experiments()
        runner.run_multiple_experiments(experiments)

    elif experiment_type == 'full':
        print("运行完整实验...")
        experiments = get_full_experiments()
        runner.run_multiple_experiments(experiments)

    elif experiment_type == 'ablation':
        print("运行消融实验...")
        experiments = get_ablation_experiments()
        runner.run_multiple_experiments(experiments)

    elif experiment_type == 'single':
        print("运行单个实验...")
        # 默认配置
        config = {
            'num_clients': 20,
            'rounds': 50,
            'local_epochs': 3,
            'frac': 0.3,
            'aggregation': 'fedavg',
            'eval_every': 10,
            'seq_len': 96,
            'pred_len': 24,
            'd_model': 64,
            'n_layers': 2,
            'save_model': True
        }
        runner.run_single_experiment(config, '单个测试实验')

    else:
        print(f"未知的实验类型: {experiment_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()