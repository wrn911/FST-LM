# -*- coding: utf-8 -*-
"""
实验结果保存工具
"""

import os
import json
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path


class ResultsSaver:
    """实验结果保存器"""

    def __init__(self, save_dir='results', dataset_name='milano'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.dataset_name = dataset_name

        # 创建结果文件
        self.csv_file = self.save_dir / f"{dataset_name}_results.csv"
        self.json_file = self.save_dir / f"{dataset_name}_detailed_results.json"

        # 初始化CSV文件（如果不存在）
        self._init_csv_file()

        # 存储当前实验的所有轮次结果
        self.round_results = []

    def _init_csv_file(self):
        """初始化CSV文件表头"""
        if not self.csv_file.exists():
            headers = [
                'timestamp', 'dataset', 'method', 'round',
                'test_mse', 'test_mae', 'val_mse', 'val_mae',
                'train_loss', 'num_clients', 'aggregation'
            ]
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def save_round_results(self, round_idx, test_metrics=None, val_metrics=None,
                           train_loss=None, method_name='unknown',
                           num_clients=0, aggregation='unknown'):
        """保存单轮结果"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 准备CSV行数据
        row_data = [
            timestamp,
            self.dataset_name,
            method_name,
            round_idx,
            test_metrics.get('mse', '') if test_metrics else '',
            test_metrics.get('mae', '') if test_metrics else '',
            val_metrics.get('mse', '') if val_metrics else '',
            val_metrics.get('mae', '') if val_metrics else '',
            train_loss if train_loss is not None else '',
            num_clients,
            aggregation
        ]

        # 写入CSV
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # 存储详细结果
        round_result = {
            'round': round_idx,
            'timestamp': timestamp,
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'train_loss': train_loss,
            'method': method_name,
            'num_clients': num_clients,
            'aggregation': aggregation
        }
        self.round_results.append(round_result)

        print(f"✓ 已保存第 {round_idx} 轮结果: "
              f"Test MSE={test_metrics.get('mse', 'N/A'):.6f}, "
              f"Test MAE={test_metrics.get('mae', 'N/A'):.6f}")

    def save_final_results(self, additional_info=None):
        """保存最终的详细结果到JSON"""
        final_results = {
            'dataset': self.dataset_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'round_results': self.round_results,
            'summary': self._generate_summary(),
            'additional_info': additional_info or {}
        }

        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"✓ 详细结果已保存至: {self.json_file}")
        print(f"✓ CSV结果已保存至: {self.csv_file}")

        return final_results

    def _generate_summary(self):
        """生成结果摘要"""
        if not self.round_results:
            return {}

        # 提取有效的测试指标
        test_mses = [r['test_metrics']['mse'] for r in self.round_results
                     if r['test_metrics'] and 'mse' in r['test_metrics']]
        test_maes = [r['test_metrics']['mae'] for r in self.round_results
                     if r['test_metrics'] and 'mae' in r['test_metrics']]

        summary = {
            'total_rounds': len(self.round_results),
            'method': self.round_results[0]['method'] if self.round_results else 'unknown'
        }

        if test_mses:
            summary.update({
                'best_test_mse': min(test_mses),
                'final_test_mse': test_mses[-1],
                'avg_test_mse': sum(test_mses) / len(test_mses),
                'best_test_mae': min(test_maes) if test_maes else None,
                'final_test_mae': test_maes[-1] if test_maes else None,
                'avg_test_mae': sum(test_maes) / len(test_maes) if test_maes else None
            })

        return summary

    def load_results_for_comparison(self):
        """加载结果用于对比分析"""
        if self.csv_file.exists():
            df = pd.read_csv(self.csv_file)
            return df
        return pd.DataFrame()

    @classmethod
    def compare_methods(cls, results_dir='results', dataset_name='milano'):
        """比较不同方法的结果"""
        csv_file = Path(results_dir) / f"{dataset_name}_results.csv"

        if not csv_file.exists():
            print(f"结果文件不存在: {csv_file}")
            return None

        df = pd.read_csv(csv_file)

        # 按方法分组统计
        comparison = df.groupby('method').agg({
            'test_mse': ['min', 'mean', 'std'],
            'test_mae': ['min', 'mean', 'std'],
            'round': 'count'
        }).round(6)

        print(f"\n=== {dataset_name} 数据集方法对比 ===")
        print(comparison)

        return comparison