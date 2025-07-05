#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM中心化时序预测Baseline
与联邦学习TimeLLM进行对比的传统中心化方法
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import random
import argparse
import json
from pathlib import Path


class LSTMPredictor(nn.Module):
    """LSTM时序预测模型"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=24, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


class CentralizedDataProcessor:
    """中心化数据处理器"""

    def __init__(self, seq_len=96, pred_len=24, test_days=7, val_days=1):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.test_days = test_days
        self.val_days = val_days

    def load_and_prepare_data(self, data_file, num_stations=50):
        """加载数据并准备中心化训练数据"""
        print(f"加载数据文件: {data_file}")

        # 读取HDF5文件
        with h5py.File(data_file, 'r') as f:
            print(f"数据文件字段: {list(f.keys())}")

            idx = f['idx'][()]
            cell = f['cell'][()]
            lng = f['lng'][()]
            lat = f['lat'][()]
            traffic_data = f['net'][()][:, cell - 1]

        # 构建DataFrame
        df = pd.DataFrame(
            traffic_data,
            index=pd.to_datetime(idx.ravel(), unit='s'),
            columns=cell
        )
        df.fillna(0, inplace=True)

        print(f"原始数据形状: {df.shape}")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        # 选择基站（与联邦学习保持一致）
        selected_stations = sorted(random.sample(list(cell), num_stations))
        df_selected = df[selected_stations].copy()

        print(f"选择基站数量: {len(selected_stations)}")
        print(f"基站ID: {selected_stations}")

        # 获取坐标信息
        selected_cells_idx = np.where(np.isin(cell, selected_stations))[0]
        coordinates = {
            cell_id: {'lng': float(lng[idx]), 'lat': float(lat[idx])}
            for cell_id, idx in zip(selected_stations, selected_cells_idx)
        }

        return df_selected, selected_stations, coordinates

    def prepare_centralized_training_data(self, df_stations):
        """准备中心化训练数据 - 将所有基站数据合并"""
        print("准备中心化训练数据...")

        all_sequences_X = []
        all_sequences_y = []
        station_info = []

        # 为每个基站创建时序序列
        for station_id in df_stations.columns:
            station_data = df_stations[station_id]

            # 检查数据质量
            if station_data.std() == 0 or station_data.isna().all():
                print(f"基站 {station_id}: 数据质量差，跳过")
                continue

            # 标准化（基于整个时序的统计量）
            mean = station_data.mean()
            std = station_data.std()
            if std == 0:
                std = 1.0

            normalized_data = (station_data - mean) / std

            # 创建时序序列
            X, y = self._create_sequences(normalized_data)

            if len(X) > 0:
                all_sequences_X.append(X)
                all_sequences_y.append(y)

                # 保存标准化参数，用于后续反标准化
                station_info.append({
                    'station_id': station_id,
                    'mean': mean,
                    'std': std,
                    'samples': len(X)
                })

        if not all_sequences_X:
            raise ValueError("没有有效的基站数据")

        # 合并所有基站的序列
        combined_X = np.concatenate(all_sequences_X, axis=0)
        combined_y = np.concatenate(all_sequences_y, axis=0)

        print(f"合并后训练数据形状: X={combined_X.shape}, y={combined_y.shape}")
        print(f"有效基站数量: {len(station_info)}")

        return combined_X, combined_y, station_info

    def prepare_test_data_per_station(self, df_stations, station_info):
        """为每个基站准备独立的测试数据"""
        station_test_data = {}

        for info in station_info:
            station_id = info['station_id']
            station_data = df_stations[station_id]

            # 使用训练时的标准化参数
            normalized_data = (station_data - info['mean']) / info['std']

            # 创建时序序列
            X, y = self._create_sequences(normalized_data)

            if len(X) > 0:
                # 只取测试集部分
                test_samples = self.test_days * 24
                if len(X) >= test_samples:
                    X_test = X[-test_samples:]
                    y_test = y[-test_samples:]

                    station_test_data[station_id] = {
                        'X': X_test,
                        'y': y_test,
                        'norm_params': {'mean': info['mean'], 'std': info['std']}
                    }

        print(f"准备了 {len(station_test_data)} 个基站的测试数据")
        return station_test_data

    def _create_sequences(self, data):
        """创建时序序列"""
        X, y = [], []

        for i in range(self.seq_len, len(data) - self.pred_len + 1):
            X.append(data.iloc[i - self.seq_len:i].values)
            y.append(data.iloc[i:i + self.pred_len].values)

        return np.array(X), np.array(y)

    def split_train_val(self, X, y):
        """分割训练和验证集"""
        val_samples = int(len(X) * 0.1)  # 10%作为验证集

        X_train = X[:-val_samples] if val_samples > 0 else X
        y_train = y[:-val_samples] if val_samples > 0 else y
        X_val = X[-val_samples:] if val_samples > 0 else None
        y_val = y[-val_samples:] if val_samples > 0 else None

        print(f"训练集: {len(X_train)} 样本")
        if X_val is not None:
            print(f"验证集: {len(X_val)} 样本")

        return (X_train, y_train), (X_val, y_val)


class CentralizedLSTMTrainer:
    """中心化LSTM训练器"""

    def __init__(self, model, device='cuda', lr=0.001, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()

            # 添加特征维度：(batch_size, seq_len) -> (batch_size, seq_len, 1)
            if len(X_batch.shape) == 2:
                X_batch = X_batch.unsqueeze(-1)

            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                # 添加特征维度：(batch_size, seq_len) -> (batch_size, seq_len, 1)
                if len(X_batch.shape) == 2:
                    X_batch = X_batch.unsqueeze(-1)

                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=100, patience=20):
        """完整训练流程"""
        print("开始训练中心化LSTM模型...")

        best_model_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                    print(f"Epoch {epoch:3d}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f} (最佳)")
                else:
                    patience_counter += 1
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch:3d}: 训练损失={train_loss:.6f}, 验证损失={val_loss:.6f}")

                if patience_counter >= patience:
                    print(f"早停触发! 最佳验证损失: {self.best_val_loss:.6f}")
                    break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:3d}: 训练损失={train_loss:.6f}")
                best_model_state = self.model.state_dict().copy()

        if best_model_state:
            self.model.load_state_dict(best_model_state)

        print(f"训练完成! 最终损失: {self.best_val_loss:.6f}")

    def predict_for_station(self, X_test, y_test, norm_params):
        """为单个基站进行预测"""
        self.model.eval()

        # 创建测试数据加载器
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device).float()

                # 添加特征维度：(batch_size, seq_len) -> (batch_size, seq_len, 1)
                if len(X_batch.shape) == 2:
                    X_batch = X_batch.unsqueeze(-1)

                predictions = self.model(X_batch)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y_batch.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 反标准化
        pred_denorm = predictions * norm_params['std'] + norm_params['mean']
        target_denorm = targets * norm_params['std'] + norm_params['mean']

        return pred_denorm, target_denorm


def calculate_metrics(predictions, targets):
    """计算评估指标"""
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    mse = mean_squared_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_flat, pred_flat)

    # MAPE
    mask = target_flat != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((target_flat[mask] - pred_flat[mask]) / target_flat[mask])) * 100
    else:
        mape = float('inf')

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}


def plot_training_history(trainer, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(trainer.train_losses) + 1)
    plt.plot(epochs, trainer.train_losses, label='训练损失', color='blue')

    if trainer.val_losses:
        plt.plot(epochs, trainer.val_losses, label='验证损失', color='red')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('中心化LSTM训练历史')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史保存至: {save_path}")

    plt.show()


def plot_sample_predictions(station_results, num_samples=3, save_path=None):
    """绘制部分基站的预测结果示例"""
    station_ids = list(station_results.keys())[:num_samples]

    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, station_id in enumerate(station_ids):
        result = station_results[station_id]
        predictions = result['predictions'][:72].flatten()  # 前3天
        targets = result['targets'][:72].flatten()

        time_steps = range(len(predictions))

        axes[i].plot(time_steps, targets, label='真实值', alpha=0.7, color='blue')
        axes[i].plot(time_steps, predictions, label='预测值', alpha=0.7, color='red')
        axes[i].set_title(f'基站 {station_id} - RMSE: {result["metrics"]["RMSE"]:.3f}')
        axes[i].set_xlabel('时间(小时)')
        axes[i].set_ylabel('流量值')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果示例保存至: {save_path}")

    plt.show()


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='中心化LSTM时序预测Baseline')

    # 数据参数
    parser.add_argument('--data_file', type=str, default='dataset/milano.h5',
                        help='数据文件路径')
    parser.add_argument('--num_stations', type=int, default=50,
                        help='使用的基站数量')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='预测序列长度')
    parser.add_argument('--test_days', type=int, default=7,
                        help='测试集天数')

    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')

    # 系统参数
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_results', action='store_true',
                        help='是否保存结果')

    args = parser.parse_args()

    # 设备选择
    device = 'cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu'
    print(f"使用设备: {device}")

    # 设置随机种子
    set_seed(args.seed)

    print("=" * 60)
    print("中心化LSTM时序预测Baseline")
    print("用于与联邦学习TimeLLM进行性能对比")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在 {args.data_file}")
        return

    # 1. 数据准备
    processor = CentralizedDataProcessor(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        test_days=args.test_days
    )

    # 加载数据
    df_stations, station_ids, coordinates = processor.load_and_prepare_data(
        args.data_file, args.num_stations
    )

    # 准备中心化训练数据
    X_combined, y_combined, station_info = processor.prepare_centralized_training_data(df_stations)

    # 分割训练和验证集
    (X_train, y_train), (X_val, y_val) = processor.split_train_val(X_combined, y_combined)

    # 创建数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 模型创建和训练
    print("\n创建中心化LSTM模型...")
    model = LSTMPredictor(
        input_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        output_size=args.pred_len,
        dropout=args.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    trainer = CentralizedLSTMTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience
    )

    # 3. 测试：在每个基站上分别测试全局模型
    print("\n在各基站上测试中心化模型...")
    station_test_data = processor.prepare_test_data_per_station(df_stations, station_info)

    station_results = {}
    all_metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': []}

    for station_id, test_data in tqdm(station_test_data.items(), desc="测试基站"):
        predictions, targets = trainer.predict_for_station(
            test_data['X'], test_data['y'], test_data['norm_params']
        )

        metrics = calculate_metrics(predictions, targets)

        station_results[station_id] = {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'coordinates': coordinates[station_id]
        }

        for metric_name, value in metrics.items():
            all_metrics[metric_name].append(value)

    # 4. 结果报告
    print("\n" + "=" * 60)
    print("中心化LSTM Baseline 测试结果")
    print("=" * 60)

    print(f"成功测试的基站数量: {len(station_results)}")
    print(f"模型参数量: {total_params:,}")

    print("\n全局平均指标:")
    for metric_name, values in all_metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  平均{metric_name}: {mean_val:.6f} (±{std_val:.6f})")

    # 最佳和最差基站
    if all_metrics['RMSE']:
        rmse_values = all_metrics['RMSE']
        best_idx = np.argmin(rmse_values)
        worst_idx = np.argmax(rmse_values)
        stations = list(station_results.keys())

        print(f"\n最佳基站: {stations[best_idx]} (RMSE: {rmse_values[best_idx]:.6f})")
        print(f"最差基站: {stations[worst_idx]} (RMSE: {rmse_values[worst_idx]:.6f})")

    # 5. 保存结果和可视化
    if args.save_results:
        os.makedirs('results', exist_ok=True)

        # 保存详细结果
        results_to_save = {}
        for station_id, result in station_results.items():
            results_to_save[str(station_id)] = {
                'metrics': result['metrics'],
                'coordinates': result['coordinates']
            }

        with open('results/centralized_lstm_detailed.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        # 保存汇总结果
        summary = {
            'model_params': total_params,
            'total_stations': len(station_results),
            'avg_metrics': {
                metric_name: {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
                for metric_name, values in all_metrics.items() if values
            }
        }

        with open('results/centralized_lstm_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 绘制和保存图表
        plot_training_history(trainer, 'results/centralized_lstm_training.png')
        plot_sample_predictions(station_results, save_path='results/centralized_lstm_predictions.png')

        print(f"\n结果已保存至 results/ 目录")
    else:
        plot_training_history(trainer)
        plot_sample_predictions(station_results)

    print(f"\n中心化LSTM Baseline完成!")
    print(f"最终平均RMSE: {np.mean(all_metrics['RMSE']):.6f}")


if __name__ == "__main__":
    main()