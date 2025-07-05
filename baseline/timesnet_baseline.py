#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimesNet中心化时序预测Baseline
与联邦学习TimeLLM进行对比的TimesNet中心化方法
"""

import os
import sys
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import random
import argparse
import json
from pathlib import Path
import math


# ===== TimesNet相关组件 =====

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """TimesNet模型"""

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = getattr(configs, 'label_len', 0)
        self.pred_len = configs.pred_len

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return None


# ===== 数据处理和训练组件 =====

class TimesNetConfig:
    """TimesNet配置类"""

    def __init__(self, args):
        # 任务配置
        self.task_name = 'long_term_forecast'
        self.seq_len = args.seq_len
        self.label_len = 0
        self.pred_len = args.pred_len

        # 模型配置
        self.d_model = args.d_model
        self.d_ff = args.d_model * 4
        self.e_layers = args.e_layers
        self.top_k = args.top_k
        self.num_kernels = args.num_kernels

        # 数据配置
        self.enc_in = 1  # 单变量时序
        self.c_out = 1  # 输出维度
        self.embed = 'timeF'
        self.freq = 'h'

        # 训练配置
        self.dropout = args.dropout


class CentralizedTimesNetDataProcessor:
    """中心化TimesNet数据处理器"""

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

        # 选择基站
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
        """准备中心化训练数据"""
        print("准备TimesNet中心化训练数据...")

        all_sequences_X = []
        all_sequences_y = []
        all_sequences_X_mark = []
        all_sequences_y_mark = []
        station_info = []

        # 为每个基站创建时序序列
        for station_id in df_stations.columns:
            station_data = df_stations[station_id]

            # 检查数据质量
            if station_data.std() == 0 or station_data.isna().all():
                print(f"基站 {station_id}: 数据质量差，跳过")
                continue

            # 标准化
            mean = station_data.mean()
            std = station_data.std()
            if std == 0:
                std = 1.0

            normalized_data = (station_data - mean) / std

            # 创建时序序列
            X, y, X_mark, y_mark = self._create_timesnet_sequences(normalized_data, df_stations.index)

            if len(X) > 0:
                all_sequences_X.append(X)
                all_sequences_y.append(y)
                all_sequences_X_mark.append(X_mark)
                all_sequences_y_mark.append(y_mark)

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
        combined_X_mark = np.concatenate(all_sequences_X_mark, axis=0)
        combined_y_mark = np.concatenate(all_sequences_y_mark, axis=0)

        print(f"合并后训练数据形状:")
        print(f"  X: {combined_X.shape}, y: {combined_y.shape}")
        print(f"  X_mark: {combined_X_mark.shape}, y_mark: {combined_y_mark.shape}")
        print(f"有效基站数量: {len(station_info)}")

        return combined_X, combined_y, combined_X_mark, combined_y_mark, station_info

    def _create_timesnet_sequences(self, data, time_index):
        """创建TimesNet格式的时序序列"""
        X, y = [], []
        X_mark, y_mark = [], []

        for i in range(self.seq_len, len(data) - self.pred_len + 1):
            # 历史序列和未来序列
            X.append(data.iloc[i - self.seq_len:i].values.reshape(-1, 1))
            y.append(data.iloc[i:i + self.pred_len].values.reshape(-1, 1))

            # 时间特征
            hist_times = time_index[i - self.seq_len:i]
            future_times = time_index[i:i + self.pred_len]

            X_mark.append(self._extract_time_features(hist_times))
            y_mark.append(self._extract_time_features(future_times))

        return np.array(X), np.array(y), np.array(X_mark), np.array(y_mark)

    def _extract_time_features(self, time_index):
        """提取时间特征 [month, day, weekday, hour]"""
        features = []
        for t in time_index:
            features.append([
                t.month - 1,  # 0-11
                t.day - 1,  # 0-30
                t.weekday(),  # 0-6
                t.hour  # 0-23
            ])
        return np.array(features)

    def prepare_test_data_per_station(self, df_stations, station_info):
        """为每个基站准备独立的测试数据"""
        station_test_data = {}

        for info in station_info:
            station_id = info['station_id']
            station_data = df_stations[station_id]
            time_index = df_stations.index

            # 使用训练时的标准化参数
            normalized_data = (station_data - info['mean']) / info['std']

            # 创建时序序列
            X, y, X_mark, y_mark = self._create_timesnet_sequences(normalized_data, time_index)

            if len(X) > 0:
                # 只取测试集部分
                test_samples = self.test_days * 24
                if len(X) >= test_samples:
                    X_test = X[-test_samples:]
                    y_test = y[-test_samples:]
                    X_mark_test = X_mark[-test_samples:]
                    y_mark_test = y_mark[-test_samples:]

                    station_test_data[station_id] = {
                        'X': X_test,
                        'y': y_test,
                        'X_mark': X_mark_test,
                        'y_mark': y_mark_test,
                        'norm_params': {'mean': info['mean'], 'std': info['std']}
                    }

        print(f"准备了 {len(station_test_data)} 个基站的测试数据")
        return station_test_data

    def split_train_val(self, X, y, X_mark, y_mark):
        """分割训练和验证集"""
        val_samples = int(len(X) * 0.1)

        X_train = X[:-val_samples] if val_samples > 0 else X
        y_train = y[:-val_samples] if val_samples > 0 else y
        X_mark_train = X_mark[:-val_samples] if val_samples > 0 else X_mark
        y_mark_train = y_mark[:-val_samples] if val_samples > 0 else y_mark

        X_val = X[-val_samples:] if val_samples > 0 else None
        y_val = y[-val_samples:] if val_samples > 0 else None
        X_mark_val = X_mark[-val_samples:] if val_samples > 0 else None
        y_mark_val = y_mark[-val_samples:] if val_samples > 0 else None

        print(f"训练集: {len(X_train)} 样本")
        if X_val is not None:
            print(f"验证集: {len(X_val)} 样本")

        return (X_train, y_train, X_mark_train, y_mark_train), (X_val, y_val, X_mark_val, y_mark_val)


class TimesNetTrainer:
    """TimesNet训练器"""

    def __init__(self, model, device='cuda', lr=0.0001, weight_decay=1e-4):
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

        for batch_data in train_loader:
            X_batch, y_batch, X_mark_batch, y_mark_batch = batch_data

            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()
            X_mark_batch = X_mark_batch.to(self.device).float()
            y_mark_batch = y_mark_batch.to(self.device).float()

            self.optimizer.zero_grad()

            # TimesNet前向传播
            predictions = self.model(X_batch, X_mark_batch, None, y_mark_batch)
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
            for batch_data in val_loader:
                X_batch, y_batch, X_mark_batch, y_mark_batch = batch_data

                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()
                X_mark_batch = X_mark_batch.to(self.device).float()
                y_mark_batch = y_mark_batch.to(self.device).float()

                predictions = self.model(X_batch, X_mark_batch, None, y_mark_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader=None, epochs=100, patience=20):
        """完整训练流程"""
        print("开始训练TimesNet模型...")

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

    def predict_for_station(self, X_test, y_test, X_mark_test, y_mark_test, norm_params):
        """为单个基站进行预测"""
        self.model.eval()

        # 创建测试数据加载器
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test),
            torch.FloatTensor(X_mark_test),
            torch.FloatTensor(y_mark_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in test_loader:
                X_batch, y_batch, X_mark_batch, y_mark_batch = batch_data

                X_batch = X_batch.to(self.device).float()
                X_mark_batch = X_mark_batch.to(self.device).float()
                y_mark_batch = y_mark_batch.to(self.device).float()

                predictions = self.model(X_batch, X_mark_batch, None, y_mark_batch)

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
    plt.title('TimesNet训练历史')
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
    parser = argparse.ArgumentParser(description='TimesNet中心化时序预测Baseline')

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

    # TimesNet模型参数
    parser.add_argument('--d_model', type=int, default=64,
                        help='模型隐藏维度')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='TimesNet层数')
    parser.add_argument('--top_k', type=int, default=5,
                        help='FFT频域Top-K')
    parser.add_argument('--num_kernels', type=int, default=6,
                        help='Inception卷积核数量')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
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
    print("TimesNet中心化时序预测Baseline")
    print("用于与联邦学习TimeLLM进行性能对比")
    print("=" * 60)

    # 检查数据文件
    if not os.path.exists(args.data_file):
        print(f"错误: 数据文件不存在 {args.data_file}")
        return

    # 1. 数据准备
    processor = CentralizedTimesNetDataProcessor(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        test_days=args.test_days
    )

    # 加载数据
    df_stations, station_ids, coordinates = processor.load_and_prepare_data(
        args.data_file, args.num_stations
    )

    # 准备中心化训练数据
    X_combined, y_combined, X_mark_combined, y_mark_combined, station_info = processor.prepare_centralized_training_data(
        df_stations)

    # 分割训练和验证集
    (X_train, y_train, X_mark_train, y_mark_train), (X_val, y_val, X_mark_val, y_mark_val) = processor.split_train_val(
        X_combined, y_combined, X_mark_combined, y_mark_combined
    )

    # 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(X_mark_train),
        torch.FloatTensor(y_mark_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if X_val is not None:
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(X_mark_val),
            torch.FloatTensor(y_mark_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. 模型创建和训练
    print("\n创建TimesNet模型...")
    config = TimesNetConfig(args)
    model = TimesNet(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print(f"模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  e_layers: {config.e_layers}")
    print(f"  top_k: {config.top_k}")
    print(f"  num_kernels: {config.num_kernels}")

    trainer = TimesNetTrainer(
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
    print("\n在各基站上测试TimesNet模型...")
    station_test_data = processor.prepare_test_data_per_station(df_stations, station_info)

    station_results = {}
    all_metrics = {'MSE': [], 'RMSE': [], 'MAE': [], 'MAPE': []}

    for station_id, test_data in tqdm(station_test_data.items(), desc="测试基站"):
        predictions, targets = trainer.predict_for_station(
            test_data['X'],
            test_data['y'],
            test_data['X_mark'],
            test_data['y_mark'],
            test_data['norm_params']
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
    print("TimesNet Baseline 测试结果")
    print("=" * 60)

    print(f"成功测试的基站数量: {len(station_results)}")
    print(f"模型参数量: {total_params:,}")

    print(f"\n模型配置:")
    print(f"  序列长度: {args.seq_len}")
    print(f"  预测长度: {args.pred_len}")
    print(f"  模型维度: {args.d_model}")
    print(f"  TimesNet层数: {args.e_layers}")
    print(f"  FFT Top-K: {args.top_k}")

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

        with open('results/timesnet_baseline_detailed.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)

        # 保存汇总结果
        summary = {
            'model_type': 'TimesNet',
            'model_params': total_params,
            'model_config': {
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'd_model': args.d_model,
                'e_layers': args.e_layers,
                'top_k': args.top_k,
                'num_kernels': args.num_kernels
            },
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

        with open('results/timesnet_baseline_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 绘制和保存图表
        plot_training_history(trainer, 'results/timesnet_baseline_training.png')
        plot_sample_predictions(station_results, save_path='results/timesnet_baseline_predictions.png')

        print(f"\n结果已保存至 results/ 目录")
        print(f"详细结果: results/timesnet_baseline_detailed.json")
        print(f"汇总结果: results/timesnet_baseline_summary.json")
    else:
        plot_training_history(trainer)
        plot_sample_predictions(station_results)

    print(f"\nTimesNet Baseline完成!")
    print(f"最终平均RMSE: {np.mean(all_metrics['RMSE']):.6f}")
    print(f"相比LSTM，TimesNet能够捕获数据中的多周期性特征")


if __name__ == "__main__":
    main()