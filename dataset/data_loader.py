# -*- coding: utf-8 -*-
"""
联邦学习数据加载和预处理模块
"""

import h5py
import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import logging


class FederatedDataLoader:
    """联邦学习数据加载器"""

    def __init__(self, args):
        self.args = args
        self.selected_cells = None
        self.cell_coordinates = None
        self.logger = logging.getLogger(__name__)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        从HDF5文件加载数据

        Returns:
            normalized_df: 标准化后的流量数据
            original_df: 原始流量数据
            coordinates: 基站坐标信息
        """
        self.logger.info("开始加载数据...")

        # 设置随机种子
        self._set_seed()

        # 读取HDF5文件
        file_path = os.path.join(self.args.dataset_dir, self.args.file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        with h5py.File(file_path, 'r') as f:
            self.logger.info(f"HDF5文件字段: {list(f.keys())}")

            idx = f['idx'][()]  # 时间戳
            cell = f['cell'][()]  # 基站ID
            lng = f['lng'][()]  # 经度
            lat = f['lat'][()]  # 纬度
            traffic_data = f[self.args.data_type][()][:, cell - 1]  # 流量数据

        # 构建DataFrame
        df = pd.DataFrame(
            traffic_data,
            index=pd.to_datetime(idx.ravel(), unit='s'),
            columns=cell
        )
        df.fillna(0, inplace=True)

        self.logger.info(f"原始数据形状: {df.shape}")
        self.logger.info(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        # 随机选择基站
        self._select_cells(cell, lng, lat)

        # 筛选选中基站的数据
        df_selected = df[self.selected_cells].copy()

        self.logger.info(f"选中基站数量: {len(self.selected_cells)}")
        self.logger.info(f"选中基站ID前5个: {self.selected_cells[:5]}")

        # 数据标准化
        normalized_df = self._normalize_data(df_selected)

        return normalized_df, df_selected, self.cell_coordinates

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)

    def _select_cells(self, cell, lng, lat):
        """选择基站并获取坐标"""
        cell_pool = list(cell)
        self.selected_cells = sorted(random.sample(cell_pool, self.args.num_clients))
        selected_cells_idx = np.where(np.isin(cell_pool, self.selected_cells))[0]

        # 获取选中基站的坐标
        self.cell_coordinates = {
            cell_id: {'lng': lng[idx], 'lat': lat[idx]}
            for cell_id, idx in zip(self.selected_cells, selected_cells_idx)
        }

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score标准化"""
        self.logger.info("执行Z-score标准化...")

        # 基于训练集计算标准化参数
        train_data = df.iloc[:-self.args.test_days * 24]
        mean = train_data.mean()
        std = train_data.std()

        # 避免除零错误
        std = std.replace(0, 1)

        # 标准化全部数据
        normalized_df = (df - mean) / std

        # 保存标准化参数供后续使用
        self.norm_params = {'mean': mean, 'std': std}

        self.logger.info("标准化完成")
        return normalized_df

    def create_sequences_for_cell(self, cell_data: pd.Series) -> Dict:
        """
        为单个基站创建时序序列（TimeLLM风格）

        Args:
            cell_data: 单个基站的时序数据

        Returns:
            sequences: 包含训练、验证、测试序列的字典
        """
        history_sequences = []  # 历史序列
        target_sequences = []  # 目标序列

        # 生成滑动窗口序列
        for idx in range(self.args.seq_len, len(cell_data) - self.args.pred_len + 1):
            # 历史序列：前seq_len个时间点
            history = cell_data.iloc[idx - self.args.seq_len:idx].values
            history_sequences.append(history)

            # 目标序列：接下来pred_len个时间点
            target = cell_data.iloc[idx:idx + self.args.pred_len].values
            target_sequences.append(target)

        # 转换为numpy数组
        history_array = np.array(history_sequences)  # shape: (num_samples, seq_len)
        target_array = np.array(target_sequences)  # shape: (num_samples, pred_len)

        # 数据分割
        test_len = self.args.test_days * 24
        val_len = self.args.val_days * 24
        train_len = len(history_array) - test_len - val_len

        sequences = {
            'train': {
                'history': history_array[:train_len],
                'target': target_array[:train_len]
            },
            'test': {
                'history': history_array[-test_len:],
                'target': target_array[-test_len:]
            }
        }

        # 如果有验证集
        if val_len > 0:
            sequences['val'] = {
                'history': history_array[train_len:train_len + val_len],
                'target': target_array[train_len:train_len + val_len]
            }

        return sequences

    def prepare_federated_data(self, normalized_df: pd.DataFrame) -> Dict:
        """
        为联邦学习准备数据

        Args:
            normalized_df: 标准化后的流量数据

        Returns:
            federated_data: 联邦学习数据
        """
        self.logger.info("准备联邦学习数据...")

        federated_data = {
            'clients': {},
            'coordinates': self.cell_coordinates,
            'metadata': {
                'num_clients': len(self.selected_cells),
                'client_ids': self.selected_cells,
                'norm_params': self.norm_params
            }
        }

        # 为每个客户端（基站）准备时序数据
        for cell_id in self.selected_cells:
            cell_data = normalized_df[cell_id]
            sequences = self.create_sequences_for_cell(cell_data)

            federated_data['clients'][cell_id] = {
                'sequences': sequences,
                'coordinates': self.cell_coordinates[cell_id],
                'data_stats': {
                    'train_samples': len(sequences['train']['history']),
                    'test_samples': len(sequences['test']['history']),
                    'val_samples': len(sequences['val']['history']) if 'val' in sequences else 0
                }
            }

        self.logger.info("联邦数据准备完成")
        return federated_data

    def create_data_loaders(self, sequences: Dict, batch_size: int = None) -> Dict:
        """
        创建PyTorch数据加载器

        Args:
            sequences: 时序序列数据
            batch_size: 批处理大小

        Returns:
            data_loaders: 数据加载器字典
        """
        if batch_size is None:
            batch_size = self.args.local_bs

        data_loaders = {}

        for split in ['train', 'val', 'test']:
            if split in sequences:
                X = torch.FloatTensor(sequences[split]['history'])
                y = torch.FloatTensor(sequences[split]['target'])

                dataset = TensorDataset(X, y)

                # 训练集需要shuffle，测试集不需要
                shuffle = (split == 'train')
                drop_last = (split == 'train')

                data_loaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=0  # 避免多进程问题
                )

        return data_loaders


def get_federated_data(args):
    """
    获取联邦学习数据的便捷函数

    Args:
        args: 配置参数

    Returns:
        federated_data: 联邦学习数据
        data_loader: 数据加载器实例
    """
    # 创建数据加载器
    data_loader = FederatedDataLoader(args)

    # 加载数据
    normalized_df, original_df, coordinates = data_loader.load_data()

    # 准备联邦数据
    federated_data = data_loader.prepare_federated_data(normalized_df)

    return federated_data, data_loader