# -*- coding: utf-8 -*-
"""
联邦学习数据加载和预处理模块 - 添加原始流量统计
"""
import copy

import h5py
import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, List
import logging


class DataAugmentationManager:
    """数据增强管理器"""

    def __init__(self, similarity_threshold=0.6, candidate_pool_size=5,
                 lambda_min=0.6, lambda_max=0.8):
        self.similarity_threshold = similarity_threshold
        self.candidate_pool_size = candidate_pool_size
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # 约束配置
        self.constraint_config = {
            'enable_statistical_constraint': True,  # 统计约束
            'enable_range_constraint': True,  # 范围约束
            'enable_trend_constraint': True,  # 趋势约束
            'max_deviation_ratio': 0.3,  # 最大偏离比例
            'min_correlation_threshold': 0.5  # 最小相关性阈值
        }

    def apply_regularization_constraints(self, enhanced_seq, enhanced_target,
                                         original_seq, original_target):
        """应用正则化约束机制"""
        constrained_seq = enhanced_seq.copy()
        constrained_target = enhanced_target.copy()

        # 约束1: 统计特征约束
        if self.constraint_config['enable_statistical_constraint']:
            constrained_seq = self._apply_statistical_constraint(
                constrained_seq, original_seq
            )
            constrained_target = self._apply_statistical_constraint(
                constrained_target, original_target
            )

        # 约束2: 数值范围约束
        if self.constraint_config['enable_range_constraint']:
            constrained_seq = self._apply_range_constraint(
                constrained_seq, original_seq
            )
            constrained_target = self._apply_range_constraint(
                constrained_target, original_target
            )

        # 约束3: 时序相关性约束
        if self.constraint_config['enable_trend_constraint']:
            constrained_seq = self._apply_correlation_constraint(
                constrained_seq, original_seq
            )

        return constrained_seq, constrained_target

    def _apply_statistical_constraint(self, enhanced_data, original_data):
        """统计特征约束：确保均值和标准差在合理范围内"""
        orig_mean = np.mean(original_data)
        orig_std = np.std(original_data)

        enhanced_mean = np.mean(enhanced_data)
        enhanced_std = np.std(enhanced_data)

        max_deviation = self.constraint_config['max_deviation_ratio']

        # 约束均值偏离
        if abs(enhanced_mean - orig_mean) > max_deviation * orig_mean:
            # 软约束：线性调整到合理范围
            target_mean = orig_mean * (1 + max_deviation * np.sign(enhanced_mean - orig_mean))
            enhanced_data = enhanced_data - enhanced_mean + target_mean

        # 约束标准差偏离
        current_std = np.std(enhanced_data)
        if abs(current_std - orig_std) > max_deviation * orig_std:
            target_std = orig_std * (1 + max_deviation)
            if current_std > 0:
                enhanced_data = (enhanced_data - np.mean(enhanced_data)) * (target_std / current_std) + np.mean(
                    enhanced_data)

        return enhanced_data

    def _apply_range_constraint(self, enhanced_data, original_data):
        """数值范围约束：确保数据在合理的物理范围内"""
        orig_min = np.min(original_data)
        orig_max = np.max(original_data)

        # 计算合理的扩展范围
        data_range = orig_max - orig_min
        extended_min = max(0, orig_min - 0.2 * data_range)  # 确保非负
        extended_max = orig_max + 0.2 * data_range

        # 应用软裁剪（使用tanh函数平滑约束）
        enhanced_data = np.clip(enhanced_data, extended_min, extended_max)

        return enhanced_data

    def _apply_correlation_constraint(self, enhanced_seq, original_seq):
        """时序相关性约束：确保时序特征不被破坏"""
        # 计算与原始序列的相关性
        correlation = np.corrcoef(enhanced_seq.flatten(), original_seq.flatten())[0, 1]

        if np.isnan(correlation) or correlation < self.constraint_config['min_correlation_threshold']:
            # 如果相关性太低，使用加权平均进行修正
            correction_weight = 0.3  # 30%原始数据用于修正
            enhanced_seq = (1 - correction_weight) * enhanced_seq + correction_weight * original_seq

        return enhanced_seq

    def apply_periodic_mixup(self, seq, target, historical_data, current_idx):
        """时间周期性Mixup（集成约束机制）"""
        candidate = self.find_similar_candidates(seq, historical_data, current_idx)

        if candidate is not None:
            lambda_val = random.uniform(self.lambda_min, self.lambda_max)
            enhanced_seq = lambda_val * seq + (1 - lambda_val) * candidate
            enhanced_target = target.copy()  # target保持不变

            # 应用正则化约束
            enhanced_seq, enhanced_target = self.apply_regularization_constraints(
                enhanced_seq, enhanced_target, seq, target
            )

            return enhanced_seq, enhanced_target

        return seq, target

    def apply_jittering(self, seq, noise_level=0.03):
        """添加高斯噪声（集成约束机制）"""
        noise = np.random.normal(0, noise_level * np.std(seq), seq.shape)
        enhanced_seq = seq + noise

        # 应用范围约束
        enhanced_seq = self._apply_range_constraint(enhanced_seq, seq)

        return enhanced_seq

    def apply_scaling(self, seq, target, scale_range=(0.85, 1.15)):
        """随机缩放（集成约束机制）"""
        scale_factor = random.uniform(*scale_range)
        enhanced_seq = seq * scale_factor
        enhanced_target = target * scale_factor

        # 应用统计约束
        enhanced_seq = self._apply_statistical_constraint(enhanced_seq, seq)
        enhanced_target = self._apply_statistical_constraint(enhanced_target, target)

        return enhanced_seq, enhanced_target

    def find_similar_candidates(self, target_sequence, historical_data, current_idx):
        """找到相似的候选序列"""
        if len(historical_data) < 5:  # 历史数据不足
            return None

        similarities = []
        target_flat = target_sequence.flatten()

        # 在历史数据中寻找相似序列
        history_window = max(0, current_idx - 50)  # 最近50个样本作为候选池
        search_end = min(current_idx, len(historical_data))

        for i in range(history_window, search_end):
            if i == current_idx:  # 跳过自己
                continue

            hist_seq = historical_data[i]
            hist_flat = hist_seq.flatten()

            # 计算皮尔逊相关系数
            if len(hist_flat) == len(target_flat):
                corr = np.corrcoef(target_flat, hist_flat)[0, 1]

                if not np.isnan(corr) and corr > self.similarity_threshold:
                    similarities.append((i, corr, hist_seq))

        if not similarities:
            return None

        # 按相似性排序，取前N个作为候选池
        similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = similarities[:self.candidate_pool_size]

        # 从候选池中随机选择一个
        _, _, selected_candidate = random.choice(candidates)
        return selected_candidate


class FederatedDataLoader:
    """联邦学习数据加载器"""
    def __init__(self, args):
        self.args = args
        self.selected_cells = None
        self.cell_coordinates = None
        self.original_traffic_stats = None  # 新增：原始流量统计
        self.logger = logging.getLogger(__name__)

        # 添加数据增强配置
        self.augmentation_config = {
            'enable_augmentation': getattr(args, 'enable_augmentation', False),
            'mixup_prob': getattr(args, 'mixup_prob', 0.2),
            'jittering_prob': getattr(args, 'jittering_prob', 0.15),
            'scaling_prob': getattr(args, 'scaling_prob', 0.1),
            'augmentation_ratio': getattr(args, 'augmentation_ratio', 0.3),
            'similarity_threshold': getattr(args, 'similarity_threshold', 0.6),
            'candidate_pool_size': getattr(args, 'candidate_pool_size', 5),
            'lambda_min': getattr(args, 'augmentation_lambda_min', 0.6),
            'lambda_max': getattr(args, 'augmentation_lambda_max', 0.8),
            # 直接在这里添加约束配置，避免后续update
            'enable_regularization_constraints': getattr(args, 'enable_regularization_constraints', True),
            'max_deviation_ratio': getattr(args, 'max_deviation_ratio', 0.3),
            'min_correlation_threshold': getattr(args, 'min_correlation_threshold', 0.5),
            'constraint_correction_weight': getattr(args, 'constraint_correction_weight', 0.3)
        }

        # 初始化数据增强管理器（只创建一次）
        if self.augmentation_config['enable_augmentation']:
            self.aug_manager = DataAugmentationManager(
                similarity_threshold=self.augmentation_config['similarity_threshold'],
                candidate_pool_size=self.augmentation_config['candidate_pool_size'],
                lambda_min=self.augmentation_config['lambda_min'],
                lambda_max=self.augmentation_config['lambda_max']
            )

            # 更新约束配置
            self.aug_manager.constraint_config.update({
                'max_deviation_ratio': self.augmentation_config['max_deviation_ratio'],
                'min_correlation_threshold': self.augmentation_config['min_correlation_threshold'],
                'enable_statistical_constraint': self.augmentation_config['enable_regularization_constraints'],
                'enable_range_constraint': self.augmentation_config['enable_regularization_constraints'],
                'enable_trend_constraint': self.augmentation_config['enable_regularization_constraints']
            })

            self.logger.info("数据增强功能已启用（包含正则化约束）")
        else:
            self.aug_manager = None

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

        # 计算原始流量统计（在标准化之前）
        self._calculate_original_traffic_stats(df_selected)

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
            cell_id: {'lng': float(lng[idx]), 'lat': float(lat[idx])}
            for cell_id, idx in zip(self.selected_cells, selected_cells_idx)
        }

    def _calculate_original_traffic_stats(self, df: pd.DataFrame):
        """计算原始流量统计信息"""
        self.logger.info("计算原始流量统计...")

        self.original_traffic_stats = {}

        for cell_id in self.selected_cells:
            cell_data = df[cell_id]

            # 基本统计量
            mean_traffic = float(cell_data.mean())
            std_traffic = float(cell_data.std())
            min_traffic = float(cell_data.min())
            max_traffic = float(cell_data.max())
            median_traffic = float(cell_data.median())

            # 计算趋势 - 使用线性回归斜率
            time_index = np.arange(len(cell_data))
            if len(cell_data) > 1:
                slope, _ = np.polyfit(time_index, cell_data.values, 1)
                if slope > 0.01:  # 阈值可以调整
                    trend = 'increasing'
                elif slope < -0.01:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
                slope = 0.0

            # 计算最近期的趋势（最后30%的数据）
            recent_window = max(1, int(len(cell_data) * 0.3))
            recent_data = cell_data.iloc[-recent_window:]
            recent_mean = float(recent_data.mean())

            # 计算波动性（变异系数）
            cv = std_traffic / mean_traffic if mean_traffic > 0 else 0.0

            # 计算峰值特征
            q75 = float(cell_data.quantile(0.75))
            q25 = float(cell_data.quantile(0.25))

            self.original_traffic_stats[cell_id] = {
                'mean': mean_traffic,
                'std': std_traffic,
                'min': min_traffic,
                'max': max_traffic,
                'median': median_traffic,
                'trend': trend,
                'trend_slope': float(slope),
                'recent_mean': recent_mean,
                'coefficient_of_variation': float(cv),
                'q25': q25,
                'q75': q75,
                'iqr': q75 - q25,
                'data_points': len(cell_data)
            }

        self.logger.info("原始流量统计计算完成")

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
        为单个基站创建时序序列（集成数据增强）
        """
        # 先创建原始序列
        original_sequences = self._create_basic_sequences(cell_data)

        # 如果启用数据增强，应用增强策略
        if (self.augmentation_config['enable_augmentation'] and
                self.aug_manager is not None):
            augmented_sequences = self._apply_augmentation_strategies(
                original_sequences, cell_data
            )
            # 合并原始和增强序列
            merged_sequences = self._merge_sequences(original_sequences, augmented_sequences)
            self.logger.info(f"数据增强完成，训练样本从 {len(original_sequences['train']['history'])} "
                             f"增加到 {len(merged_sequences['train']['history'])}")
            return merged_sequences

        return original_sequences

    def _create_basic_sequences(self, cell_data: pd.Series) -> Dict:
        """创建基础时序序列（原create_sequences_for_cell的逻辑）"""
        history_sequences = []
        target_sequences = []

        # 生成滑动窗口序列
        for idx in range(self.args.seq_len, len(cell_data) - self.args.pred_len + 1):
            history = cell_data.iloc[idx - self.args.seq_len:idx].values
            history_sequences.append(history)
            target = cell_data.iloc[idx:idx + self.args.pred_len].values
            target_sequences.append(target)

        # 转换为numpy数组
        history_array = np.array(history_sequences)
        target_array = np.array(target_sequences)

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

        if val_len > 0:
            sequences['val'] = {
                'history': history_array[train_len:train_len + val_len],
                'target': target_array[train_len:train_len + val_len]
            }

        return sequences

    def _apply_augmentation_strategies(self, sequences, raw_data):
        """应用多种增强策略"""
        augmented_data = {'train': {'history': [], 'target': []}}

        original_history = sequences['train']['history']
        original_target = sequences['train']['target']
        raw_values = raw_data.values  # 原始时序数据

        # 计算需要增强的样本数量
        num_to_augment = int(len(original_history) * self.augmentation_config['augmentation_ratio'])
        augment_indices = random.sample(range(len(original_history)), num_to_augment)

        self.logger.info(f"对 {num_to_augment}/{len(original_history)} 个样本应用数据增强")

        for i in augment_indices:
            current_seq = original_history[i].copy()
            current_target = original_target[i].copy()
            enhanced_seq = current_seq.copy()
            enhanced_target = current_target.copy()

            augmentation_applied = []

            # 策略1: 时间周期性Mixup
            if random.random() < self.augmentation_config['mixup_prob']:
                enhanced_seq, enhanced_target = self.aug_manager.apply_periodic_mixup(
                    enhanced_seq, enhanced_target, original_history, i
                )
                augmentation_applied.append("Mixup")

            # 策略2: Jittering（时序抖动）
            if random.random() < self.augmentation_config['jittering_prob']:
                enhanced_seq = self.aug_manager.apply_jittering(enhanced_seq)
                augmentation_applied.append("Jittering")

            # 策略3: Scaling（幅度缩放）
            if random.random() < self.augmentation_config['scaling_prob']:
                enhanced_seq, enhanced_target = self.aug_manager.apply_scaling(
                    enhanced_seq, enhanced_target
                )
                augmentation_applied.append("Scaling")

            # 只有当至少应用了一种增强策略时才添加
            if augmentation_applied:
                augmented_data['train']['history'].append(enhanced_seq)
                augmented_data['train']['target'].append(enhanced_target)

        # 转换为numpy数组
        if augmented_data['train']['history']:
            augmented_data['train']['history'] = np.array(augmented_data['train']['history'])
            augmented_data['train']['target'] = np.array(augmented_data['train']['target'])

        return augmented_data

    def _merge_sequences(self, original_sequences, augmented_sequences):
        """合并原始和增强序列"""
        merged = copy.deepcopy(original_sequences)

        # 合并训练数据
        if (augmented_sequences['train']['history'] is not None and
                len(augmented_sequences['train']['history']) > 0):
            merged['train']['history'] = np.concatenate([
                original_sequences['train']['history'],
                augmented_sequences['train']['history']
            ], axis=0)

            merged['train']['target'] = np.concatenate([
                original_sequences['train']['target'],
                augmented_sequences['train']['target']
            ], axis=0)

        return merged

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
            'original_traffic_stats': self.original_traffic_stats,  # 新增
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
                'original_traffic_stats': self.original_traffic_stats[cell_id],  # 新增
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

                # 检查是否为TimeLLM模型，需要不同的数据格式
                if hasattr(self.args, 'model_type') and self.args.model_type == 'timellm':
                    # TimeLLM需要的数据格式：(batch_size, seq_len, features)
                    X = X.unsqueeze(-1)  # 添加特征维度
                    y = y.unsqueeze(-1)  # 添加特征维度

                    batch_size_data, seq_len, _ = X.shape
                    pred_len = y.shape[1]

                    # 创建时间特征标记（简化版）
                    # 这里创建基本的时间特征，实际使用时可以根据真实时间戳生成
                    x_mark = torch.zeros(batch_size_data, seq_len, 4)  # [month, day, weekday, hour]
                    y_mark = torch.zeros(batch_size_data, pred_len, 4)

                    dataset = TensorDataset(X, y, x_mark, y_mark)
                else:
                    # 原有格式，用于其他模型
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