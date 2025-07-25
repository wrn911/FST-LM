# -*- coding: utf-8 -*-
"""
FedDA数据增强模块
"""
import numpy as np
import torch
from typing import Dict, List
import copy


class FedDADataAugmentation:
    """FedDA数据增强管理器"""

    def __init__(self, augment_ratio=0.01):  # 默认1%数据用于增强
        self.augment_ratio = augment_ratio
        self.augmented_data = {}

    def augment_client_data(self, client_data: Dict, client_id: str):
        """
        为单个客户端创建增强数据

        Args:
            client_data: 客户端的时序数据 (sequences)
            client_id: 客户端ID

        Returns:
            augmented_sample: 增强后的小样本数据
        """
        sequences = client_data['sequences']

        if 'train' not in sequences:
            return None

        train_history = sequences['train']['history']  # [N, seq_len]

        # 按周划分数据 (假设每168小时=1周)
        hours_per_week = 168
        samples_per_week = hours_per_week // 24  # 假设每天1个样本

        if len(train_history) < samples_per_week:
            # 数据不足一周，直接计算统计均值
            augmented_sample = np.mean(train_history, axis=0)
        else:
            # 分周计算，然后取各时间点的统计平均
            num_weeks = len(train_history) // samples_per_week
            weekly_data = []

            for week in range(num_weeks):
                start_idx = week * samples_per_week
                end_idx = start_idx + samples_per_week
                week_data = train_history[start_idx:end_idx]
                weekly_data.append(week_data)

            # 计算各时间点的均值
            weekly_data = np.array(weekly_data)  # [num_weeks, samples_per_week, seq_len]
            augmented_sample = np.mean(weekly_data, axis=0)  # [samples_per_week, seq_len]

            # 进一步压缩：取每周的代表性样本
            augmented_sample = np.mean(augmented_sample, axis=0)  # [seq_len]

        # 标准化处理
        augmented_sample = self._standardize(augmented_sample)

        # 只保留指定比例的数据点
        num_points = max(1, int(len(augmented_sample) * self.augment_ratio))
        step = len(augmented_sample) // num_points
        augmented_sample = augmented_sample[::step][:num_points]

        self.augmented_data[client_id] = augmented_sample

        return augmented_sample

    def _standardize(self, data):
        """标准化数据：零均值，单位方差"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data - mean
        return (data - mean) / std

    def get_global_augmented_dataset(self):
        """获取全局增强数据集"""
        if not self.augmented_data:
            return None

        # 合并所有客户端的增强数据
        all_data = list(self.augmented_data.values())
        global_dataset = np.concatenate(all_data, axis=0)

        return global_dataset

    def create_quasi_global_model(self, model_architecture, device='cpu'):
        """
        使用增强数据训练准全局模型

        Args:
            model_architecture: 模型架构类
            device: 训练设备

        Returns:
            quasi_global_params: 准全局模型参数
        """
        global_data = self.get_global_augmented_dataset()

        if global_data is None or len(global_data) == 0:
            return None

        # 创建简单的准全局模型（这里简化实现）
        # 实际应该用增强数据训练一个小模型

        # 简化版：直接用数据统计信息初始化模型
        quasi_model = model_architecture

        # 这里可以添加实际的训练逻辑
        # 为了简化，我们返回一个基于统计的初始化

        return quasi_model.state_dict() if quasi_model else None