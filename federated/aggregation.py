# -*- coding: utf-8 -*-
"""
联邦学习聚合算法 - 添加LoRA支持
"""

import torch
import copy
from typing import List, Dict


class FedAvgAggregator:
    """FedAvg聚合算法"""

    def __init__(self):
        pass

    def aggregate(self, client_models: List[Dict], client_weights: List[float] = None):
        """
        聚合客户端模型参数

        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        # 如果没有提供权重，则使用平均权重
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len  # -*- coding: utf-8 -*-


"""
联邦学习聚合算法 - 添加LoRA支持
"""

import torch
import copy
from typing import List, Dict


class FedAvgAggregator:
    """FedAvg聚合算法"""

    def __init__(self):
        pass

    def aggregate(self, client_models: List[Dict], client_weights: List[float] = None):
        """
        聚合客户端模型参数

        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        # 如果没有提供权重，则使用平均权重
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)

        # 确保权重总和为1
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # 初始化聚合模型参数
        aggregated_model = copy.deepcopy(client_models[0])

        # 将第一个模型参数乘以对应权重
        for key in aggregated_model.keys():
            aggregated_model[key] = aggregated_model[key] * client_weights[0]

        # 累加其他客户端的加权参数
        for i in range(1, len(client_models)):
            weight = client_weights[i]
            for key in aggregated_model.keys():
                aggregated_model[key] += client_models[i][key] * weight

        return aggregated_model


class WeightedFedAvgAggregator:
    """基于样本数量的加权FedAvg聚合算法"""

    def __init__(self):
        pass

    def aggregate(self, client_models: List[Dict], client_info: List[Dict]):
        """
        基于样本数量进行加权聚合

        Args:
            client_models: 客户端模型参数列表
            client_info: 客户端信息列表（包含样本数量）

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        # 计算基于样本数量的权重
        total_samples = sum([info['num_samples'] for info in client_info])
        client_weights = [info['num_samples'] / total_samples for info in client_info]

        # 使用FedAvg进行聚合
        fedavg_aggregator = FedAvgAggregator()
        return fedavg_aggregator.aggregate(client_models, client_weights)


class LoRAFedAvgAggregator:
    """专门用于LoRA参数的联邦聚合算法"""

    def __init__(self):
        pass

    def aggregate(self, client_models: List[Dict], client_weights: List[float] = None):
        """
        只聚合LoRA相关参数，保持基础模型参数不变

        Args:
            client_models: 客户端模型参数列表
            client_weights: 客户端权重列表

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        # 如果没有提供权重，则使用平均权重
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)

        # 确保权重总和为1
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # 识别LoRA参数和基础模型参数
        lora_keys = set()
        base_keys = set()

        for key in client_models[0].keys():
            if self._is_lora_param(key):
                lora_keys.add(key)
            else:
                base_keys.add(key)

        print(f"检测到 {len(lora_keys)} 个LoRA参数, {len(base_keys)} 个基础模型参数")

        # 初始化聚合模型
        aggregated_model = copy.deepcopy(client_models[0])

        # 只聚合LoRA参数
        for key in lora_keys:
            # 重置为零，然后累加加权参数
            aggregated_model[key] = torch.zeros_like(aggregated_model[key])
            for i, client_model in enumerate(client_models):
                aggregated_model[key] += client_model[key] * client_weights[i]

        # 基础模型参数保持第一个客户端的值（应该都是相同的冻结参数）
        for key in base_keys:
            aggregated_model[key] = client_models[0][key]

        return aggregated_model

    def _is_lora_param(self, param_name: str) -> bool:
        """判断参数是否为LoRA参数"""
        lora_keywords = ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']
        return any(keyword in param_name for keyword in lora_keywords)


def get_aggregator(aggregation_method: str, **kwargs):
    """
    获取聚合器

    Args:
        aggregation_method: 聚合方法名称
        **kwargs: 额外参数，用于LLM聚合器

    Returns:
        aggregator: 聚合器实例
    """
    if aggregation_method.lower() == 'fedavg':
        return FedAvgAggregator()
    elif aggregation_method.lower() == 'weighted':
        return WeightedFedAvgAggregator()
    elif aggregation_method.lower() == 'lora_fedavg':
        return LoRAFedAvgAggregator()
    elif aggregation_method.lower() == 'lora_fedprox':  # 新增: LoRA版本的FedProx
        return LoRAFedAvgAggregator()  # FedProx聚合阶段与LoRA FedAvg相同
    elif aggregation_method.lower() == 'fedprox':  # 保留非LoRA版本
        return FedAvgAggregator()
    elif aggregation_method.lower() == 'enhanced_multi_dim_llm':
        from .enhanced_multi_dimensional_llm_aggregator import EnhancedMultiDimensionalLLMAggregator

        # 添加新的参数传递
        aggregator_kwargs = kwargs.copy()
        aggregator_kwargs.update({
            'alpha_max': kwargs.get('alpha_max', 0.9),
            'alpha_min': kwargs.get('alpha_min', 0.2),
            'decay_type': kwargs.get('decay_type', 'sigmoid'),
            'base_constraint': kwargs.get('base_constraint', 0.25),
        })

        return EnhancedMultiDimensionalLLMAggregator(**aggregator_kwargs)
    else:
        raise ValueError(f"不支持的聚合方法: {aggregation_method}")