# -*- coding: utf-8 -*-
"""
联邦学习聚合算法 - 添加LoRA支持
"""
import numpy as np
import torch
import copy
from typing import List, Dict

from scipy import linalg


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


class FedAttAggregator:
    """FedAtt注意力聚合算法 - 支持LoRA"""

    def __init__(self, epsilon=1.0, is_lora_mode=False):
        self.epsilon = epsilon
        self.is_lora_mode = is_lora_mode

    def aggregate(self, client_models: List[Dict], client_weights: List[float] = None):
        if self.is_lora_mode:
            # 分类所有客户端参数
            all_lora = []
            all_timellm = []

            for model in client_models:
                lora_params, timellm_params, _ = self._classify_parameters(model)
                all_lora.append(lora_params)
                all_timellm.append(timellm_params)

            # 分别对LoRA和TimeLLM参数应用注意力聚合
            aggregated_lora = self._average_weights_att(all_lora, all_lora[0]) if all_lora and all_lora[0] else {}
            aggregated_timellm = self._average_weights_att(all_timellm, all_timellm[0]) if all_timellm and all_timellm[
                0] else {}

            # 重构完整模型
            return self._reconstruct_full_model(aggregated_lora, aggregated_timellm, client_models[0])
        else:
            # 标准模式：所有参数
            return self._average_weights_att(client_models, client_models[0])

    def _average_weights_att(self, w_clients, w_server):  # 移除epsilon参数
        """注意力加权聚合"""
        epsilon = self.epsilon  # 使用实例变量

        w_next = copy.deepcopy(w_server)
        att = {}
        for k in w_server.keys():
            w_next[k] = torch.zeros_like(w_server[k])
            att[k] = torch.zeros(len(w_clients))

        for k in w_next.keys():
            for i in range(0, len(w_clients)):
                att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server[k].cpu() - w_clients[i][k].cpu())))

        for k in w_next.keys():
            att[k] = torch.nn.functional.softmax(att[k], dim=0)

        for k in w_next.keys():
            att_weight = torch.zeros_like(w_server[k])
            for i in range(0, len(w_clients)):
                att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])

            w_next[k] = w_server[k] - torch.mul(att_weight, epsilon)

        return w_next

    def _reconstruct_full_model(self, lora_result, timellm_result, reference_model):
        """重构完整模型：聚合后的可训练参数 + 冻结的LLM参数"""
        full_model = copy.deepcopy(reference_model)

        # 更新LoRA参数
        for key, value in lora_result.items():
            full_model[key] = value

        # 更新TimeLLM参数
        for key, value in timellm_result.items():
            full_model[key] = value

        return full_model

    def _classify_parameters(self, model_params):
        """分类模型参数"""
        lora_params = {}
        timellm_params = {}
        frozen_params = {}

        for key, value in model_params.items():
            if self._is_lora_param(key):
                lora_params[key] = value
            elif self._is_timellm_param(key):
                timellm_params[key] = value
            else:
                frozen_params[key] = value  # LLM基础参数

        return lora_params, timellm_params, frozen_params

    def _is_timellm_param(self, param_name: str) -> bool:
        """判断是否为TimeLLM层参数"""
        timellm_keywords = [
            'ts2language',  # 时序映射层
            'output_projection',  # 输出投影层
            'normalize_layers',  # 归一化层
            'patch_embedding'  # 补丁嵌入层
        ]
        return any(keyword in param_name for keyword in timellm_keywords)

    def _extract_lora_params(self, client_models):
        """提取LoRA参数 - 复用现有逻辑"""
        lora_models = []
        for model in client_models:
            lora_params = {}
            for key, value in model.items():
                if self._is_lora_param(key):  # 复用现有函数
                    lora_params[key] = value
            lora_models.append(lora_params)
        return lora_models

    def _is_lora_param(self, param_name: str) -> bool:
        """判断是否为LoRA参数 - 复用现有逻辑"""
        lora_keywords = ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']
        return any(keyword in param_name for keyword in lora_keywords)


class FedDAAggregator:
    """FedDA双注意力聚合算法 - 完整版"""

    def __init__(self, num_clusters=3, rho=0.1, gamma=0.01,
                 enable_augmentation=True, augment_ratio=0.01):
        self.num_clusters = num_clusters
        self.rho = rho
        self.gamma = gamma
        self.enable_augmentation = enable_augmentation
        self.augment_ratio = augment_ratio

        # 初始化组件
        if self.enable_augmentation:
            from .data_augmentation import FedDADataAugmentation
            self.data_augmenter = FedDADataAugmentation(augment_ratio)

        from .geographic_clustering import FedDAGeographicClustering
        self.geo_clusterer = FedDAGeographicClustering(num_clusters)

        self.quasi_global_model = None
        self.clusters = None
        self.is_initialized = False

    def aggregate(self, client_models: List[Dict], client_info: List[Dict] = None,
                  federated_data: Dict = None, round_idx: int = 0):
        """
        完整的FedDA聚合过程

        Args:
            client_models: 客户端模型参数
            client_info: 客户端信息
            federated_data: 联邦数据（包含坐标等）
            round_idx: 当前轮次
        """
        # 第一次运行时进行初始化
        if not self.is_initialized and federated_data is not None:
            self._initialize_fedda(federated_data, client_info)
            self.is_initialized = True

        # 根据聚类结果组织客户端模型
        if self.clusters:
            clustered_models = self._organize_models_by_cluster(client_models, client_info)
        else:
            # 回退到简单聚类
            clustered_models = self._simple_clustering(client_models)

        # 步骤1：簇内聚合
        cluster_models = []
        for cluster_models_list in clustered_models.values():
            if cluster_models_list:
                cluster_model = self._dual_attention_aggregation(cluster_models_list)
                cluster_models.append(cluster_model)

        # 步骤2：簇间聚合
        if cluster_models:
            global_model = self._dual_attention_aggregation(cluster_models)
        else:
            global_model = copy.deepcopy(client_models[0]) if client_models else {}

        return global_model

    def _initialize_fedda(self, federated_data: Dict, client_info: List[Dict]):
        """初始化FedDA组件"""
        print("🔧 初始化FedDA组件...")

        augmented_data = {}

        # 数据增强
        if self.enable_augmentation:
            print("  📊 执行数据增强...")
            for client_id, client_data in federated_data['clients'].items():
                augmented_sample = self.data_augmenter.augment_client_data(
                    client_data, client_id
                )
                if augmented_sample is not None:
                    augmented_data[client_id] = augmented_sample

            print(f"  ✅ 完成 {len(augmented_data)} 个客户端的数据增强")

        # 地理聚类
        if augmented_data:
            print("  🗺️  执行地理位置聚类...")
            self.clusters = self.geo_clusterer.iterative_clustering(
                federated_data, augmented_data
            )
            print(f"  ✅ 聚类完成: {len(self.clusters)} 个簇")
            for cluster_id, client_list in self.clusters.items():
                print(f"    簇 {cluster_id}: {len(client_list)} 个客户端")

        print("🎯 FedDA初始化完成!")

    def _organize_models_by_cluster(self, client_models, client_info):
        """根据聚类结果组织模型"""
        clustered_models = {i: [] for i in range(self.num_clusters)}

        for i, (model, info) in enumerate(zip(client_models, client_info)):
            client_id = info['client_id']
            cluster_id = self.geo_clusterer.get_client_cluster(client_id)
            clustered_models[cluster_id].append(model)

        return clustered_models

    def _dual_attention_aggregation(self, models):
        """核心：双注意力聚合算法"""
        if not models:
            return {}

        if len(models) == 1:
            return copy.deepcopy(models[0])

        # 初始化输出模型
        output_model = copy.deepcopy(models[0])

        # 对每个参数层应用注意力机制
        for param_name in output_model.keys():
            # 计算注意力权重
            attention_weights = self._compute_attention_weights(
                models, param_name, output_model[param_name]
            )

            # 应用注意力加权更新
            output_model[param_name] = self._apply_attention_update(
                models, param_name, output_model[param_name], attention_weights
            )

        return output_model

    def _compute_attention_weights(self, models, param_name, output_param):
        """计算层级注意力权重"""
        distances = []

        for model in models:
            if param_name in model:
                # 计算Frobenius范数距离
                dist = torch.norm(model[param_name] - output_param, p='fro').item()
                distances.append(dist)
            else:
                distances.append(float('inf'))

        # 应用softmax获得注意力权重
        distances = torch.tensor(distances)
        attention_weights = torch.softmax(-distances, dim=0)  # 距离越小权重越大

        return attention_weights

    def _apply_attention_update(self, models, param_name, output_param, attention_weights):
        """应用注意力权重更新参数"""
        # 梯度计算
        gradient = torch.zeros_like(output_param)

        for i, model in enumerate(models):
            if param_name in model:
                gradient += attention_weights[i] * (output_param - model[param_name])

        # 添加准全局模型的正则化项（如果存在）
        if self.quasi_global_model and param_name in self.quasi_global_model:
            beta = 1.0  # 准全局模型权重
            gradient += self.rho * beta * (output_param - self.quasi_global_model[param_name])

        # 梯度下降更新
        updated_param = output_param - self.gamma * gradient

        return updated_param

    # 需要添加这个方法
    def _simple_clustering(self, client_models):
        """简单聚类回退方案"""
        clusters = {i: [] for i in range(self.num_clusters)}
        for i, model in enumerate(client_models):
            cluster_id = i % self.num_clusters
            clusters[cluster_id].append(model)
        return clusters

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
    elif aggregation_method.lower() == 'fedatt':
        epsilon = kwargs.get('fedatt_epsilon', 1.0)
        is_lora_mode = kwargs.get('is_lora_mode', False)  # 传入LoRA模式标志
        return FedAttAggregator(epsilon, is_lora_mode)
    elif aggregation_method.lower() == 'fedda':
        num_clusters = kwargs.get('fedda_clusters', 2)
        rho = kwargs.get('fedda_rho', 0.1)
        gamma = kwargs.get('fedda_gamma', 0.01)
        return FedDAAggregator(num_clusters, rho, gamma)
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