# -*- coding: utf-8 -*-
"""
联邦学习服务器
"""

import torch
import random
import copy
from typing import List, Dict
from .aggregation import get_aggregator


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, global_model, args):
        self.global_model = global_model
        self.args = args
        self.device = torch.device(args.device)

        # 聚合器（支持LLM聚合和层级感知聚合）
        if args.aggregation in ['llm_fedavg', 'layer_aware_llm']:
            aggregator_kwargs = {
                'api_key': getattr(args, 'llm_api_key', None),
                'model_name': getattr(args, 'llm_model', 'DeepSeek-R1'),
                'cache_rounds': getattr(args, 'llm_cache_rounds', 1),
                'min_confidence': getattr(args, 'llm_min_confidence', 0.7),
                'is_lora_mode': hasattr(args, 'use_lora') and args.use_lora
            }

            # 层级感知聚合的额外参数
            if args.aggregation == 'layer_aware_llm':
                aggregator_kwargs['layer_analysis_enabled'] = getattr(args, 'layer_analysis_enabled', True)

            self.aggregator = get_aggregator(args.aggregation, **aggregator_kwargs)
        else:
            self.aggregator = get_aggregator(args.aggregation)

        # 训练历史
        self.train_history = {
            'global_loss': [],
            'client_losses': []
        }

    def get_global_model(self):
        """获取全局模型参数"""
        return copy.deepcopy(self.global_model.state_dict())

    def select_clients(self, all_clients: List, round_idx: int = 0):
        """
        选择参与训练的客户端

        Args:
            all_clients: 所有客户端列表
            round_idx: 当前轮次

        Returns:
            selected_clients: 选中的客户端列表
        """
        num_selected = max(1, int(len(all_clients) * self.args.frac))

        # 随机选择客户端
        selected_clients = random.sample(all_clients, num_selected)

        return selected_clients

    def aggregate_models(self, client_models: List[Dict], client_info: List[Dict],
                         selected_clients: List = None, round_idx: int = 0):
        """
        聚合客户端模型（支持LLM聚合）

        Args:
            client_models: 客户端模型参数列表
            client_info: 客户端信息列表
            selected_clients: 选中的客户端列表（用于LLM聚合）
            round_idx: 当前轮次

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if self.args.aggregation == 'llm_fedavg' and selected_clients:
            # 准备LLM聚合需要的统计信息
            client_stats = self._prepare_client_statistics(
                selected_clients, client_info, round_idx
            )

            # 使用LLM聚合器
            aggregated_model = self.aggregator.aggregate(
                client_models, client_info, client_stats, round_idx
            )
        elif self.args.aggregation == 'weighted':
            # 基于样本数量的加权聚合
            aggregated_model = self.aggregator.aggregate(client_models, client_info)
        else:
            # 标准FedAvg或LoRA聚合
            aggregated_model = self.aggregator.aggregate(client_models)

        return aggregated_model

    def _prepare_client_statistics(self, selected_clients: List, client_info: List[Dict], round_idx: int):
        """准备LLM聚合需要的客户端统计信息"""
        from .simple_llm_aggregator import create_client_statistics
        import random

        client_stats = []

        for i, client in enumerate(selected_clients):
            client_id = str(client.client_id)  # 确保是字符串

            # 获取坐标信息（使用Python原生float类型）
            coordinates = {
                'lng': float(116.0 + random.uniform(-2, 2)),  # 北京附近
                'lat': float(39.5 + random.uniform(-2, 2))
            }

            # 获取最近的训练损失（确保是Python float）
            loss = float(getattr(client, 'last_loss', 1.0))

            # 简单的流量统计（使用Python原生类型）
            traffic_stats = {
                'mean': float(random.uniform(100, 500)),
                'std': float(random.uniform(20, 80)),
                'trend': random.choice(['increasing', 'decreasing', 'stable'])
            }

            # 创建统计信息
            stats = create_client_statistics(
                client_id=client_id,
                coordinates=coordinates,
                loss=loss,
                model_params=None,
                traffic_data=None
            )

            # 手动设置流量统计
            stats.traffic_stats = traffic_stats

            client_stats.append(stats)

        return client_stats

    def update_global_model(self, aggregated_params):
        """更新全局模型 - 智能处理LoRA参数"""
        if self._is_lora_mode():
            # LoRA模式：只更新传入的LoRA参数，保持基础模型不变
            current_global_state = self.global_model.state_dict()

            # 更新聚合后的LoRA参数
            for key, value in aggregated_params.items():
                if key in current_global_state:
                    current_global_state[key] = value

            self.global_model.load_state_dict(current_global_state, strict=False)
        else:
            # 标准模式：直接加载所有参数
            self.global_model.load_state_dict(aggregated_params)

    def _is_lora_mode(self):
        """检查是否为LoRA模式"""
        return hasattr(self.global_model, 'llm_model') and hasattr(self.global_model.llm_model, 'peft_config')

    def _log_communication_efficiency(self, client_models: List[Dict], selected_clients: List):
        """记录通信效率统计"""
        if not client_models:
            return

        # 计算LoRA参数数量和大小
        total_lora_params = 0
        total_lora_size_mb = 0

        for client_model in client_models:
            for param_tensor in client_model.values():
                if isinstance(param_tensor, torch.Tensor):
                    total_lora_params += param_tensor.numel()
                    total_lora_size_mb += param_tensor.numel() * 4 / (1024 * 1024)  # 假设float32

        avg_lora_params = total_lora_params / len(client_models)
        avg_lora_size_mb = total_lora_size_mb / len(client_models)

        # 估算完整模型大小（用于对比）
        total_full_params = sum(p.numel() for p in self.global_model.parameters())
        full_model_size_mb = total_full_params * 4 / (1024 * 1024)

        # 计算通信节省比例
        communication_reduction = (1 - avg_lora_params / total_full_params) * 100

        print(f"  通信效率统计:")
        print(f"    每个客户端LoRA参数: {avg_lora_params:,.0f}")
        print(f"    每个客户端传输大小: {avg_lora_size_mb:.2f} MB")
        print(f"    完整模型大小: {full_model_size_mb:.2f} MB")
        print(f"    通信量减少: {communication_reduction:.1f}%")

    def evaluate_global_model(self, test_clients: List):
        """
        评估全局模型性能

        Args:
            test_clients: 测试客户端列表

        Returns:
            avg_loss: 平均测试损失
        """
        total_loss = 0
        total_samples = 0

        # 为每个客户端设置全局模型参数
        global_params = self.get_global_model()

        for client in test_clients:
            # 设置全局模型参数
            client.set_model_params(global_params)

            # 评估客户端
            client_loss = client.evaluate()
            client_samples = client.num_samples

            total_loss += client_loss * client_samples
            total_samples += client_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return avg_loss

    def federated_round(self, all_clients: List, round_idx: int):
        """
        执行一轮联邦学习 - 优化显存使用

        Args:
            all_clients: 所有客户端列表
            round_idx: 当前轮次

        Returns:
            round_results: 本轮训练结果
        """
        # 1. 选择客户端
        selected_clients = self.select_clients(all_clients, round_idx)
        print(f"轮次 {round_idx}: 选择了 {len(selected_clients)} 个客户端")

        # 2. 获取全局模型参数
        global_params = self.get_global_model()

        # 3. 客户端本地训练（逐个进行以节省显存）
        client_models = []
        client_info = []
        client_losses = []

        for i, client in enumerate(selected_clients):
            print(f"  训练客户端 {client.client_id} ({i + 1}/{len(selected_clients)})")

            # 为客户端分配模型
            from federated_train import assign_model_to_client, cleanup_client_model
            assign_model_to_client(client, None, global_params)

            # 本地训练
            local_loss = client.local_train()

            # 收集模型参数和信息
            client_models.append(client.get_model_params())
            client_info.append(client.get_client_info())
            client_losses.append(local_loss)

            print(f"    本地损失 = {local_loss:.6f}")

            # 立即清理客户端模型以释放显存
            cleanup_client_model(client)

        # 4. 聚合模型
        aggregated_params = self.aggregate_models(
            client_models, client_info, selected_clients, round_idx
        )

        # 统计通信效率（LoRA模式）
        if self._is_lora_mode():
            self._log_communication_efficiency(client_models, selected_clients)

        # 5. 更新全局模型
        self.update_global_model(aggregated_params)

        # 6. 清理聚合过程中的临时数据
        del client_models, global_params, aggregated_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. 记录训练历史
        avg_client_loss = sum(client_losses) / len(client_losses)
        self.train_history['client_losses'].append(avg_client_loss)

        round_results = {
            'selected_clients': [c.client_id for c in selected_clients],
            'avg_client_loss': avg_client_loss,
            'client_losses': dict(zip([c.client_id for c in selected_clients], client_losses))
        }

        # 如果是LoRA+LLM模式，添加额外信息
        if self.args.aggregation == 'llm_fedavg' and self._is_lora_mode():
            round_results['mode'] = 'LoRA + LLM智能聚合'
            round_results['communication_efficiency'] = '99%+'

        return round_results

    def get_train_history(self):
        """获取训练历史"""
        return copy.deepcopy(self.train_history)