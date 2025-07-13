# -*- coding: utf-8 -*-
"""
联邦学习服务器
"""
import numpy as np
import torch
import random
import copy
from typing import List, Dict
from .aggregation import get_aggregator
from utils.utils import assign_model_to_client, cleanup_client_model


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, global_model, args):
        self.global_model = global_model
        self.args = args
        self.device = torch.device(args.device)

        # 聚合器（支持多种LLM聚合方式）
        if args.aggregation in ['llm_fedavg', 'layer_aware_llm', 'multi_dim_llm', 'enhanced_multi_dim_llm']:
            aggregator_kwargs = {
                'api_key': getattr(args, 'llm_api_key', None),
                'model_name': getattr(args, 'llm_model', 'DeepSeek-R1'),
                'cache_rounds': getattr(args, 'llm_cache_rounds', 1),
                'min_confidence': getattr(args, 'llm_min_confidence', 0.7),
                'is_lora_mode': hasattr(args, 'use_lora') and args.use_lora
            }

            # 增强版多维度聚合的额外参数
            if args.aggregation == 'enhanced_multi_dim_llm':
                aggregator_kwargs['dimensions'] = getattr(args, 'enhanced_multi_dim_dimensions',
                                                          ['model_performance', 'data_quality', 'spatial_distribution',
                                                           'temporal_stability', 'traffic_pattern'])
                aggregator_kwargs['server_instance'] = self
                aggregator_kwargs['verbose'] = getattr(args, 'expert_verbose', True)

                # 新增动态融合参数
                aggregator_kwargs['alpha_max'] = getattr(args, 'alpha_max', 0.9)
                aggregator_kwargs['alpha_min'] = getattr(args, 'alpha_min', 0.2)
                aggregator_kwargs['decay_type'] = getattr(args, 'decay_type', 'sigmoid')
                aggregator_kwargs['base_constraint'] = getattr(args, 'base_constraint', 0.25)

            self.aggregator = get_aggregator(args.aggregation, **aggregator_kwargs)
        else:
            self.aggregator = get_aggregator(args.aggregation)

        # 训练历史
        self.train_history = {
            'global_loss': [],
            'client_losses': []
        }

        # 新增：客户端历史数据跟踪
        self.client_history = {
            'losses': {},  # {client_id: [loss1, loss2, ...]}
            'performance_trends': {},  # {client_id: [trend1, trend2, ...]}
            'participation_count': {},  # {client_id: count}
            'last_seen_round': {}  # {client_id: round_idx}
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
        聚合客户端模型（支持多种LLM聚合）

        Args:
            client_models: 客户端模型参数列表
            client_info: 客户端信息列表
            selected_clients: 选中的客户端列表（用于LLM聚合）
            round_idx: 当前轮次

        Returns:
            aggregated_model: 聚合后的模型参数
        """
        if self.args.aggregation in ['llm_fedavg', 'layer_aware_llm', 'multi_dim_llm', 'enhanced_multi_dim_llm'] and selected_clients:
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
        """准备LLM聚合需要的客户端统计信息 - 使用真实数据"""
        client_stats = []

        for i, client in enumerate(selected_clients):
            client_id = str(client.client_id)  # 确保是字符串

            # 获取真实坐标信息
            coordinates = client.get_coordinates()
            # 确保坐标是Python float类型
            coordinates = {
                'lng': float(coordinates.get('lng', 0.0)),
                'lat': float(coordinates.get('lat', 0.0))
            }

            # 获取最近的训练损失（确保是Python float）
            loss = float(getattr(client, 'last_loss', 1.0))

            # 获取真实的流量统计信息
            real_traffic_stats = client.get_real_traffic_stats()

            # 转换为LLM聚合需要的格式（确保都是Python原生类型）
            traffic_stats = {
                'mean': float(real_traffic_stats.get('mean', 0.0)),
                'std': float(real_traffic_stats.get('std', 0.0)),
                'min': float(real_traffic_stats.get('min', 0.0)),
                'max': float(real_traffic_stats.get('max', 0.0)),
                'median': float(real_traffic_stats.get('median', 0.0)),
                'trend': str(real_traffic_stats.get('trend', 'stable')),
                'trend_slope': float(real_traffic_stats.get('trend_slope', 0.0)),
                'recent_mean': float(real_traffic_stats.get('recent_mean', 0.0)),
                'coefficient_of_variation': float(real_traffic_stats.get('coefficient_of_variation', 0.0)),
                'q25': float(real_traffic_stats.get('q25', 0.0)),
                'q75': float(real_traffic_stats.get('q75', 0.0)),
                'iqr': float(real_traffic_stats.get('iqr', 0.0)),
                'data_points': int(real_traffic_stats.get('data_points', 0))
            }

            # 创建统计信息
            stats = create_client_statistics(
                client_id=client_id,
                coordinates=coordinates,
                loss=loss,
                model_params=None,
                traffic_data=None  # 不需要传递原始数据，直接使用计算好的统计信息
            )

            # 设置真实的流量统计
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

    def evaluate_global_model_detailed(self, test_clients: List, eval_type='val'):
        """
        详细评估全局模型性能

        Args:
            test_clients: 测试客户端列表
            eval_type: 评估类型 ('val' 或 'test')

        Returns:
            avg_loss: 平均损失
            client_losses: 各客户端损失详情
        """
        total_loss = 0
        total_samples = 0
        client_losses = {}

        # 获取全局模型参数
        global_params = self.get_global_model()

        for client in test_clients:
            # 设置全局模型参数
            assign_model_to_client(client, None, global_params)  # 直接使用导入的函数

            # 根据评估类型选择数据加载器
            if eval_type == 'val' and hasattr(client, 'val_loader'):
                eval_loader = client.val_loader
            elif eval_type == 'test' and hasattr(client, 'test_loader'):
                eval_loader = client.test_loader
            else:
                eval_loader = client.data_loader  # 默认使用训练数据

            # 评估客户端
            client_loss = client.evaluate(eval_loader)
            client_samples = len(eval_loader.dataset)

            total_loss += client_loss * client_samples
            total_samples += client_samples
            client_losses[client.client_id] = client_loss

            # 清理客户端模型
            cleanup_client_model(client)  # 直接使用导入的函数

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')

        # 清理全局参数
        del global_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return avg_loss, client_losses

    def final_test_evaluation(self, test_clients: List):
        """
        最终测试集评估

        Args:
            test_clients: 测试客户端列表

        Returns:
            comprehensive_results: 综合评估结果
        """
        print("\n" + "=" * 60)
        print("最终测试集评估")
        print("=" * 60)

        # 测试集评估
        test_loss, test_client_losses = self.evaluate_global_model_detailed(test_clients, 'test')

        # 为了对比，也计算验证集损失
        val_loss, val_client_losses = self.evaluate_global_model_detailed(test_clients, 'val')

        # 计算统计信息
        test_losses = list(test_client_losses.values())
        val_losses = list(val_client_losses.values())

        results = {
            'test_loss': {
                'avg': test_loss,
                'std': np.std(test_losses),
                'min': min(test_losses),
                'max': max(test_losses),
                'client_losses': test_client_losses
            },
            'val_loss': {
                'avg': val_loss,
                'std': np.std(val_losses),
                'min': min(val_losses),
                'max': max(val_losses),
                'client_losses': val_client_losses
            }
        }

        # 打印结果
        print(f"测试集平均损失: {test_loss:.6f} (±{np.std(test_losses):.6f})")
        print(f"验证集平均损失: {val_loss:.6f} (±{np.std(val_losses):.6f})")
        print(f"测试集损失范围: [{min(test_losses):.6f}, {max(test_losses):.6f}]")

        # 找出表现最好和最差的客户端
        best_test_client = min(test_client_losses.items(), key=lambda x: x[1])
        worst_test_client = max(test_client_losses.items(), key=lambda x: x[1])

        print(f"测试集最佳客户端: {best_test_client[0]} (损失: {best_test_client[1]:.6f})")
        print(f"测试集最差客户端: {worst_test_client[0]} (损失: {worst_test_client[1]:.6f})")

        return results

    def update_client_history(self, selected_clients: List, client_losses: List[float], round_idx: int):
        """更新客户端历史数据"""
        for client, loss in zip(selected_clients, client_losses):
            client_id = str(client.client_id)

            # 更新损失历史
            if client_id not in self.client_history['losses']:
                self.client_history['losses'][client_id] = []
            self.client_history['losses'][client_id].append(float(loss))

            # 保持最近N轮的历史（避免内存过度增长）
            max_history = 20
            if len(self.client_history['losses'][client_id]) > max_history:
                self.client_history['losses'][client_id] = self.client_history['losses'][client_id][-max_history:]

            # 更新参与次数
            if client_id not in self.client_history['participation_count']:
                self.client_history['participation_count'][client_id] = 0
            self.client_history['participation_count'][client_id] += 1

            # 更新最后参与轮次
            self.client_history['last_seen_round'][client_id] = round_idx

            # 计算并存储性能趋势
            self._calculate_and_store_trend(client_id, round_idx)

    def _calculate_and_store_trend(self, client_id: str, round_idx: int):
        """计算并存储客户端性能趋势"""
        losses = self.client_history['losses'][client_id]

        if len(losses) < 2:
            trend_info = {
                'short_term': 'stable',
                'long_term': 'stable',
                'slope': 0.0,
                'volatility': 0.0,
                'improvement_rate': 0.0
            }
        else:
            trend_info = self._analyze_loss_trend(losses)

        # 存储趋势信息
        if client_id not in self.client_history['performance_trends']:
            self.client_history['performance_trends'][client_id] = []

        self.client_history['performance_trends'][client_id].append({
            'round': round_idx,
            'trend_info': trend_info
        })

    def _analyze_loss_trend(self, losses: List[float]) -> Dict:
        """分析损失趋势的详细信息"""
        import numpy as np

        losses_array = np.array(losses)
        n = len(losses_array)

        # 1. 短期趋势（最近3-5轮）
        short_term_window = min(5, n)
        recent_losses = losses_array[-short_term_window:]
        short_term_trend = self._get_trend_direction(recent_losses)

        # 2. 长期趋势（所有历史）
        long_term_trend = self._get_trend_direction(losses_array)

        # 3. 计算线性回归斜率
        if n >= 3:
            x = np.arange(n)
            slope, _ = np.polyfit(x, losses_array, 1)
        else:
            slope = 0.0

        # 4. 计算波动性（标准差）
        volatility = float(np.std(losses_array)) if n > 1 else 0.0

        # 5. 计算改进率（相对于初始损失的改进百分比）
        if n >= 2 and losses_array[0] != 0:
            improvement_rate = (losses_array[0] - losses_array[-1]) / losses_array[0] * 100
        else:
            improvement_rate = 0.0

        # 6. 计算趋势强度
        trend_strength = self._calculate_trend_strength(losses_array)

        return {
            'short_term': short_term_trend,
            'long_term': long_term_trend,
            'slope': float(slope),
            'volatility': volatility,
            'improvement_rate': improvement_rate,
            'trend_strength': trend_strength,
            'consistency': self._calculate_consistency(losses_array)
        }

    def _get_trend_direction(self, losses: np.ndarray) -> str:
        """获取趋势方向"""
        if len(losses) < 2:
            return 'stable'

        # 使用线性回归斜率判断趋势
        x = np.arange(len(losses))
        slope, _ = np.polyfit(x, losses, 1)

        # 设置阈值来判断趋势
        if slope < -0.005:  # 明显下降
            return 'improving'
        elif slope > 0.005:  # 明显上升
            return 'deteriorating'
        else:
            return 'stable'

    def _calculate_trend_strength(self, losses: np.ndarray) -> float:
        """计算趋势强度（0-1之间，1表示趋势非常明显）"""
        if len(losses) < 3:
            return 0.0

        # 使用相关系数的绝对值作为趋势强度
        x = np.arange(len(losses))
        correlation_matrix = np.corrcoef(x, losses)
        correlation = abs(correlation_matrix[0, 1])

        return float(correlation) if not np.isnan(correlation) else 0.0

    def _calculate_consistency(self, losses: np.ndarray) -> float:
        """计算学习一致性（波动小=一致性高）"""
        if len(losses) < 2:
            return 1.0

        # 计算相对标准差（变异系数）
        mean_loss = np.mean(losses)
        if mean_loss == 0:
            return 1.0

        cv = np.std(losses) / mean_loss
        # 将变异系数转换为一致性分数（0-1之间，1表示最一致）
        consistency = 1.0 / (1.0 + cv)

        return float(consistency)

    def get_client_trend_summary(self, client_id: str) -> Dict:
        """获取客户端趋势摘要"""
        client_id = str(client_id)

        if client_id not in self.client_history['losses']:
            return {
                'status': 'new_client',
                'description': 'new',
                'score': 1.0,
                'details': {}
            }

        losses = self.client_history['losses'][client_id]
        participation = self.client_history['participation_count'].get(client_id, 0)

        if len(losses) < 2:
            return {
                'status': 'insufficient_data',
                'description': 'stable',
                'score': 1.0,
                'details': {'participation_count': participation}
            }

        # 获取最新的趋势分析
        trend_info = self._analyze_loss_trend(losses)

        # 计算综合趋势评分
        trend_score = self._calculate_trend_score(trend_info, participation)

        # 生成趋势描述
        description = self._generate_trend_description(trend_info)

        return {
            'status': 'analyzed',
            'description': description,
            'score': trend_score,
            'details': {
                'participation_count': participation,
                'trend_info': trend_info,
                'recent_losses': losses[-5:],  # 最近5轮损失
            }
        }

    def _calculate_trend_score(self, trend_info: Dict, participation: int) -> float:
        """计算趋势评分（用于聚合权重计算）"""
        base_score = 1.0

        # 1. 趋势方向评分
        if trend_info['short_term'] == 'improving':
            trend_bonus = 0.3
        elif trend_info['short_term'] == 'deteriorating':
            trend_bonus = -0.2
        else:
            trend_bonus = 0.0

        # 2. 改进率评分
        improvement_bonus = min(0.2, trend_info['improvement_rate'] / 100)

        # 3. 一致性评分
        consistency_bonus = (trend_info['consistency'] - 0.5) * 0.2

        # 4. 参与度评分（参与次数越多，越可信）
        participation_bonus = min(0.1, participation / 20 * 0.1)

        # 5. 趋势强度评分
        strength_bonus = trend_info['trend_strength'] * 0.1

        final_score = base_score + trend_bonus + improvement_bonus + consistency_bonus + participation_bonus + strength_bonus

        # 确保评分在合理范围内
        return max(0.1, min(2.0, final_score))

    def _generate_trend_description(self, trend_info: Dict) -> str:
        """生成趋势描述"""
        short_term = trend_info['short_term']
        long_term = trend_info['long_term']
        improvement_rate = trend_info['improvement_rate']
        consistency = trend_info['consistency']

        if short_term == 'improving' and long_term == 'improving':
            if improvement_rate > 10:
                return 'strongly_improving'
            else:
                return 'improving'
        elif short_term == 'deteriorating' and long_term == 'deteriorating':
            return 'deteriorating'
        elif short_term != long_term:
            if consistency > 0.7:
                return f'recently_{short_term}'
            else:
                return 'unstable'
        else:
            if consistency > 0.8:
                return 'stable_good'
            else:
                return 'stable'

    def save_checkpoint(self, save_path, round_idx, best_val_loss=None, train_history=None):
        """保存检查点"""
        import os

        checkpoint = {
            'round': round_idx,
            'global_model_state_dict': self.global_model.state_dict(),
            'train_history': train_history or self.train_history,
            'best_val_loss': best_val_loss,
            'args': vars(self.args),
            'client_history': getattr(self, 'client_history', {}),
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"检查点已保存: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        import os

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 恢复全局模型
        self.global_model.load_state_dict(checkpoint['global_model_state_dict'])

        # 恢复训练历史
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.client_history = checkpoint.get('client_history', getattr(self, 'client_history', {}))

        start_round = checkpoint['round'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        print(f"已从轮次 {checkpoint['round']} 恢复检查点")
        return start_round, best_val_loss

    def save_best_model(self, save_path, val_loss, round_idx):
        """保存最优验证模型"""
        import os

        best_model = {
            'round': round_idx,
            'val_loss': val_loss,
            'global_model_state_dict': self.global_model.state_dict(),
            'args': vars(self.args)
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model, save_path)
        print(f"最优模型已保存: {save_path} (轮次 {round_idx}, 验证损失: {val_loss:.6f})")

    def federated_round(self, all_clients: List, round_idx: int):
        """
        执行一轮联邦学习 - 优化显存使用并记录历史数据
        """
        # 1. 选择客户端
        selected_clients = self.select_clients(all_clients, round_idx)
        print(f"轮次 {round_idx}: 选择了 {len(selected_clients)} 个客户端")

        # 2. 获取全局模型参数
        global_params = self.get_global_model()

        # 3. 客户端本地训练
        client_models = []
        client_info = []
        client_losses = []

        for i, client in enumerate(selected_clients):
            print(f"  训练客户端 {client.client_id} ({i + 1}/{len(selected_clients)})")

            # 为客户端分配模型
            from utils.utils import assign_model_to_client, cleanup_client_model
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

        # 4. 更新客户端历史数据（在聚合之前）
        self.update_client_history(selected_clients, client_losses, round_idx)

        # 5. 聚合模型
        aggregated_params = self.aggregate_models(
            client_models, client_info, selected_clients, round_idx
        )

        # 统计通信效率（LoRA模式）
        if self._is_lora_mode():
            self._log_communication_efficiency(client_models, selected_clients)

        # 6. 更新全局模型
        self.update_global_model(aggregated_params)

        # 7. 清理聚合过程中的临时数据
        del client_models, global_params, aggregated_params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 8. 记录训练历史（增强版）
        avg_client_loss = sum(client_losses) / len(client_losses)
        self.train_history['client_losses'].append(avg_client_loss)

        # 如果有验证集，计算验证损失
        if hasattr(selected_clients[0], 'val_loader'):
            print("计算验证集损失...")
            val_loss, _ = self.evaluate_global_model_detailed(selected_clients[:5], 'val')
            if 'val_losses' not in self.train_history:
                self.train_history['val_losses'] = []
            self.train_history['val_losses'].append(val_loss)
            print(f"验证集损失: {val_loss:.6f}")

        round_results = {
            'selected_clients': [c.client_id for c in selected_clients],
            'avg_client_loss': avg_client_loss,
            'client_losses': dict(zip([c.client_id for c in selected_clients], client_losses))
        }

        # 添加验证损失到结果中
        if 'val_losses' in self.train_history and self.train_history['val_losses']:
            round_results['val_loss'] = self.train_history['val_losses'][-1]

        return round_results

    def _get_round_trend_summary(self, selected_clients: List) -> Dict:
        """获取本轮的趋势分析摘要"""
        trend_summary = {}

        for client in selected_clients:
            client_id = str(client.client_id)
            trend_info = self.get_client_trend_summary(client_id)
            trend_summary[client_id] = {
                'description': trend_info['description'],
                'score': trend_info['score'],
                'participation': trend_info['details'].get('participation_count', 0)
            }

        return trend_summary

    def get_train_history(self):
        """获取训练历史"""
        return copy.deepcopy(self.train_history)

# 客户端统计信息类（简化版）
class ClientStatistics:
    """客户端统计信息"""

    def __init__(self, client_id: str, coordinates: Dict, loss: float, traffic_stats: Dict = None):
        self.client_id = client_id
        self.coordinates = coordinates
        self.loss = loss
        self.traffic_stats = traffic_stats or {}

def create_client_statistics(client_id: str, coordinates: Dict, loss: float,
                             model_params: Dict = None, traffic_data=None) -> ClientStatistics:
    """创建客户端统计信息 - 确保数据类型正确"""

    # 确保所有数值都是Python原生类型
    client_id = str(client_id)
    loss = float(loss)

    # 确保坐标是Python float
    coordinates = {
        'lng': float(coordinates.get('lng', 0)),
        'lat': float(coordinates.get('lat', 0))
    }

    traffic_stats = {}
    if traffic_data is not None:
        import numpy as np
        traffic_stats = {
            'mean': float(np.mean(traffic_data)),
            'std': float(np.std(traffic_data)),
            'trend': 'increasing' if len(traffic_data) > 1 and traffic_data[-1] > traffic_data[0] else 'stable'
        }

    return ClientStatistics(
        client_id=client_id,
        coordinates=coordinates,
        loss=loss,
        traffic_stats=traffic_stats
    )