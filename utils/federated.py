# -*- coding: utf-8 -*-
"""
联邦学习核心组件
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import logging
from typing import Dict, List, Tuple
from collections import OrderedDict
import math
from dataset.data_loader import FederatedDataLoader

class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, data_loaders, args):
        self.client_id = client_id
        self.model = model
        self.data_loaders = data_loaders
        self.args = args
        self.logger = logging.getLogger(f"Client-{client_id}")

        # 优化器和损失函数
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

        # 训练统计
        self.train_losses = []
        self.val_losses = []

    def local_train(self, global_model_state: Dict) -> Tuple[Dict, Dict]:
        """
        本地训练

        Args:
            global_model_state: 全局模型状态

        Returns:
            local_state: 本地模型状态
            train_stats: 训练统计信息
        """
        # 加载全局模型参数
        self.model.load_state_dict(global_model_state)
        self.model.train()

        epoch_losses = []
        total_samples = 0

        # 本地训练多个epoch
        for epoch in range(self.args.local_epochs):
            batch_losses = []

            for batch_idx, (data, target) in enumerate(self.data_loaders['train']):
                data, target = data.to(self.args.device), target.to(self.args.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                batch_losses.append(loss.item())
                total_samples += data.size(0)

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)

            if epoch % max(1, self.args.local_epochs // 3) == 0:
                self.logger.debug(f"本地Epoch {epoch + 1}/{self.args.local_epochs}, Loss: {epoch_loss:.6f}")

        # 验证（如果有验证集）
        val_loss = self._validate() if 'val' in self.data_loaders else None

        # 更新学习率
        if val_loss is not None:
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step(epoch_losses[-1])

        # 训练统计
        train_stats = {
            'client_id': self.client_id,
            'train_loss': np.mean(epoch_losses),
            'val_loss': val_loss,
            'total_samples': total_samples,
            'lr': self.optimizer.param_groups[0]['lr']
        }

        self.train_losses.extend(epoch_losses)
        if val_loss is not None:
            self.val_losses.append(val_loss)

        return copy.deepcopy(self.model.state_dict()), train_stats

    def _validate(self) -> float:
        """验证模型性能"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for data, target in self.data_loaders['val']:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def evaluate(self, model_state: Dict = None) -> Dict:
        """
        评估模型性能

        Args:
            model_state: 模型状态（如果为None则使用当前模型）

        Returns:
            eval_metrics: 评估指标
        """
        if model_state is not None:
            self.model.load_state_dict(model_state)

        self.model.eval()

        test_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.data_loaders['test']:
                data, target = data.to(self.args.device), target.to(self.args.device)
                output = self.model(data)

                test_loss = self.criterion(output, target)
                test_losses.append(test_loss.item())

                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # 计算评估指标
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        # 计算MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

        eval_metrics = {
            'client_id': self.client_id,
            'test_loss': np.mean(test_losses),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'num_test_samples': len(targets)
        }

        return eval_metrics


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, global_model, federated_data, args):
        self.global_model = global_model
        self.federated_data = federated_data
        self.args = args
        self.logger = logging.getLogger("FedServer")

        # 客户端管理
        self.clients = {}
        self.client_coordinates = {}
        self._initialize_clients()

        # 训练历史
        self.global_losses = []
        self.client_stats_history = []
        self.eval_history = []

    def _initialize_clients(self):
        """初始化所有客户端"""
        self.logger.info("初始化客户端...")

        data_loader = FederatedDataLoader(self.args)

        for client_id in self.federated_data['metadata']['client_ids']:
            # 创建客户端模型（深拷贝全局模型）
            client_model = copy.deepcopy(self.global_model)

            # 获取客户端数据
            client_sequences = self.federated_data['clients'][client_id]['sequences']
            client_data_loaders = data_loader.create_data_loaders(client_sequences)

            # 创建客户端
            client = FederatedClient(
                client_id=client_id,
                model=client_model,
                data_loaders=client_data_loaders,
                args=self.args
            )

            self.clients[client_id] = client
            self.client_coordinates[client_id] = self.federated_data['clients'][client_id]['coordinates']

        self.logger.info(f"成功初始化 {len(self.clients)} 个客户端")

    def select_clients(self, round_num: int) -> List:
        """
        选择参与本轮训练的客户端

        Args:
            round_num: 轮次编号

        Returns:
            selected_clients: 选中的客户端ID列表
        """
        num_selected = max(1, int(self.args.frac * len(self.clients)))

        # 随机选择客户端
        np.random.seed(round_num + self.args.seed)  # 确保可重现性
        all_client_ids = list(self.clients.keys())
        selected_client_ids = np.random.choice(
            all_client_ids,
            size=num_selected,
            replace=False
        )

        return selected_client_ids.tolist()

    def aggregate_models(self, client_states: List[Dict], client_stats: List[Dict]) -> Dict:
        """
        聚合客户端模型

        Args:
            client_states: 客户端模型状态列表
            client_stats: 客户端训练统计列表

        Returns:
            global_state: 聚合后的全局模型状态
        """
        if self.args.aggregation == 'fedavg':
            return self._fedavg_aggregate(client_states, client_stats)
        elif self.args.aggregation == 'weighted':
            return self._weighted_aggregate(client_states, client_stats)
        else:
            raise ValueError(f"不支持的聚合算法: {self.args.aggregation}")

    def _fedavg_aggregate(self, client_states: List[Dict], client_stats: List[Dict]) -> Dict:
        """FedAvg聚合算法"""
        # 获取样本数量作为权重
        total_samples = sum(stat['total_samples'] for stat in client_stats)
        weights = [stat['total_samples'] / total_samples for stat in client_stats]

        # 聚合参数
        global_state = OrderedDict()

        for key in client_states[0].keys():
            global_state[key] = torch.zeros_like(client_states[0][key])

            for i, client_state in enumerate(client_states):
                global_state[key] += weights[i] * client_state[key]

        return global_state

    def _weighted_aggregate(self, client_states: List[Dict], client_stats: List[Dict]) -> Dict:
        """基于性能和坐标的加权聚合"""
        weights = []

        for i, stat in enumerate(client_stats):
            # 基础权重：样本数量
            sample_weight = stat['total_samples']

            # 性能权重：训练损失越小权重越大
            performance_weight = 1.0 / (1.0 + stat['train_loss'])

            # 坐标权重（如果启用）
            coord_weight = 1.0
            if self.args.use_coordinates:
                coord_weight = self._calculate_coordinate_weight(stat['client_id'])

            # 综合权重
            total_weight = sample_weight * performance_weight * coord_weight
            weights.append(total_weight)

        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # 聚合参数
        global_state = OrderedDict()

        for key in client_states[0].keys():
            global_state[key] = torch.zeros_like(client_states[0][key])

            for i, client_state in enumerate(client_states):
                global_state[key] += weights[i] * client_state[key]

        return global_state

    def _calculate_coordinate_weight(self, client_id) -> float:
        """
        基于坐标计算权重（简单示例）
        实际使用时可以结合Gemini API进行智能计算
        """
        # 这里是一个简单的示例，可以替换为更复杂的逻辑
        coord = self.client_coordinates[client_id]

        # 基于经纬度计算权重（示例逻辑）
        # 可以考虑地理聚类、覆盖相似性等因素
        base_lng, base_lat = 9.1, 45.4  # 米兰市中心大概坐标

        distance = math.sqrt((coord['lng'] - base_lng) ** 2 + (coord['lat'] - base_lat) ** 2)

        # 距离市中心越近权重越高（示例逻辑）
        weight = 1.0 / (1.0 + distance * 10)

        return weight

    def train_round(self, round_num: int) -> Dict:
        """
        执行一轮联邦训练

        Args:
            round_num: 轮次编号

        Returns:
            round_stats: 本轮训练统计
        """
        self.logger.info(f"开始第 {round_num + 1} 轮训练...")

        # 选择客户端
        selected_client_ids = self.select_clients(round_num)
        self.logger.info(f"选中客户端: {selected_client_ids}")

        # 获取当前全局模型状态
        global_state = self.global_model.state_dict()

        # 客户端并行训练
        client_states = []
        client_stats = []

        for client_id in selected_client_ids:
            client = self.clients[client_id]

            self.logger.debug(f"客户端 {client_id} 开始本地训练...")

            # 本地训练
            local_state, train_stat = client.local_train(global_state)

            client_states.append(local_state)
            client_stats.append(train_stat)

            self.logger.debug(f"客户端 {client_id} 训练完成，损失: {train_stat['train_loss']:.6f}")

        # 聚合模型
        self.logger.debug("开始模型聚合...")
        aggregated_state = self.aggregate_models(client_states, client_stats)

        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state)

        # 计算全局统计
        avg_train_loss = np.mean([stat['train_loss'] for stat in client_stats])
        avg_val_loss = np.mean([stat['val_loss'] for stat in client_stats if stat['val_loss'] is not None])
        total_samples = sum([stat['total_samples'] for stat in client_stats])

        round_stats = {
            'round': round_num + 1,
            'selected_clients': len(selected_client_ids),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss if not math.isnan(avg_val_loss) else None,
            'total_samples': total_samples,
            'client_stats': client_stats
        }

        # 保存历史
        self.global_losses.append(avg_train_loss)
        self.client_stats_history.append(client_stats)

        self.logger.info(f"第 {round_num + 1} 轮完成，平均训练损失: {avg_train_loss:.6f}")

        return round_stats

    def evaluate_global_model(self) -> Dict:
        """评估全局模型性能"""
        self.logger.info("评估全局模型...")

        global_state = self.global_model.state_dict()
        client_eval_results = []

        # 在所有客户端上评估
        for client_id, client in self.clients.items():
            eval_result = client.evaluate(global_state)
            client_eval_results.append(eval_result)

        # 计算全局指标
        global_metrics = {
            'avg_test_loss': np.mean([r['test_loss'] for r in client_eval_results]),
            'avg_mse': np.mean([r['mse'] for r in client_eval_results]),
            'avg_mae': np.mean([r['mae'] for r in client_eval_results]),
            'avg_rmse': np.mean([r['rmse'] for r in client_eval_results]),
            'avg_mape': np.mean([r['mape'] for r in client_eval_results]),
            'total_test_samples': sum([r['num_test_samples'] for r in client_eval_results]),
            'client_results': client_eval_results
        }

        self.eval_history.append(global_metrics)

        self.logger.info(f"全局评估完成 - MSE: {global_metrics['avg_mse']:.6f}, "
                         f"MAE: {global_metrics['avg_mae']:.6f}, RMSE: {global_metrics['avg_rmse']:.6f}")

        return global_metrics