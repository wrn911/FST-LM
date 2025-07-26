# -*- coding: utf-8 -*-
"""
联邦学习客户端 - 添加真实坐标和流量数据支持
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm


class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id, model, data_loader, args, coordinates=None, original_traffic_stats=None):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.args = args
        self.device = torch.device(args.device)

        # 存储真实的坐标和流量统计信息
        self.coordinates = coordinates or {'lng': 0.0, 'lat': 0.0}
        self.original_traffic_stats = original_traffic_stats or {}

        # 这些将在需要时动态创建
        self.optimizer = None
        self.criterion = None

        # 客户端统计信息
        self.num_samples = len(data_loader.dataset)

        # 新增: 存储全局模型参数用于FedProx (支持LoRA)
        self.global_params = None
        self.use_fedprox = getattr(args, 'use_fedprox', False)
        self.fedprox_mu = getattr(args, 'fedprox_mu', 0.1)
        self.is_lora_mode = hasattr(args, 'use_lora') and args.use_lora

    def get_model_params(self):
        """获取模型参数 - LoRA模式只返回可训练参数"""
        if hasattr(self.model, 'llm_model') and hasattr(self.model.llm_model, 'peft_config'):
            # LoRA模式：只返回可训练的LoRA参数，大幅减少通信量
            trainable_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:  # 只获取可训练参数
                    trainable_params[name] = param.data.clone()
            return trainable_params
        else:
            # 标准模式：返回所有参数
            return copy.deepcopy(self.model.state_dict())

    def set_model_params(self, model_params):
        """设置模型参数 - LoRA模式只更新可训练参数"""
        if hasattr(self.model, 'llm_model') and hasattr(self.model.llm_model, 'peft_config'):
            # LoRA模式：只更新传入的参数（应该都是LoRA参数）
            current_state = self.model.state_dict()

            # 只更新传入的参数
            for key, value in model_params.items():
                if key in current_state:
                    current_state[key] = value

            self.model.load_state_dict(current_state, strict=False)
        else:
            # 标准模式：加载所有参数
            self.model.load_state_dict(model_params)

    def local_train(self):
        """本地训练"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(self.args.local_epochs):
            for x, y in self.data_loader:
                x, y = x.to(self.device), y.to(self.device)

                # 准备TimeLLM输入格式
                x_enc = x.unsqueeze(-1)  # [B, seq_len, 1]
                y_true = y.unsqueeze(-1)  # [B, pred_len, 1]

                # 创建占位符输入
                batch_size = x_enc.shape[0]
                x_mark_enc = torch.zeros(batch_size, x_enc.shape[1], 4).to(self.device)
                x_dec = torch.zeros(batch_size, self.args.pred_len, 1).to(self.device)
                x_mark_dec = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = self.criterion(outputs, y_true)

                # 新增: FedProx正则化项 (LoRA优化版本)
                if self.use_fedprox and self.global_params is not None:
                    fedprox_reg = self._compute_lora_fedprox_regularization()
                    loss += fedprox_reg

                # 反向传播
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # 清理中间变量和梯度
                del x_enc, y_true, x_mark_enc, x_dec, x_mark_dec, outputs, loss

        # 强制清理梯度
        self.optimizer.zero_grad()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        # 记录最近的训练损失，用于LLM聚合决策
        self.last_loss = avg_loss

        return avg_loss

    def evaluate(self, test_loader=None):
        """评估模型"""
        if test_loader is None:
            test_loader = self.data_loader

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                x_enc = x.unsqueeze(-1)
                y_true = y.unsqueeze(-1)

                batch_size = x_enc.shape[0]
                x_mark_enc = torch.zeros(batch_size, x_enc.shape[1], 4).to(self.device)
                x_dec = torch.zeros(batch_size, self.args.pred_len, 1).to(self.device)
                x_mark_dec = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = self.criterion(outputs, y_true)

                total_loss += loss.item()
                num_batches += 1

                # 清理中间变量
                del x_enc, y_true, x_mark_enc, x_dec, x_mark_dec, outputs, loss

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss

    def get_client_info(self):
        """获取客户端信息"""
        return {
            'client_id': self.client_id,
            'num_samples': self.num_samples
        }

    def get_real_traffic_stats(self):
        """获取真实流量统计信息（供LLM聚合使用）"""
        return self.original_traffic_stats

    def get_coordinates(self):
        """获取真实坐标信息（供LLM聚合使用）"""
        return self.coordinates

    def set_global_params_for_fedprox(self, global_params):
        """设置全局模型参数用于FedProx正则化 - LoRA优化版本"""
        if self.use_fedprox:
            self.global_params = {}
            # 直接存储参数张量，按参数名索引
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # 从state_dict中找到对应参数
                    if name in global_params:
                        self.global_params[name] = global_params[name].clone().detach().to(param.device)
            print(f"  FedProx存储了 {len(self.global_params)} 个LoRA参数用于正则化")

    def _is_lora_param(self, param_name: str) -> bool:
        """判断参数是否为LoRA参数"""
        lora_keywords = ['lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B']
        return any(keyword in param_name for keyword in lora_keywords)

    def _compute_lora_fedprox_regularization(self):
        fedprox_term = 0.0
        # 直接使用模型参数，不通过get_model_params()
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.global_params:
                param_diff = param - self.global_params[name]
                fedprox_term += torch.sum(param_diff ** 2)

        # LoRA模式使用更大的μ值
        effective_mu = self.fedprox_mu * 2.0 if self.is_lora_mode else self.fedprox_mu
        return (effective_mu / 2.0) * fedprox_term

    def _compute_fedprox_regularization(self):
        """计算标准FedProx正则化项 (保留用于兼容)"""
        fedprox_term = 0.0

        current_params = self.get_model_params()

        for key in current_params:
            if key in self.global_params:
                # 计算 ||w - w_global||^2
                param_diff = current_params[key] - self.global_params[key]
                fedprox_term += torch.sum(param_diff ** 2)

        return (self.fedprox_mu / 2.0) * fedprox_term

    def evaluate_with_metrics(self, test_loader=None):
        """评估模型并返回多个指标"""
        if test_loader is None:
            test_loader = self.data_loader

        self.model.eval()
        total_mse = 0
        total_mae = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                x_enc = x.unsqueeze(-1)
                y_true = y.unsqueeze(-1)

                batch_size = x_enc.shape[0]
                x_mark_enc = torch.zeros(batch_size, x_enc.shape[1], 4).to(self.device)
                x_dec = torch.zeros(batch_size, self.args.pred_len, 1).to(self.device)
                x_mark_dec = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                # 计算MSE和MAE
                mse = torch.nn.functional.mse_loss(outputs, y_true, reduction='sum')
                mae = torch.nn.functional.l1_loss(outputs, y_true, reduction='sum')

                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += batch_size * self.args.pred_len

                # 清理中间变量
                del x_enc, y_true, x_mark_enc, x_dec, x_mark_dec, outputs, mse, mae

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_mse = total_mse / total_samples if total_samples > 0 else float('inf')
        avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')

        return {
            'mse': avg_mse,
            'mae': avg_mae,
            'loss': avg_mse  # 保持兼容性
        }