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