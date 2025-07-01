# -*- coding: utf-8 -*-
"""
本地训练TimeLLM模型 - 简化版本
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.TimeLLM import Model
from dataset.data_loader import get_federated_data
from utils.config import get_args


class ModelConfig:
    """TimeLLM模型配置类"""

    def __init__(self, args):
        self.task_name = 'long_term_forecast'
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = 1  # 单变量时序
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_ff = args.d_model * 4

        # LLM配置
        self.llm_model = 'GPT2'
        self.llm_dim = 768
        self.llm_layers = 6

        # 补丁配置
        self.patch_len = 16
        self.stride = 8

        self.dropout = args.dropout
        self.prompt_domain = True
        self.content = "The dataset records the wireless traffic of a certain base station"


class SimpleTrainer:
    """简化版训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 初始化模型
        self.model_config = ModelConfig(args)
        self.model = Model(self.model_config).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        self.best_loss = float('inf')

    def train_epoch(self, data_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for x, y in tqdm(data_loader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)

            # 准备TimeLLM输入格式
            x_enc = x.unsqueeze(-1)  # [B, seq_len, 1]
            y_true = y.unsqueeze(-1)  # [B, pred_len, 1]

            # 创建占位符输入
            x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(self.device)
            x_dec = torch.zeros(x_enc.shape[0], self.args.pred_len, 1).to(self.device)
            x_mark_dec = torch.zeros(x_enc.shape[0], self.args.pred_len, 4).to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = self.criterion(outputs, y_true)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def validate(self, data_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Validating"):
                x, y = x.to(self.device), y.to(self.device)

                x_enc = x.unsqueeze(-1)
                y_true = y.unsqueeze(-1)

                x_mark_enc = torch.zeros(x_enc.shape[0], x_enc.shape[1], 4).to(self.device)
                x_dec = torch.zeros(x_enc.shape[0], self.args.pred_len, 1).to(self.device)
                x_mark_dec = torch.zeros(x_enc.shape[0], self.args.pred_len, 4).to(self.device)

                outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = self.criterion(outputs, y_true)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(self, train_loader, val_loader=None, epochs=50):
        """完整训练流程"""
        print("开始训练...")

        for epoch in range(1, epochs + 1):
            # 训练
            train_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch}/{epochs} - 训练损失: {train_loss:.6f}")

            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"验证损失: {val_loss:.6f}")

                # 记录最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    print(f"发现更好的模型! 损失: {val_loss:.6f}")

        print(f"训练完成! 最佳损失: {self.best_loss:.6f}")


def prepare_data(federated_data, client_id, batch_size=32):
    """准备单个客户端的数据加载器"""
    client_data = federated_data['clients'][client_id]
    sequences = client_data['sequences']

    loaders = {}

    # 训练集
    if 'train' in sequences:
        X_train = torch.FloatTensor(sequences['train']['history'])
        y_train = torch.FloatTensor(sequences['train']['target'])
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证集
    if 'val' in sequences:
        X_val = torch.FloatTensor(sequences['val']['history'])
        y_val = torch.FloatTensor(sequences['val']['target'])
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        loaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 测试集
    if 'test' in sequences:
        X_test = torch.FloatTensor(sequences['test']['history'])
        y_test = torch.FloatTensor(sequences['test']['target'])
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        loaders['test'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return loaders


def main():
    """主函数"""
    # 获取参数
    args = get_args()

    print("=== 本地训练TimeLLM ===")
    print(f"设备: {args.device}")
    print(f"序列长度: {args.seq_len}")
    print(f"预测长度: {args.pred_len}")

    # 加载数据
    print("加载数据...")
    federated_data, _ = get_federated_data(args)

    # 选择第一个客户端
    client_ids = list(federated_data['clients'].keys())
    selected_client = client_ids[0]
    print(f"选择客户端: {selected_client}")

    # 准备数据加载器
    loaders = prepare_data(federated_data, selected_client)
    print(f"训练样本: {len(loaders['train'].dataset)}")
    if 'val' in loaders:
        print(f"验证样本: {len(loaders['val'].dataset)}")

    # 创建训练器并开始训练
    trainer = SimpleTrainer(args)
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        epochs=50
    )

    # 测试
    if 'test' in loaders:
        test_loss = trainer.validate(loaders['test'])
        print(f"测试损失: {test_loss:.6f}")


if __name__ == "__main__":
    main()