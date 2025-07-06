# -*- coding: utf-8 -*-
"""
联邦学习辅助工具函数
"""

import torch
from models.TimeLLM import Model


def assign_model_to_client(client, model_template, global_params):
    """为客户端分配模型"""
    from federated_train import ModelConfig  # 局部导入

    # 创建新的模型实例
    client_model = Model(ModelConfig(client.args)).to(client.args.device)
    client_model.load_state_dict(global_params)

    # 分配给客户端
    client.model = client_model
    client.optimizer = torch.optim.Adam(
        client.model.parameters(),
        lr=client.args.lr,
        weight_decay=client.args.weight_decay
    )
    client.criterion = torch.nn.MSELoss()


def cleanup_client_model(client):
    """清理客户端模型以释放显存"""
    if client.model is not None:
        del client.model
        del client.optimizer
        del client.criterion
        client.model = None
        client.optimizer = None
        client.criterion = None

    # 强制垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()