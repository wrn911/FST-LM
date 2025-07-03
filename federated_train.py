# -*- coding: utf-8 -*-
"""
联邦学习训练主脚本
"""

import os
import sys
import torch
import random
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from models.TimeLLM import Model
from dataset.data_loader import get_federated_data
from utils.config import get_args
from federated.client import FederatedClient
from federated.server import FederatedServer
from torch.utils.data import DataLoader


class ModelConfig:
    """TimeLLM模型配置类 - 修复版本"""

    def __init__(self, args):
        self.task_name = 'long_term_forecast'
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = 1
        self.d_model = args.d_model
        self.n_heads = args.n_heads
        self.d_ff = args.d_model * 4

        # LLM配置 - 根据模型类型设置
        self.llm_model = getattr(args, 'llm_model_name', 'Qwen3')

        # 初始设置，会在模型初始化时更新为实际值
        if self.llm_model == 'Qwen3':
            self.llm_dim = 1024  # Qwen3-0.6B的隐藏维度（初始值）
            self.llm_layers = min(6, getattr(args, 'llm_layers', 6))
        elif self.llm_model == 'GPT2':
            self.llm_dim = 768
            self.llm_layers = 6
        else:
            self.llm_dim = 768  # 默认值
            self.llm_layers = 6

        # 补丁配置
        self.patch_len = 16
        self.stride = 8

        self.dropout = args.dropout
        self.prompt_domain = True
        self.content = "The dataset records the wireless traffic of a certain base station"

        # LoRA配置
        self.use_lora = getattr(args, 'use_lora', False)
        if self.use_lora:
            self.lora_rank = getattr(args, 'lora_rank', 8)
            self.lora_alpha = getattr(args, 'lora_alpha', 16)
            self.lora_dropout = getattr(args, 'lora_dropout', 0.1)
            self.lora_target_modules = getattr(args, 'lora_target_modules',
                                               ["q_proj", "k_proj", "v_proj", "o_proj"])


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_client_data_loaders(federated_data, args):
    """为所有客户端创建数据加载器"""
    client_loaders = {}

    for client_id, client_data in federated_data['clients'].items():
        sequences = client_data['sequences']

        # 训练集
        X_train = torch.FloatTensor(sequences['train']['history'])
        y_train = torch.FloatTensor(sequences['train']['target'])
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.local_bs,
            shuffle=True
        )

        client_loaders[client_id] = {
            'train': train_loader,
            'num_samples': len(train_dataset)
        }

        # 如果有测试集，也创建测试数据加载器
        if 'test' in sequences:
            X_test = torch.FloatTensor(sequences['test']['history'])
            y_test = torch.FloatTensor(sequences['test']['target'])
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.local_bs,
                shuffle=False
            )
            client_loaders[client_id]['test'] = test_loader

    return client_loaders


def create_federated_clients(federated_data, client_loaders, args):
    """创建联邦客户端 - 改为共享模型方式"""
    clients = []

    for client_id in federated_data['clients'].keys():
        # 创建客户端，但暂不分配模型（节省显存）
        client = FederatedClient(
            client_id=client_id,
            model=None,  # 暂时不分配模型
            data_loader=client_loaders[client_id]['train'],
            args=args
        )

        clients.append(client)

    return clients


def assign_model_to_client(client, model_template, global_params):
    """为客户端分配模型"""
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


def main():
    """主函数"""
    # 获取参数
    args = get_args()

    # 设置随机种子
    set_seed(args.seed)

    print("=== 联邦学习训练 ===")
    print(f"设备: {args.device}")
    print(f"客户端数量: {args.num_clients}")
    print(f"参与比例: {args.frac}")
    print(f"总轮数: {args.rounds}")
    print(f"本地训练轮数: {args.local_epochs}")
    print(f"聚合算法: {args.aggregation}")

    # 加载联邦数据
    print("\n加载联邦数据...")
    federated_data, _ = get_federated_data(args)

    # 创建客户端数据加载器
    print("创建客户端数据加载器...")
    client_loaders = create_client_data_loaders(federated_data, args)

    # 初始化全局模型
    print("初始化全局模型...")
    global_model = Model(ModelConfig(args)).to(args.device)

    # 创建联邦客户端
    print("创建联邦客户端...")
    clients = create_federated_clients(federated_data, client_loaders, args)
    print(f"成功创建 {len(clients)} 个客户端")

    # 创建联邦服务器
    server = FederatedServer(global_model, args)

    # 开始联邦训练
    print(f"\n开始联邦训练 ({args.rounds} 轮)...")

    for round_idx in range(1, args.rounds + 1):
        print(f"\n{'=' * 50}")
        print(f"联邦学习轮次: {round_idx}/{args.rounds}")

        # 显示当前显存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            print(f"GPU显存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

        # 执行一轮联邦学习
        round_results = server.federated_round(clients, round_idx)

        # 输出本轮结果
        avg_loss = round_results['avg_client_loss']
        print(f"本轮平均客户端损失: {avg_loss:.6f}")

        # 每隔一定轮数进行全局评估
        if round_idx % args.eval_every == 0:
            print("进行全局评估...")

            # 为评估临时分配模型
            global_params = server.get_global_model()
            eval_clients = clients[:min(5, len(clients))]  # 只用少数客户端评估以节省时间

            total_eval_loss = 0
            for client in eval_clients:
                assign_model_to_client(client, None, global_params)
                eval_loss = client.evaluate()
                total_eval_loss += eval_loss
                cleanup_client_model(client)

            global_loss = total_eval_loss / len(eval_clients)
            server.train_history['global_loss'].append(global_loss)
            print(f"全局模型损失: {global_loss:.6f}")

            del global_params

    print("\n联邦训练完成!")

    # 输出训练摘要
    train_history = server.get_train_history()
    if train_history['global_loss']:
        best_global_loss = min(train_history['global_loss'])
        print(f"最佳全局损失: {best_global_loss:.6f}")

    final_client_loss = train_history['client_losses'][-1] if train_history['client_losses'] else float('inf')
    print(f"最终平均客户端损失: {final_client_loss:.6f}")


if __name__ == "__main__":
    main()