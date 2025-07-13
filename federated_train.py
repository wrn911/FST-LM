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
from utils.utils import assign_model_to_client, cleanup_client_model


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
    """为所有客户端创建数据加载器 - 包含训练、验证、测试集"""
    client_loaders = {}

    for client_id, client_data in federated_data['clients'].items():
        sequences = client_data['sequences']
        client_loaders[client_id] = {}

        # 训练集
        if 'train' in sequences:
            X_train = torch.FloatTensor(sequences['train']['history'])
            y_train = torch.FloatTensor(sequences['train']['target'])
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            client_loaders[client_id]['train'] = DataLoader(
                train_dataset, batch_size=args.local_bs, shuffle=True
            )
            client_loaders[client_id]['num_samples'] = len(train_dataset)

        # 验证集
        if 'val' in sequences:
            X_val = torch.FloatTensor(sequences['val']['history'])
            y_val = torch.FloatTensor(sequences['val']['target'])
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            client_loaders[client_id]['val'] = DataLoader(
                val_dataset, batch_size=args.local_bs, shuffle=False
            )

        # 测试集
        if 'test' in sequences:
            X_test = torch.FloatTensor(sequences['test']['history'])
            y_test = torch.FloatTensor(sequences['test']['target'])
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            client_loaders[client_id]['test'] = DataLoader(
                test_dataset, batch_size=args.local_bs, shuffle=False
            )

    return client_loaders


def create_federated_clients(federated_data, client_loaders, args):
    """创建联邦客户端 - 支持多种数据加载器和真实数据"""
    clients = []

    for client_id in federated_data['clients'].keys():
        # 获取该客户端的真实坐标和流量统计信息
        client_data = federated_data['clients'][client_id]
        coordinates = client_data['coordinates']
        original_traffic_stats = client_data['original_traffic_stats']

        # 创建客户端，传递真实的坐标和流量数据
        client = FederatedClient(
            client_id=client_id,
            model=None,  # 暂时不分配模型
            data_loader=client_loaders[client_id]['train'],  # 主要训练数据
            args=args,
            coordinates=coordinates,  # 真实坐标
            original_traffic_stats=original_traffic_stats  # 真实流量统计
        )

        # 添加验证和测试数据加载器
        if 'val' in client_loaders[client_id]:
            client.val_loader = client_loaders[client_id]['val']
        if 'test' in client_loaders[client_id]:
            client.test_loader = client_loaders[client_id]['test']

        clients.append(client)

    return clients


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

    # 验证和展示真实数据（新增）
    from utils.utils import validate_real_data, print_real_data_summary

    if validate_real_data(federated_data):
        print_real_data_summary(federated_data)
    else:
        print("数据验证失败，请检查数据文件")
        return

    # 创建客户端数据加载器
    print("\n创建客户端数据加载器...")
    client_loaders = create_client_data_loaders(federated_data, args)

    # 初始化全局模型
    print("初始化全局模型...")
    global_model = Model(ModelConfig(args)).to(args.device)

    # 创建联邦客户端（现在会传递真实数据）
    print("创建联邦客户端...")
    clients = create_federated_clients(federated_data, client_loaders, args)
    print(f"成功创建 {len(clients)} 个客户端")

    # 打印客户端真实数据样本（新增）
    print("\n客户端真实数据样本:")
    sample_client = clients[0]
    print(f"  基站 {sample_client.client_id}:")
    print(f"    坐标: ({sample_client.coordinates['lng']:.3f}, {sample_client.coordinates['lat']:.3f})")
    traffic_stats = sample_client.get_real_traffic_stats()
    print(f"    流量统计: 均值={traffic_stats['mean']:.1f}, 趋势={traffic_stats['trend']}")
    print(f"    变异系数: {traffic_stats.get('coefficient_of_variation', 0):.3f}")

    # 创建联邦服务器
    server = FederatedServer(global_model, args)

    # 检查是否需要恢复训练
    start_round = 1
    best_val_loss = float('inf')

    if args.resume:
        try:
            start_round, best_val_loss = server.load_checkpoint(args.resume)
            print(f"从轮次 {start_round} 恢复训练，当前最佳验证损失: {best_val_loss:.6f}")
        except Exception as e:
            print(f"恢复检查点失败: {e}")
            print("将从头开始训练")
            start_round = 1
            best_val_loss = float('inf')

    # 开始联邦训练
    print(f"\n开始联邦训练 (从轮次 {start_round} 到 {args.rounds})...")

    for round_idx in range(start_round, args.rounds + 1):
        print(f"\n{'=' * 50}")
        print(f"联邦学习轮次: {round_idx}/{args.rounds}")

        # 显示当前显存使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3  # GB
            print(f"GPU显存使用: {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")

        # 执行一轮联邦学习
        round_results = server.federated_round(clients, round_idx)

        # 输出本轮结果（增强版）
        avg_loss = round_results['avg_client_loss']
        print(f"本轮平均客户端损失: {avg_loss:.6f}")

        # 如果有验证损失，也输出
        if 'val_loss' in round_results:
            print(f"本轮验证集损失: {round_results['val_loss']:.6f}")

        # 每隔一定轮数进行全局评估
        if round_idx % args.eval_every == 0:
            print("进行全局模型评估...")

            eval_clients = clients[:min(10, len(clients))]  # 增加评估客户端数量

            # 验证集评估
            if hasattr(eval_clients[0], 'val_loader'):
                val_loss, val_client_losses = server.evaluate_global_model_detailed(eval_clients, 'val')
                server.train_history['global_loss'].append(val_loss)
                print(f"全局验证损失: {val_loss:.6f}")

                # 保存最优模型
                if args.save_best_model and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"{args.save_dir}/best_model.pth"
                    server.save_best_model(best_model_path, val_loss, round_idx)
                    print(f"🎯 发现更优模型！验证损失: {val_loss:.6f}")

            # 如果也想看测试集表现（可选，但不用于模型选择）
            if hasattr(eval_clients[0], 'test_loader') and round_idx % (args.eval_every * 2) == 0:
                test_loss, _ = server.evaluate_global_model_detailed(eval_clients, 'test')
                print(f"当前测试损失: {test_loss:.6f} (仅供参考)")

        # 保存检查点
        if args.save_checkpoint and round_idx % args.checkpoint_interval == 0:
            checkpoint_path = f"{args.save_dir}/checkpoint_round_{round_idx}.pth"
            server.save_checkpoint(checkpoint_path, round_idx, best_val_loss)

    # 训练结束，保存最终检查点
    if args.save_checkpoint:
        final_checkpoint_path = f"{args.save_dir}/final_checkpoint.pth"
        server.save_checkpoint(final_checkpoint_path, args.rounds, best_val_loss)
        print(f"最终检查点已保存: {final_checkpoint_path}")

    print("\n联邦训练完成!")

    # 最终测试集评估（如果已实现）
    if hasattr(server, 'final_test_evaluation'):
        final_results = server.final_test_evaluation(clients)

    # 输出训练摘要
    train_history = server.get_train_history()

    print(f"\n{'=' * 60}")
    print("训练摘要")
    print(f"{'=' * 60}")

    # 训练损失
    final_client_loss = train_history['client_losses'][-1] if train_history['client_losses'] else float('inf')
    print(f"最终平均客户端训练损失: {final_client_loss:.6f}")

    # 验证损失
    if 'val_losses' in train_history and train_history['val_losses']:
        final_val_loss = train_history['val_losses'][-1]
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"最终验证损失: {final_val_loss:.6f}")

    # 全局评估损失
    if train_history['global_loss']:
        best_global_loss = min(train_history['global_loss'])
        print(f"最佳全局损失: {best_global_loss:.6f}")

    # 模型保存摘要
    if args.save_best_model:
        print(f"\n📁 模型保存信息:")
        print(f"   最优模型: {args.save_dir}/best_model.pth")
        print(f"   最优验证损失: {best_val_loss:.6f}")

    if args.save_checkpoint:
        print(f"   最终检查点: {args.save_dir}/final_checkpoint.pth")
        print(f"   检查点保存间隔: 每 {args.checkpoint_interval} 轮")

    # 如果使用多维度LLM聚合，生成趋势分析
    if args.aggregation == 'multi_dim_llm':
        print(f"\n{'=' * 60}")
        print("客户端学习趋势分析")
        print(f"{'=' * 60}")

        try:
            from utils.trend_visualizer import visualize_trends
            visualize_trends(server, args.save_dir)
        except Exception as e:
            print(f"趋势分析生成失败: {e}")
            # 至少打印基本的趋势摘要
            if hasattr(server, 'client_history') and server.client_history['losses']:
                print(f"参与训练的客户端数量: {len(server.client_history['losses'])}")

                # 显示各客户端的基本趋势信息
                for client_id in server.client_history['losses'].keys():
                    trend_summary = server.get_client_trend_summary(client_id)
                    print(f"  客户端 {client_id}: {trend_summary['description']} (评分: {trend_summary['score']:.2f})")

    # 保存详细结果
    if hasattr(args, 'save_results') and args.save_results:
        import json
        results_to_save = {
            'train_history': train_history,
            'args': vars(args)
        }

        # 添加趋势分析结果
        if args.aggregation == 'multi_dim_llm' and hasattr(server, 'client_history'):
            results_to_save['client_trends'] = {}
            for client_id in server.client_history['losses'].keys():
                trend_summary = server.get_client_trend_summary(client_id)
                results_to_save['client_trends'][client_id] = trend_summary

        # 添加最终测试结果
        if 'final_results' in locals():
            results_to_save['final_test_results'] = final_results

        with open(f"{args.save_dir}/training_results.json", 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"详细结果已保存至: {args.save_dir}/training_results.json")


if __name__ == "__main__":
    main()