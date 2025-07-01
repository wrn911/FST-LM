# -*- coding: utf-8 -*-
"""
配置参数管理模块
"""
import argparse
import os
import torch


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='联邦学习无线流量预测')

    # === 数据相关参数 ===
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='数据集目录路径')
    parser.add_argument('--file_path', type=str, default='milano.h5',
                        help='HDF5数据文件名')
    parser.add_argument('--data_type', type=str, default='net',
                        help='流量类型 (net/call/sms)')

    # === 时序参数 ===
    parser.add_argument('--seq_len', type=int, default=96,
                        help='历史序列长度 (小时)')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='预测序列长度 (小时)')
    parser.add_argument('--test_days', type=int, default=7,
                        help='测试集天数')
    parser.add_argument('--val_days', type=int, default=1,
                        help='验证集天数')

    # === 联邦学习参数 ===
    parser.add_argument('--num_clients', type=int, default=50,
                        help='参与联邦学习的基站数量')
    parser.add_argument('--frac', type=float, default=0.3,
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--rounds', type=int, default=100,
                        help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='本地训练轮数')
    parser.add_argument('--local_bs', type=int, default=32,
                        help='本地批处理大小')

    # === 模型参数 ===
    parser.add_argument('--d_model', type=int, default=64,
                        help='模型隐藏维度')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Transformer层数')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout概率')

    # === 训练参数 ===
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    # === 聚合参数 ===
    parser.add_argument('--aggregation', type=str, default='fedavg',
                        choices=['fedavg', 'weighted'],
                        help='聚合算法')
    parser.add_argument('--use_coordinates', action='store_true',
                        help='是否使用坐标信息进行加权聚合')

    # === 系统参数 ===
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='日志级别')

    # === 评估参数 ===
    parser.add_argument('--eval_every', type=int, default=10,
                        help='每隔多少轮评估一次')
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存模型')

    # === TimeLLM特定参数 ===
    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'lstm', 'mlp', 'timellm'],
                        help='模型类型')
    parser.add_argument('--llm_model', type=str, default='GPT2',
                        choices=['GPT2', 'LLAMA', 'BERT'],
                        help='LLM模型类型')
    parser.add_argument('--llm_dim', type=int, default=768,
                        help='LLM模型维度 (GPT2-small:768, LLAMA7b:4096)')
    parser.add_argument('--llm_layers', type=int, default=6,
                        help='使用的LLM层数')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch长度')
    parser.add_argument('--stride', type=int, default=8,
                        help='Patch步长')
    parser.add_argument('--prompt_domain', type=int, default=0,
                        help='是否使用领域提示')

    # TimeLLM必需的额外参数（参考原始代码）
    parser.add_argument('--d_ff', type=int, default=32,
                        help='dimension of fcn')
    parser.add_argument('--label_len', type=int, default=48,
                        help='start token length')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT',
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu',
                        help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='whether to output attention in encoder')
    parser.add_argument('--factor', type=int, default=1,
                        help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='window size of moving average')

    # TimeLLM原始训练参数
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1,
                        help='num of decoder layers')

    args = parser.parse_args()

    # 设备自动检测
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def print_args(args):
    """打印配置参数"""
    print("=" * 80)
    print("联邦学习配置参数")
    print("=" * 80)

    print(f"数据配置:")
    print(f"  数据文件: {args.file_path}")
    print(f"  客户端数量: {args.num_clients}")
    print(f"  历史长度: {args.seq_len}小时")
    print(f"  预测长度: {args.pred_len}小时")

    print(f"\n联邦学习配置:")
    print(f"  总轮数: {args.rounds}")
    print(f"  参与比例: {args.frac}")
    print(f"  本地训练轮数: {args.local_epochs}")
    print(f"  批处理大小: {args.local_bs}")
    print(f"  聚合算法: {args.aggregation}")

    print(f"\n模型配置:")
    print(f"  隐藏维度: {args.d_model}")
    print(f"  注意力头数: {args.n_heads}")
    print(f"  层数: {args.n_layers}")
    print(f"  学习率: {args.lr}")

    print(f"\n系统配置:")
    print(f"  设备: {args.device}")
    print(f"  随机种子: {args.seed}")
    print(f"  保存目录: {args.save_dir}")

    if args.model_type == 'timellm':
        print(f"  LLM模型: {args.llm_model}")
        print(f"  LLM维度: {args.llm_dim}")
        print(f"  LLM层数: {args.llm_layers}")
        print(f"  Patch长度: {args.patch_len}")
        print(f"  Patch步长: {args.stride}")

    print("=" * 80)