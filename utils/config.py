# -*- coding: utf-8 -*-
"""
配置参数管理模块 - 添加LoRA支持
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
    parser.add_argument('--frac', type=float, default=0.1,
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

    # === LoRA参数 ===
    parser.add_argument('--use_lora', action='store_true',
                        help='是否使用LoRA进行参数高效微调')
    parser.add_argument('--lora_rank', type=int, default=16,
                        help='LoRA秩 (r参数，控制适应矩阵的秩)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA缩放参数 (alpha，通常设为2*rank)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA层的dropout率')
    parser.add_argument('--lora_target_modules', type=str,
                        default='c_attn,c_proj',
                        help='LoRA目标模块，逗号分隔 (GPT2默认: c_attn,c_proj)')

    # === 训练参数 ===
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    # === 聚合参数 ===
    parser.add_argument('--aggregation', type=str, default='fedavg',
                        choices=['fedavg', 'weighted', 'lora_fedavg', 'llm_fedavg'],
                        help='聚合算法')
    parser.add_argument('--use_coordinates', action='store_true',
                        help='是否使用坐标信息进行加权聚合')

    # === LLM聚合参数 ===
    parser.add_argument('--llm_api_key', type=str, default=None,
                        help='LLM API密钥 (用于智能聚合)')
    parser.add_argument('--llm_model', type=str, default='gemini-2.5-flash',
                        help='使用的LLM模型名称')
    parser.add_argument('--llm_cache_rounds', type=int, default=1,
                        help='LLM权重缓存轮数（1表示每轮都调用）')
    parser.add_argument('--llm_min_confidence', type=float, default=0.7,
                        help='LLM决策最小置信度阈值')

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

    args = parser.parse_args()

    # 设备自动检测
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 处理LoRA目标模块
    if args.lora_target_modules:
        args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

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

    print(f"\nLoRA配置:")
    print(f"  使用LoRA: {args.use_lora}")
    if args.use_lora:
        print(f"  LoRA秩: {args.lora_rank}")
        print(f"  LoRA alpha: {args.lora_alpha}")
        print(f"  LoRA dropout: {args.lora_dropout}")
        print(f"  目标模块: {args.lora_target_modules}")

    print(f"\nLLM聚合配置:")
    if args.aggregation == 'llm_fedavg':
        print(f"  使用LLM聚合: 是")
        print(f"  LLM模型: {args.llm_model}")
        print(f"  缓存轮数: {args.llm_cache_rounds}")
        print(f"  最小置信度: {args.llm_min_confidence}")
        print(f"  API密钥: {'已设置' if args.llm_api_key else '未设置'}")
    else:
        print(f"  使用LLM聚合: 否")

    print(f"\n系统配置:")
    print(f"  设备: {args.device}")
    print(f"  随机种子: {args.seed}")
    print(f"  保存目录: {args.save_dir}")

    print("=" * 80)