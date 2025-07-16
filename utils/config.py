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
    parser.add_argument('--num_clients', type=int, default=10,
                        help='参与联邦学习的基站数量')
    parser.add_argument('--frac', type=float, default=0.3,
                        help='每轮参与训练的客户端比例')
    parser.add_argument('--rounds', type=int, default=100,
                        help='联邦学习轮数')
    parser.add_argument('--local_epochs', type=int, default=5,
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
    parser.add_argument('--llm_model_name', type=str, default='Qwen3',
                        choices=['Qwen3', 'GPT2', 'BERT', 'LLAMA', 'DeepSeek'],
                        help='使用的LLM模型')
    parser.add_argument('--llm_layers', type=int, default=12,
                        help='LLM模型层数（Qwen3-0.6B建议4-6层）')

    # === LoRA参数 ===
    parser.add_argument('--use_lora', action='store_true',
                        help='是否使用LoRA进行参数高效微调')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA秩 (r参数，控制适应矩阵的秩)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA缩放参数 (alpha，通常设为2*rank)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA层的dropout率')
    parser.add_argument('--lora_target_modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj',
                        help='LoRA目标模块，逗号分隔 (Qwen3默认: q_proj,k_proj,v_proj,o_proj)')

    # === 训练参数 ===
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')

    # === 数据增强参数 ===
    parser.add_argument('--enable_augmentation', action='store_true',
                        help='是否启用数据增强策略')
    parser.add_argument('--mixup_prob', type=float, default=0.2,
                        help='Mixup增强概率')
    parser.add_argument('--jittering_prob', type=float, default=0.15,
                        help='Jittering增强概率')
    parser.add_argument('--scaling_prob', type=float, default=0.1,
                        help='Scaling增强概率')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                        help='增强样本占总样本的比例')
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                        help='相似性阈值（皮尔逊相关系数）')
    parser.add_argument('--candidate_pool_size', type=int, default=5,
                        help='候选池大小')
    parser.add_argument('--augmentation_lambda_min', type=float, default=0.6,
                        help='Mixup lambda值下限')
    parser.add_argument('--augmentation_lambda_max', type=float, default=0.8,
                        help='Mixup lambda值上限')
    parser.add_argument('--enable_regularization_constraints', action='store_true', default=True,
                        help='是否启用正则化约束机制')
    parser.add_argument('--max_deviation_ratio', type=float, default=0.3,
                        help='统计特征最大偏离比例')
    parser.add_argument('--min_correlation_threshold', type=float, default=0.5,
                        help='最小时序相关性阈值')
    parser.add_argument('--constraint_correction_weight', type=float, default=0.3,
                        help='约束修正时的原始数据权重')

    # === 聚合参数 ===
    parser.add_argument('--aggregation', type=str, default='enhanced_multi_dim_llm',
                        choices=['fedavg', 'weighted', 'lora_fedavg', 'llm_fedavg', 'lora_fedprox',
                                 'layer_aware_llm', 'multi_dim_llm', 'enhanced_multi_dim_llm'],  # 新增选项
                        help='聚合算法')

    # === FedProx参数 ===
    parser.add_argument('--use_fedprox', action='store_true',
                        help='是否使用FedProx算法')
    parser.add_argument('--fedprox_mu', type=float, default=0.1,
                        help='FedProx正则化参数μ')

    # === 增强版多维度LLM聚合参数 ===
    parser.add_argument('--enhanced_multi_dim_dimensions', type=str,
                        default='model_performance,data_quality,spatial_distribution,temporal_stability,traffic_pattern',
                        help='增强版多维度聚合的评分维度，逗号分隔（5个实用专家）')

    # 各维度初始权重（5个专家）
    parser.add_argument('--model_performance_weight', type=float, default=0.35,
                        help='模型性能维度权重')
    parser.add_argument('--data_quality_weight', type=float, default=0.25,
                        help='数据质量维度权重')
    parser.add_argument('--spatial_distribution_weight', type=float, default=0.15,
                        help='空间分布维度权重')
    parser.add_argument('--temporal_stability_weight', type=float, default=0.15,
                        help='时序稳定性维度权重')
    parser.add_argument('--traffic_pattern_weight', type=float, default=0.10,
                        help='流量模式维度权重')

    # 评分策略参数
    parser.add_argument('--enable_dimension_analysis', action='store_true',
                        help='是否启用维度分析日志')
    parser.add_argument('--expert_verbose', action='store_true', default=True,
                        help='是否显示详细的专家决策过程')
    parser.add_argument('--expert_temperature', type=float, default=1.2,
                        help='专家评分的softmax温度参数')
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                        help='数据质量阈值（变异系数）')

    # === LLM聚合参数 ===
    parser.add_argument('--llm_api_key', type=str, default="118aea86606f4e2f82750c54d3bb380c.DxtxnaKQhFz5EHPY",
                        help='DeepSeek API密钥 (用于智能聚合)')
    parser.add_argument('--llm_model', type=str, default='glm-4-flash-250414',
                        help='使用的LLM模型名称')
    parser.add_argument('--llm_cache_rounds', type=int, default=1,
                        help='LLM权重缓存轮数（1表示每轮都调用）')
    parser.add_argument('--llm_min_confidence', type=float, default=0.7,
                        help='LLM决策最小置信度阈值')

    # === 层级感知聚合参数 ===
    parser.add_argument('--layer_analysis_enabled', action='store_true',
                        help='是否启用层级感知聚合分析')
    parser.add_argument('--layer_aware_cache_rounds', type=int, default=1,
                        help='层级感知聚合的缓存轮数')

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

    # === 模型保存和恢复参数 ===
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='是否保存训练检查点')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='每隔多少轮保存一次检查点')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--save_best_model', action='store_true', default=True,
                        help='是否保存最优验证模型')

    # === 动态权重融合参数 ===
    parser.add_argument('--alpha_max', type=float, default=0.9,
                        help='前期LLM权重上限')
    parser.add_argument('--alpha_min', type=float, default=0.2,
                        help='后期LLM权重下限')
    parser.add_argument('--decay_type', type=str, default='sigmoid',
                        choices=['sigmoid', 'exponential', 'linear'],
                        help='衰减类型')
    parser.add_argument('--base_constraint', type=float, default=0.25,
                        help='基础约束强度')

    # === 历史贡献度参数 ===
    parser.add_argument('--stability_weight', type=float, default=0.4,
                        help='参与稳定性权重')
    parser.add_argument('--quality_weight', type=float, default=0.35,
                        help='梯度质量权重')
    parser.add_argument('--consistency_weight', type=float, default=0.25,
                        help='协作一致性权重')
    parser.add_argument('--min_safe_weight', type=float, default=0.05,
                        help='最小安全权重')

    args = parser.parse_args()

    # 设备自动检测
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 处理LoRA目标模块
    if args.lora_target_modules:
        args.lora_target_modules = [m.strip() for m in args.lora_target_modules.split(',')]

    # 处理增强版多维度聚合参数
    if args.enhanced_multi_dim_dimensions:
        args.enhanced_multi_dim_dimensions = [d.strip() for d in args.enhanced_multi_dim_dimensions.split(',')]

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