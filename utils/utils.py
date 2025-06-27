# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import os
import json
import logging
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def setup_logging(args):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 设置日志级别
    log_level = getattr(logging, args.log_level.upper())

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler(os.path.join(args.exp_dir, 'training.log'))  # 文件输出
        ]
    )

    # 设置matplotlib和其他库的日志级别，避免过多输出
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def save_results(results: Dict, save_dir: str):
    """保存实验结果"""
    # 保存为JSON格式（便于查看）
    json_path = os.path.join(save_dir, 'results.json')

    # 创建可序列化的结果副本
    serializable_results = {}
    for key, value in results.items():
        if key == 'args':
            serializable_results[key] = value
        elif isinstance(value, (list, dict, str, int, float, bool)):
            serializable_results[key] = value
        else:
            # 对于复杂对象，尝试转换为基本类型
            try:
                serializable_results[key] = json.loads(json.dumps(value, default=str))
            except:
                serializable_results[key] = str(value)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    # 保存为pickle格式（保留完整数据）
    pickle_path = os.path.join(save_dir, 'results.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    logging.info(f"结果已保存到: {json_path} 和 {pickle_path}")


def plot_training_curves(training_history: List[Dict], save_dir: str):
    """绘制训练曲线"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    # 提取数据
    rounds = [stats['round'] for stats in training_history]
    train_losses = [stats['avg_train_loss'] for stats in training_history]
    val_losses = [stats['avg_val_loss'] for stats in training_history if stats['avg_val_loss'] is not None]
    val_rounds = [stats['round'] for stats in training_history if stats['avg_val_loss'] is not None]

    # 提取评估数据
    eval_rounds = []
    test_losses = []
    mse_values = []
    mae_values = []

    for stats in training_history:
        if 'eval_metrics' in stats:
            eval_rounds.append(stats['round'])
            test_losses.append(stats['eval_metrics']['avg_test_loss'])
            mse_values.append(stats['eval_metrics']['avg_mse'])
            mae_values.append(stats['eval_metrics']['avg_mae'])

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('联邦学习训练过程', fontsize=16, fontweight='bold')

    # 训练和验证损失
    axes[0, 0].plot(rounds, train_losses, label='训练损失', color='blue', alpha=0.8)
    if val_losses:
        axes[0, 0].plot(val_rounds, val_losses, label='验证损失', color='orange', alpha=0.8)
    if test_losses:
        axes[0, 0].plot(eval_rounds, test_losses, label='测试损失', color='red', alpha=0.8, marker='o', markersize=4)

    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].set_title('损失变化曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MSE曲线
    if mse_values:
        axes[0, 1].plot(eval_rounds, mse_values, label='MSE', color='green', marker='s', markersize=4)
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].set_title('均方误差变化')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # MAE曲线
    if mae_values:
        axes[1, 0].plot(eval_rounds, mae_values, label='MAE', color='purple', marker='^', markersize=4)
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('平均绝对误差变化')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 客户端参与情况
    client_counts = [stats['selected_clients'] for stats in training_history]
    axes[1, 1].plot(rounds, client_counts, label='参与客户端数', color='brown', marker='o', markersize=3)
    axes[1, 1].set_xlabel('轮次')
    axes[1, 1].set_ylabel('客户端数量')
    axes[1, 1].set_title('每轮参与客户端数量')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"训练曲线已保存到: {plot_path}")


def plot_client_performance(training_history: List[Dict], save_dir: str):
    """绘制客户端性能分布图"""
    # 提取最后一轮的客户端统计
    last_round_stats = training_history[-1]['client_stats']

    client_ids = [stat['client_id'] for stat in last_round_stats]
    train_losses = [stat['train_loss'] for stat in last_round_stats]
    sample_counts = [stat['total_samples'] for stat in last_round_stats]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 客户端训练损失分布
    axes[0].bar(range(len(client_ids)), train_losses, alpha=0.7, color='skyblue')
    axes[0].set_xlabel('客户端索引')
    axes[0].set_ylabel('训练损失')
    axes[0].set_title('各客户端最终训练损失')
    axes[0].grid(True, alpha=0.3)

    # 客户端样本数分布
    axes[1].bar(range(len(client_ids)), sample_counts, alpha=0.7, color='lightcoral')
    axes[1].set_xlabel('客户端索引')
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('各客户端训练样本数量')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, 'client_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"客户端性能图已保存到: {plot_path}")


def calculate_communication_cost(training_history: List[Dict], model_size_mb: float):
    """计算通信成本"""
    total_rounds = len(training_history)
    total_clients_participated = sum([stats['selected_clients'] for stats in training_history])

    # 每轮通信成本 = 参与客户端数量 × 2 × 模型大小 (上传 + 下载)
    total_communication_mb = total_clients_participated * 2 * model_size_mb

    communication_stats = {
        'total_rounds': total_rounds,
        'total_clients_participated': total_clients_participated,
        'model_size_mb': model_size_mb,
        'total_communication_mb': total_communication_mb,
        'avg_communication_per_round_mb': total_communication_mb / total_rounds,
        'communication_efficiency': total_communication_mb / total_rounds / sum(
            [stats['selected_clients'] for stats in training_history]) * total_rounds
    }

    return communication_stats


def generate_summary_report(results: Dict, save_dir: str):
    """生成实验总结报告"""
    report_path = os.path.join(save_dir, 'summary_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("联邦学习实验总结报告\n")
        f.write("=" * 80 + "\n\n")

        # 实验配置
        f.write("1. 实验配置\n")
        f.write("-" * 40 + "\n")
        args = results['args']
        f.write(f"数据集: {args['file_path']}\n")
        f.write(f"客户端数量: {args['num_clients']}\n")
        f.write(f"参与比例: {args['frac']}\n")
        f.write(f"总轮数: {args['rounds']}\n")
        f.write(f"本地训练轮数: {args['local_epochs']}\n")
        f.write(f"批处理大小: {args['local_bs']}\n")
        f.write(f"聚合算法: {args['aggregation']}\n")
        f.write(f"学习率: {args['lr']}\n")
        f.write(f"模型维度: {args['d_model']}\n")
        f.write(f"历史序列长度: {args['seq_len']}\n")
        f.write(f"预测序列长度: {args['pred_len']}\n\n")

        # 训练结果
        f.write("2. 训练结果\n")
        f.write("-" * 40 + "\n")
        f.write(f"训练时间: {results['training_time'] / 60:.1f} 分钟\n")
        f.write(f"最佳损失: {results['best_loss']:.6f}\n")

        final_eval = results['final_evaluation']
        f.write(f"最终测试损失: {final_eval['avg_test_loss']:.6f}\n")
        f.write(f"最终MSE: {final_eval['avg_mse']:.6f}\n")
        f.write(f"最终MAE: {final_eval['avg_mae']:.6f}\n")
        f.write(f"最终RMSE: {final_eval['avg_rmse']:.6f}\n")
        f.write(f"最终MAPE: {final_eval['avg_mape']:.2f}%\n\n")

        # 客户端性能
        f.write("3. 客户端性能统计\n")
        f.write("-" * 40 + "\n")
        client_results = final_eval['client_results']

        mse_values = [r['mse'] for r in client_results]
        mae_values = [r['mae'] for r in client_results]

        f.write(f"MSE - 最小: {min(mse_values):.6f}, 最大: {max(mse_values):.6f}, "
                f"标准差: {np.std(mse_values):.6f}\n")
        f.write(f"MAE - 最小: {min(mae_values):.6f}, 最大: {max(mae_values):.6f}, "
                f"标准差: {np.std(mae_values):.6f}\n\n")

        # 收敛性分析
        f.write("4. 收敛性分析\n")
        f.write("-" * 40 + "\n")
        training_history = results['training_history']
        initial_loss = training_history[0]['avg_train_loss']
        final_loss = training_history[-1]['avg_train_loss']
        improvement = (initial_loss - final_loss) / initial_loss * 100

        f.write(f"初始训练损失: {initial_loss:.6f}\n")
        f.write(f"最终训练损失: {final_loss:.6f}\n")
        f.write(f"损失改善幅度: {improvement:.2f}%\n")

        # 找到收敛轮次（损失变化小于阈值的连续轮数）
        convergence_round = None
        threshold = 0.001
        consecutive_rounds = 5

        for i in range(consecutive_rounds, len(training_history)):
            recent_losses = [training_history[j]['avg_train_loss']
                             for j in range(i - consecutive_rounds, i)]
            if max(recent_losses) - min(recent_losses) < threshold:
                convergence_round = i - consecutive_rounds + 1
                break

        if convergence_round:
            f.write(f"收敛轮次: {convergence_round} (损失变化 < {threshold})\n")
        else:
            f.write("在训练过程中未明显收敛\n")

        f.write("\n" + "=" * 80 + "\n")

    logging.info(f"总结报告已保存到: {report_path}")


def load_results(exp_dir: str) -> Dict:
    """加载实验结果"""
    pickle_path = os.path.join(exp_dir, 'results.pkl')

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    else:
        json_path = os.path.join(exp_dir, 'results.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"未找到结果文件: {exp_dir}")


def compare_experiments(exp_dirs: List[str], save_dir: str):
    """比较多个实验结果"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('实验结果对比', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for i, exp_dir in enumerate(exp_dirs):
        try:
            results = load_results(exp_dir)
            training_history = results['training_history']

            exp_name = os.path.basename(exp_dir)
            color = colors[i % len(colors)]

            rounds = [stats['round'] for stats in training_history]
            train_losses = [stats['avg_train_loss'] for stats in training_history]

            # 训练损失对比
            axes[0, 0].plot(rounds, train_losses, label=exp_name, color=color, alpha=0.8)

            # 最终评估对比
            final_eval = results['final_evaluation']
            metrics = ['avg_mse', 'avg_mae', 'avg_rmse']
            values = [final_eval[metric] for metric in metrics]

            x_pos = np.arange(len(metrics))
            axes[0, 1].bar(x_pos + i * 0.15, values, width=0.15, label=exp_name, alpha=0.8)

        except Exception as e:
            logging.warning(f"无法加载实验 {exp_dir}: {e}")

    # 设置图表
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('训练损失')
    axes[0, 0].set_title('训练损失对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('指标')
    axes[0, 1].set_ylabel('值')
    axes[0, 1].set_title('最终评估指标对比')
    axes[0, 1].set_xticks(np.arange(len(metrics)))
    axes[0, 1].set_xticklabels(['MSE', 'MAE', 'RMSE'])
    axes[0, 1].legend()

    plt.tight_layout()

    comparison_path = os.path.join(save_dir, 'experiments_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"实验对比图已保存到: {comparison_path}")


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """打印进度条"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()