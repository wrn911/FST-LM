# -*- coding: utf-8 -*-
"""
客户端趋势可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json


class TrendVisualizer:
    """客户端趋势可视化器"""

    def __init__(self, save_dir='results'):
        self.save_dir = save_dir

    def plot_client_trends(self, server_instance, save_path=None):
        """绘制所有客户端的损失趋势"""
        client_history = server_instance.client_history

        if not client_history['losses']:
            print("没有客户端历史数据可以绘制")
            return

        # 创建子图
        num_clients = len(client_history['losses'])
        cols = min(3, num_clients)
        rows = (num_clients + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if num_clients == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (client_id, losses) in enumerate(client_history['losses'].items()):
            if i >= len(axes):
                break

            ax = axes[i]

            # 绘制损失曲线
            rounds = range(1, len(losses) + 1)
            ax.plot(rounds, losses, 'b-o', markersize=4, linewidth=2, label='训练损失')

            # 添加趋势线
            if len(losses) >= 3:
                x = np.array(rounds)
                y = np.array(losses)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), 'r--', alpha=0.7, linewidth=1, label=f'趋势线 (斜率:{z[0]:.4f})')

            # 获取趋势摘要
            trend_summary = server_instance.get_client_trend_summary(client_id)

            # 设置标题和标签
            ax.set_title(
                f'客户端 {client_id}\n趋势: {trend_summary["description"]} (评分: {trend_summary["score"]:.2f})')
            ax.set_xlabel('训练轮次')
            ax.set_ylabel('损失值')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 隐藏多余的子图
        for i in range(num_clients, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"趋势图已保存至: {save_path}")

        plt.show()

    def plot_trend_distribution(self, server_instance, save_path=None):
        """绘制趋势分布统计"""
        client_history = server_instance.client_history

        if not client_history['losses']:
            print("没有客户端历史数据可以分析")
            return

        # 收集所有客户端的趋势信息
        trend_descriptions = []
        trend_scores = []
        improvement_rates = []

        for client_id in client_history['losses'].keys():
            trend_summary = server_instance.get_client_trend_summary(client_id)
            trend_descriptions.append(trend_summary['description'])
            trend_scores.append(trend_summary['score'])

            # 计算改进率
            losses = client_history['losses'][client_id]
            if len(losses) >= 2:
                improvement_rate = (losses[0] - losses[-1]) / losses[0] * 100
                improvement_rates.append(improvement_rate)

        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 趋势描述分布
        from collections import Counter
        trend_counts = Counter(trend_descriptions)
        ax1.bar(trend_counts.keys(), trend_counts.values(), color='skyblue', alpha=0.7)
        ax1.set_title('客户端趋势分布')
        ax1.set_ylabel('客户端数量')
        ax1.tick_params(axis='x', rotation=45)

        # 2. 趋势评分分布
        ax2.hist(trend_scores, bins=10, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('趋势评分分布')
        ax2.set_xlabel('趋势评分')
        ax2.set_ylabel('客户端数量')
        ax2.axvline(np.mean(trend_scores), color='red', linestyle='--', label=f'平均值: {np.mean(trend_scores):.2f}')
        ax2.legend()

        # 3. 改进率分布
        if improvement_rates:
            ax3.hist(improvement_rates, bins=10, color='orange', alpha=0.7, edgecolor='black')
            ax3.set_title('损失改进率分布')
            ax3.set_xlabel('改进率 (%)')
            ax3.set_ylabel('客户端数量')
            ax3.axvline(0, color='black', linestyle='-', alpha=0.5, label='无改进线')
            ax3.axvline(np.mean(improvement_rates), color='red', linestyle='--',
                        label=f'平均改进率: {np.mean(improvement_rates):.1f}%')
            ax3.legend()

        # 4. 参与度 vs 趋势评分散点图
        participation_counts = []
        for client_id in client_history['losses'].keys():
            participation_counts.append(client_history['participation_count'].get(client_id, 0))

        scatter = ax4.scatter(participation_counts, trend_scores, c=trend_scores,
                              cmap='viridis', alpha=0.7, s=60)
        ax4.set_title('参与度 vs 趋势评分')
        ax4.set_xlabel('参与轮数')
        ax4.set_ylabel('趋势评分')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='趋势评分')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"趋势分布图已保存至: {save_path}")

        plt.show()

    def generate_trend_report(self, server_instance, save_path=None):
        """生成趋势分析报告"""
        client_history = server_instance.client_history

        if not client_history['losses']:
            print("没有客户端历史数据可以生成报告")
            return

        report = {
            'summary': {},
            'clients': {},
            'statistics': {}
        }

        # 收集所有趋势信息
        all_scores = []
        all_descriptions = []
        all_improvements = []

        for client_id in client_history['losses'].keys():
            trend_summary = server_instance.get_client_trend_summary(client_id)
            losses = client_history['losses'][client_id]
            participation = client_history['participation_count'].get(client_id, 0)

            # 客户端详细信息
            client_info = {
                'trend_description': trend_summary['description'],
                'trend_score': trend_summary['score'],
                'participation_count': participation,
                'loss_history': losses,
                'initial_loss': losses[0] if losses else None,
                'latest_loss': losses[-1] if losses else None,
                'best_loss': min(losses) if losses else None,
                'worst_loss': max(losses) if losses else None
            }

            if len(losses) >= 2:
                improvement_rate = (losses[0] - losses[-1]) / losses[0] * 100
                client_info['improvement_rate'] = improvement_rate
                all_improvements.append(improvement_rate)

            report['clients'][client_id] = client_info
            all_scores.append(trend_summary['score'])
            all_descriptions.append(trend_summary['description'])

        # 统计信息
        from collections import Counter
        description_counts = Counter(all_descriptions)

        report['statistics'] = {
            'total_clients': len(client_history['losses']),
            'avg_trend_score': float(np.mean(all_scores)),
            'std_trend_score': float(np.std(all_scores)),
            'trend_distribution': dict(description_counts),
            'avg_improvement_rate': float(np.mean(all_improvements)) if all_improvements else 0.0,
            'clients_improving': sum(1 for desc in all_descriptions if 'improving' in desc),
            'clients_stable': sum(1 for desc in all_descriptions if 'stable' in desc),
            'clients_deteriorating': sum(1 for desc in all_descriptions if 'deteriorating' in desc)
        }

        # 摘要
        best_client = max(report['clients'].items(), key=lambda x: x[1]['trend_score'])
        worst_client = min(report['clients'].items(), key=lambda x: x[1]['trend_score'])

        report['summary'] = {
            'best_performing_client': {
                'client_id': best_client[0],
                'score': best_client[1]['trend_score'],
                'description': best_client[1]['trend_description']
            },
            'worst_performing_client': {
                'client_id': worst_client[0],
                'score': worst_client[1]['trend_score'],
                'description': worst_client[1]['trend_description']
            },
            'overall_health': 'good' if report['statistics']['avg_trend_score'] > 1.1 else
            'fair' if report['statistics']['avg_trend_score'] > 0.9 else 'poor'
        }

        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"趋势报告已保存至: {save_path}")

        # 打印摘要
        self._print_report_summary(report)

        return report

    def _print_report_summary(self, report):
        """打印报告摘要"""
        print("\n" + "=" * 60)
        print("客户端趋势分析报告摘要")
        print("=" * 60)

        stats = report['statistics']
        summary = report['summary']

        print(f"总客户端数量: {stats['total_clients']}")
        print(f"平均趋势评分: {stats['avg_trend_score']:.3f} (±{stats['std_trend_score']:.3f})")
        print(f"平均改进率: {stats['avg_improvement_rate']:.1f}%")
        print(f"整体健康状况: {summary['overall_health']}")

        print(f"\n趋势分布:")
        print(f"  改进中: {stats['clients_improving']} 个客户端")
        print(f"  稳定: {stats['clients_stable']} 个客户端")
        print(f"  恶化中: {stats['clients_deteriorating']} 个客户端")

        print(f"\n最佳客户端: {summary['best_performing_client']['client_id']}")
        print(f"  趋势: {summary['best_performing_client']['description']}")
        print(f"  评分: {summary['best_performing_client']['score']:.3f}")

        print(f"\n最差客户端: {summary['worst_performing_client']['client_id']}")
        print(f"  趋势: {summary['worst_performing_client']['description']}")
        print(f"  评分: {summary['worst_performing_client']['score']:.3f}")


def visualize_trends(server_instance, save_dir='results'):
    """便捷函数：生成所有趋势可视化"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    visualizer = TrendVisualizer(save_dir)

    # 生成所有图表和报告
    visualizer.plot_client_trends(server_instance, f"{save_dir}/client_trends.png")
    visualizer.plot_trend_distribution(server_instance, f"{save_dir}/trend_distribution.png")
    visualizer.generate_trend_report(server_instance, f"{save_dir}/trend_report.json")

    print(f"\n所有趋势分析结果已保存至 {save_dir}/ 目录")