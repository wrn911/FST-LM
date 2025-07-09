# -*- coding: utf-8 -*-
"""
联邦学习辅助工具函数 - 支持真实数据
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


def print_real_data_summary(federated_data):
    """打印真实数据摘要信息"""
    print("\n" + "=" * 60)
    print("真实数据摘要")
    print("=" * 60)

    coordinates = federated_data['coordinates']
    traffic_stats = federated_data['original_traffic_stats']

    # 坐标范围
    lngs = [coord['lng'] for coord in coordinates.values()]
    lats = [coord['lat'] for coord in coordinates.values()]

    print(f"坐标范围:")
    print(f"  经度: {min(lngs):.3f} ~ {max(lngs):.3f}")
    print(f"  纬度: {min(lats):.3f} ~ {max(lats):.3f}")

    # 流量统计摘要
    means = [stats['mean'] for stats in traffic_stats.values()]
    stds = [stats['std'] for stats in traffic_stats.values()]
    trends = [stats['trend'] for stats in traffic_stats.values()]

    print(f"\n流量统计:")
    print(f"  平均流量范围: {min(means):.1f} ~ {max(means):.1f}")
    print(f"  流量标准差范围: {min(stds):.1f} ~ {max(stds):.1f}")

    # 趋势分布
    from collections import Counter
    trend_counts = Counter(trends)
    print(f"  趋势分布: {dict(trend_counts)}")

    # 随机展示几个基站的详细信息
    import random
    sample_clients = random.sample(list(traffic_stats.keys()), min(3, len(traffic_stats)))

    print(f"\n随机样本详情:")
    for client_id in sample_clients:
        coord = coordinates[client_id]
        stats = traffic_stats[client_id]
        print(f"  基站 {client_id}:")
        print(f"    位置: ({coord['lng']:.3f}, {coord['lat']:.3f})")
        print(f"    流量: 均值={stats['mean']:.1f}, 标准差={stats['std']:.1f}")
        print(f"    趋势: {stats['trend']} (斜率={stats['trend_slope']:.4f})")
        print(f"    变异系数: {stats['coefficient_of_variation']:.3f}")


def validate_real_data(federated_data):
    """验证真实数据的完整性"""
    print("\n验证真实数据...")

    coordinates = federated_data['coordinates']
    traffic_stats = federated_data['original_traffic_stats']
    client_ids = federated_data['metadata']['client_ids']

    issues = []

    # 检查数据完整性
    for client_id in client_ids:
        # 检查坐标
        if client_id not in coordinates:
            issues.append(f"客户端 {client_id} 缺少坐标信息")
        else:
            coord = coordinates[client_id]
            if not (-180 <= coord['lng'] <= 180) or not (-90 <= coord['lat'] <= 90):
                issues.append(f"客户端 {client_id} 坐标超出合理范围")

        # 检查流量统计
        if client_id not in traffic_stats:
            issues.append(f"客户端 {client_id} 缺少流量统计信息")
        else:
            stats = traffic_stats[client_id]
            required_fields = ['mean', 'std', 'trend', 'min', 'max']
            for field in required_fields:
                if field not in stats:
                    issues.append(f"客户端 {client_id} 缺少流量字段: {field}")

    if issues:
        print("发现数据问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ 真实数据验证通过")
        return True