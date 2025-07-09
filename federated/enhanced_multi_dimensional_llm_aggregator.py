# -*- coding: utf-8 -*-
"""
增强版多维度LLM辅助联邦聚合器 - 基于真实流量数据的专家评分系统
"""

import json
import copy
import numpy as np
from typing import List, Dict, Optional
import torch
import logging
import hashlib
from .aggregation import FedAvgAggregator, LoRAFedAvgAggregator


class ExpertOutputConfig:
    """专家输出控制配置"""

    def __init__(self,
                 show_expert_process: bool = True,
                 show_llm_response: bool = True,
                 show_detailed_analysis: bool = True,
                 show_aggregation_process: bool = True,
                 show_consensus_analysis: bool = True):
        self.show_expert_process = show_expert_process  # 显示专家评分过程
        self.show_llm_response = show_llm_response  # 显示LLM完整响应
        self.show_detailed_analysis = show_detailed_analysis  # 显示详细分析
        self.show_aggregation_process = show_aggregation_process  # 显示聚合过程
        self.show_consensus_analysis = show_consensus_analysis  # 显示专家一致性分析

    @classmethod
    def minimal(cls):
        """最小输出模式 - 只显示关键结果"""
        return cls(
            show_expert_process=True,
            show_llm_response=False,
            show_detailed_analysis=False,
            show_aggregation_process=False,
            show_consensus_analysis=False
        )

    @classmethod
    def detailed(cls):
        """详细输出模式 - 显示所有信息"""
        return cls(
            show_expert_process=True,
            show_llm_response=True,
            show_detailed_analysis=True,
            show_aggregation_process=True,
            show_consensus_analysis=True
        )


class EnhancedMultiDimensionalLLMAggregator:
    """增强版多维度专家评分LLM聚合器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "glm-4-flash-250414",
                 cache_rounds: int = 1,
                 min_confidence: float = 0.7,
                 is_lora_mode: bool = False,
                 dimensions: List[str] = None,
                 server_instance=None,
                 verbose: bool = True):  # 添加详细程度控制
        """
        Args:
            verbose: 是否显示详细的专家决策过程
        """
        self.api_key = api_key
        self.model_name = model_name
        self.cache_rounds = cache_rounds
        self.min_confidence = min_confidence
        self.is_lora_mode = is_lora_mode
        self.verbose = verbose  # 控制输出详细程度

        # 实用的专家维度（删除业务价值专家）
        self.dimensions = dimensions or [
            'model_performance',  # 模型性能专家
            'data_quality',  # 数据质量专家
            'spatial_distribution',  # 空间分布专家
            'temporal_stability',  # 时序稳定性专家
            'traffic_pattern'  # 流量模式专家（5个专家）
        ]

        # 动态维度权重（根据训练阶段调整）
        self.dimension_weights = self._initialize_dimension_weights()

        # 缓存和日志
        self.score_cache = {}
        self.logger = logging.getLogger(__name__)
        self.server_instance = server_instance

        # 设置备用聚合器
        self.fallback_aggregator = LoRAFedAvgAggregator() if is_lora_mode else FedAvgAggregator()

        # 初始化LLM客户端
        self.llm_client = None
        self._init_llm_client()

    def _initialize_dimension_weights(self):
        """初始化专家维度权重（5个专家）"""
        return {
            'model_performance': 0.35,  # 模型质量最重要
            'data_quality': 0.25,  # 数据质量次之
            'spatial_distribution': 0.15,  # 地理分布
            'temporal_stability': 0.15,  # 时序稳定性
            'traffic_pattern': 0.10  # 流量模式
        }

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            if self.api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://open.bigmodel.cn/api/paas/v4/"
                )
                self.logger.info("成功初始化增强版多维度LLM客户端")
            else:
                self.logger.warning("未提供API密钥，将使用备用聚合器")
        except Exception as e:
            self.logger.error(f"LLM客户端初始化失败: {e}")

    def aggregate(self, client_models: List[Dict], client_info: List[Dict] = None,
                  client_stats: List = None, round_idx: int = 0):
        """增强版多维度评分聚合"""
        if not self.llm_client or not client_stats:
            print(f"\n⚠️  轮次 {round_idx}: 无LLM客户端或统计数据，使用备用聚合器")
            return self.fallback_aggregator.aggregate(client_models)

        try:
            print(f"\n🚀 启动增强版多维度LLM聚合 - 轮次 {round_idx}")

            # 动态调整维度权重
            old_weights = self.dimension_weights.copy()
            self._adjust_dimension_weights(round_idx)

            # 检查权重是否有变化
            weight_changed = any(abs(old_weights[k] - self.dimension_weights[k]) > 0.001
                                 for k in self.dimension_weights.keys())
            if weight_changed:
                print(f"📊 权重策略已调整 (轮次 {round_idx}):")
                for dim in self.dimension_weights.keys():
                    old_w = old_weights[dim]
                    new_w = self.dimension_weights[dim]
                    change = "↗" if new_w > old_w else "↘" if new_w < old_w else "→"
                    print(f"   • {dim.replace('_', ' ').title()}: {old_w:.1%} {change} {new_w:.1%}")

            # 获取各专家维度评分
            dimension_scores = self._get_enhanced_dimension_scores(client_stats, round_idx)

            # 计算加权总分
            final_scores = self._calculate_weighted_scores(dimension_scores)

            # 转换为聚合权重
            aggregation_weights = self._scores_to_weights(final_scores)

            print(f"\n🎯 最终聚合决策:")
            print(f"   使用增强版多维度LLM智能权重进行聚合")
            print(f"   权重分布: {[f'{w:.3%}' for w in aggregation_weights]}")

            # 执行聚合
            result = self._weighted_aggregate(client_models, aggregation_weights)

            print(f"✅ 聚合完成！")
            return result

        except Exception as e:
            print(f"❌ 增强版多维度聚合失败: {e}")
            print(f"🔄 回退到备用聚合器")
            return self.fallback_aggregator.aggregate(client_models)

    def _get_enhanced_dimension_scores(self, client_stats, round_idx):
        """获取增强版各维度评分"""
        dimension_scores = {}

        if self.verbose:
            print(f"\n{'=' * 80}")
            print(f"🧠 轮次 {round_idx} - 专家评分系统决策过程")
            print(f"{'=' * 80}")

            # 显示当前维度权重
            print(f"📊 当前专家权重分配:")
            for dim, weight in self.dimension_weights.items():
                print(f"   • {dim.replace('_', ' ').title()}: {weight:.1%}")
            print()

        for dimension in self.dimensions:
            try:
                if self.verbose:
                    print(f"🔍 {dimension.replace('_', ' ').title()}专家正在评估...")

                # 检查缓存
                cache_key = f"{dimension}_{round_idx}_{self._get_stats_hash(client_stats)}"
                if cache_key in self.score_cache:
                    dimension_scores[dimension] = self.score_cache[cache_key]
                    if self.verbose:
                        print(f"   ✅ 使用缓存结果: {[f'{s:.2f}' for s in dimension_scores[dimension]]}")
                    continue

                # 调用对应的专家评分函数
                if self.verbose:
                    print(f"   🤖 调用LLM进行专业评分...")
                scores = getattr(self, f'_score_{dimension}')(client_stats, round_idx)
                dimension_scores[dimension] = scores

                # 缓存结果
                self.score_cache[cache_key] = scores

                # 显示评分结果和分析
                if self.verbose:
                    self._print_expert_analysis(dimension, scores, client_stats)

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ {dimension}维度评分失败: {e}")
                    print(f"   🔄 使用默认评分: {[1.0] * len(client_stats)}")
                else:
                    self.logger.warning(f"{dimension}维度评分失败: {e}")
                # 使用默认评分作为备用
                dimension_scores[dimension] = [1.0] * len(client_stats)

        # 显示综合分析
        if self.verbose:
            self._print_comprehensive_analysis(dimension_scores, client_stats, round_idx)

        return dimension_scores

    def _score_model_performance(self, client_stats, round_idx):
        """模型性能专家评分"""
        performance_data = []
        for i, stats in enumerate(client_stats):
            # 结合多个性能指标
            loss = stats.loss
            trend_info = getattr(stats, 'trend_info', {})

            performance_data.append({
                'client': i,
                'loss': f"{loss:.4f}",
                'trend': trend_info.get('description', 'unknown'),
                'improvement_rate': f"{trend_info.get('improvement_rate', 0):.1f}%",
                'participation': getattr(stats, 'participation_count', 1)
            })

        prompt = f"""你是模型性能评估专家。请为{len(client_stats)}个基站的模型训练性能给出评分。

性能数据：
{json.dumps(performance_data, indent=2, ensure_ascii=False)}

评分原则：
1. **损失值权重70%**: 损失越低评分越高，但要警惕过拟合
2. **改进趋势权重20%**: strongly_improving(1.5x), improving(1.2x), stable(1.0x), deteriorating(0.7x)
3. **参与稳定性权重10%**: 参与轮数多的客户端更可靠

特殊考虑：
- 损失异常低(<0.1)可能过拟合，适当惩罚
- 改进率>20%的客户端给予奖励
- 新客户端(参与<3轮)适当保守评分

评分范围：0.3-2.0，直接输出数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_data_quality(self, client_stats, round_idx):
        """数据质量专家评分"""
        quality_data = []
        for i, stats in enumerate(client_stats):
            traffic = stats.traffic_stats
            quality_data.append({
                'client': i,
                'data_points': traffic.get('data_points', 0),
                'cv': f"{traffic.get('coefficient_of_variation', 0):.3f}",
                'iqr': f"{traffic.get('iqr', 0):.1f}",
                'trend_stability': traffic.get('trend', 'unknown'),
                'range_ratio': f"{traffic.get('max', 1) / max(traffic.get('min', 1), 0.1):.1f}"
            })

        prompt = f"""你是数据质量评估专家。请基于真实流量统计为{len(client_stats)}个基站的数据质量评分。

数据质量指标：
{json.dumps(quality_data, indent=2, ensure_ascii=False)}

评分标准：
1. **数据充足性(30%)**: data_points越多越好，<100惩罚，>500奖励
2. **变异系数(40%)**: cv<0.5(优秀1.5x), 0.5-1.0(良好1.0x), >1.0(较差0.7x)
3. **分布合理性(20%)**: iqr适中最好，过大或过小都不好
4. **趋势稳定性(10%)**: stable>increasing>decreasing

特殊处理：
- cv>2.0表示极不稳定，严重惩罚(0.4x)
- range_ratio>100表示数据异常，惩罚
- trend='stable'且cv<0.3给予奖励

评分范围：0.2-1.8，输出数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_spatial_distribution(self, client_stats, round_idx):
        """空间分布专家评分"""
        spatial_data = []
        locations = []

        for i, stats in enumerate(client_stats):
            lng, lat = stats.coordinates['lng'], stats.coordinates['lat']
            locations.append((lng, lat))
            spatial_data.append({
                'client': i,
                'lng': f"{lng:.3f}",
                'lat': f"{lat:.3f}",
                'location_type': self._classify_location_type(lng, lat)
            })

        # 计算空间分散度
        diversity_score = self._calculate_spatial_diversity(locations)

        prompt = f"""你是空间分布专家。请为{len(client_stats)}个基站的地理位置代表性评分。

位置信息：
{json.dumps(spatial_data, indent=2, ensure_ascii=False)}

空间分散度评分: {diversity_score:.3f} (0-1, 越高越分散)

评分原则：
1. **分散性奖励(50%)**: 基于整体空间分散度，越分散越好
2. **边缘价值(30%)**: 位于边缘的基站有独特价值
3. **覆盖均衡(20%)**: 避免某个区域过度集中

特殊考虑：
- 分散度>0.7时所有基站获得奖励
- 位于地理边界的基站额外加分
- 过度聚集的基站群组内评分递减

评分范围：0.6-1.4，输出数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_temporal_stability(self, client_stats, round_idx):
        """时序稳定性专家评分"""
        temporal_data = []
        for i, stats in enumerate(client_stats):
            traffic = stats.traffic_stats
            temporal_data.append({
                'client': i,
                'trend': traffic.get('trend', 'unknown'),
                'trend_slope': f"{traffic.get('trend_slope', 0):.4f}",
                'recent_vs_avg': f"{traffic.get('recent_mean', 0) / max(traffic.get('mean', 1), 0.1):.2f}",
                'cv': f"{traffic.get('coefficient_of_variation', 0):.3f}"
            })

        prompt = f"""你是时序稳定性专家。请评估{len(client_stats)}个基站的时间序列稳定性。

时序特征：
{json.dumps(temporal_data, indent=2, ensure_ascii=False)}

评分标准：
1. **趋势稳定性(40%)**: stable(1.2x), increasing(1.0x), decreasing(0.8x)
2. **斜率合理性(30%)**: |slope|<0.01(好), 0.01-0.05(一般), >0.05(差)
3. **最近期一致性(20%)**: recent_vs_avg在0.8-1.2为好
4. **波动控制(10%)**: cv<0.5为稳定

时序质量评级：
- 优秀: stable + low_cv + 一致性好 → 1.3-1.6分
- 良好: 轻微波动但整体稳定 → 1.0-1.3分  
- 一般: 有明显趋势但可控 → 0.7-1.0分
- 较差: 高波动或异常趋势 → 0.4-0.7分

评分范围：0.4-1.6，输出数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_traffic_pattern(self, client_stats, round_idx):
        """流量模式专家评分（增强版 - 基于真实统计特征）"""
        pattern_data = []
        for i, stats in enumerate(client_stats):
            traffic = stats.traffic_stats

            # 增强的流量模式分析
            mean_traffic = traffic.get('mean', 0)
            cv = traffic.get('coefficient_of_variation', 0)
            trend = traffic.get('trend', 'unknown')
            trend_slope = abs(traffic.get('trend_slope', 0))
            iqr = traffic.get('iqr', 0)

            pattern_data.append({
                'client': i,
                'avg_traffic': f"{mean_traffic:.1f}",
                'traffic_level': self._classify_traffic_level(mean_traffic),
                'stability': 'stable' if cv < 0.5 else 'moderate' if cv < 1.0 else 'unstable',
                'trend_strength': 'weak' if trend_slope < 0.01 else 'moderate' if trend_slope < 0.05 else 'strong',
                'trend_direction': trend,
                'variability': f"{cv:.3f}",
                'data_spread': f"{iqr:.1f}",
                'pattern_quality': self._assess_pattern_quality(traffic)
            })

        prompt = f"""你是流量模式专家。请基于真实流量统计评估{len(client_stats)}个基站的流量模式质量。

详细流量模式分析：
{json.dumps(pattern_data, indent=2, ensure_ascii=False)}

评分标准：
1. **流量稳定性(40%)**: 
   - stable(cv<0.5): 1.3x - 数据可靠，适合训练
   - moderate(0.5≤cv<1.0): 1.0x - 中等质量
   - unstable(cv≥1.0): 0.7x - 不稳定，影响学习

2. **流量水平合理性(25%)**: 
   - 中等流量(50-500): 1.2x - 最佳训练区间
   - 高流量(>500): 1.1x - 重要但可能有噪声
   - 低流量(<50): 0.9x - 信号较弱

3. **趋势特征(20%)**: 
   - weak trend + stable: 1.2x - 理想的平稳模式
   - moderate trend: 1.0x - 正常变化
   - strong trend: 0.8x - 可能存在异常

4. **数据分布质量(15%)**: 
   - 适中的IQR(10-100): 1.1x
   - 过小或过大的IQR: 0.9x

特殊加分项：
- stable + moderate traffic + weak trend: +0.2分
- 数据质量评级为'good'的基站: +0.1分

评分范围：0.4-1.8，输出数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    # 辅助函数
    def _classify_location_type(self, lng, lat):
        """分类位置类型（简化实现）"""
        # 这里可以根据实际地理信息进行更精确的分类
        if abs(lng - 116.3) < 0.1 and abs(lat - 39.9) < 0.1:
            return "core_business"
        elif abs(lng - 116.3) < 0.5 and abs(lat - 39.9) < 0.5:
            return "urban_area"
        else:
            return "suburban"

    def _calculate_spatial_diversity(self, locations):
        """计算空间分散度"""
        if len(locations) < 2:
            return 0.5

        distances = []
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                lng1, lat1 = locations[i]
                lng2, lat2 = locations[j]
                dist = ((lng1 - lng2) ** 2 + (lat1 - lat2) ** 2) ** 0.5
                distances.append(dist)

        avg_distance = np.mean(distances)
        max_distance = max(distances)

        # 归一化到0-1
        diversity = min(1.0, avg_distance / (max_distance + 1e-6))
        return diversity

    def _classify_traffic_level(self, mean_traffic):
        """分类流量水平"""
        if mean_traffic > 300:
            return "high"
        elif mean_traffic > 100:
            return "medium"
        else:
            return "low"

    def _classify_distribution_type(self, traffic_stats):
        """分类流量分布类型"""
        cv = traffic_stats.get('coefficient_of_variation', 0)
        if cv < 0.3:
            return "stable"
        elif cv < 0.8:
            return "moderate_variation"
        else:
            return "high_variation"

    def _assess_pattern_quality(self, traffic_stats):
        """评估流量模式质量（新增）"""
        cv = traffic_stats.get('coefficient_of_variation', 0)
        mean = traffic_stats.get('mean', 0)
        trend = traffic_stats.get('trend', 'unknown')

        # 综合评估模式质量
        quality_score = 0

        # 稳定性评分
        if cv < 0.3:
            quality_score += 3  # 非常稳定
        elif cv < 0.7:
            quality_score += 2  # 较稳定
        else:
            quality_score += 1  # 不稳定

        # 流量水平评分
        if 50 <= mean <= 500:
            quality_score += 2  # 合理范围
        else:
            quality_score += 1  # 偏离理想范围

        # 趋势评分
        if trend == 'stable':
            quality_score += 2
        elif trend in ['increasing', 'decreasing']:
            quality_score += 1

        # 转换为质量等级
        if quality_score >= 6:
            return 'excellent'
        elif quality_score >= 5:
            return 'good'
        elif quality_score >= 3:
            return 'fair'
        else:
            return 'poor'

    # 保持原有的其他方法...
    def _calculate_weighted_scores(self, dimension_scores):
        """计算加权总分"""
        num_clients = len(next(iter(dimension_scores.values())))
        final_scores = [0.0] * num_clients

        total_weight = 0.0
        for dimension, scores in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 1.0 / len(self.dimensions))
            total_weight += weight

            for i in range(num_clients):
                final_scores[i] += scores[i] * weight

        # 归一化
        if total_weight != 1.0:
            final_scores = [score / total_weight for score in final_scores]

        self.logger.info(f"加权总分: {[f'{s:.3f}' for s in final_scores]}")
        return final_scores

    def _scores_to_weights(self, scores):
        """将评分转换为聚合权重"""
        scores_array = np.array(scores)
        temperature = 1.2  # 调整温度参数
        scores_array = scores_array / temperature
        exp_scores = np.exp(scores_array - np.max(scores_array))
        weights = exp_scores / np.sum(exp_scores)

        self.logger.info(f"Softmax权重: {[f'{w:.3f}' for w in weights]}")
        return weights.tolist()

    def _weighted_aggregate(self, client_models: List[Dict], weights: List[float]) -> Dict:
        """执行加权聚合并显示详细过程"""
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        print(f"\n⚙️  执行加权模型聚合:")
        print(f"   参与聚合的客户端数量: {len(client_models)}")
        print(f"   聚合权重: {[f'{w:.3%}' for w in weights]}")

        # 显示权重分布统计
        max_weight = max(weights)
        min_weight = min(weights)
        avg_weight = sum(weights) / len(weights)

        print(f"   权重统计: 最大={max_weight:.3%}, 最小={min_weight:.3%}, 平均={avg_weight:.3%}")

        aggregated_model = {}
        total_params = 0

        # 统计参数信息
        param_info = {}
        for key in client_models[0].keys():
            param_shape = client_models[0][key].shape
            param_count = client_models[0][key].numel()
            param_info[key] = {'shape': param_shape, 'count': param_count}
            total_params += param_count

        print(f"   模型参数统计: 总计 {total_params:,} 个参数")

        if self.is_lora_mode:
            lora_params = sum(
                1 for key in param_info.keys() if any(lora_key in key for lora_key in ['lora_A', 'lora_B']))
            print(f"   LoRA模式: {lora_params} 个LoRA参数模块")

        # 执行聚合
        print(f"   🔄 正在聚合模型参数...")

        for key in client_models[0].keys():
            aggregated_model[key] = torch.zeros_like(client_models[0][key])
            for i, client_model in enumerate(client_models):
                if key in client_model:
                    aggregated_model[key] += client_model[key] * weights[i]

        print(f"   ✅ 参数聚合完成")

        # 显示聚合效果分析
        print(f"   📊 聚合效果分析:")
        print(f"      • 参数更新完成: {len(aggregated_model)} 个模块")

        if self.is_lora_mode:
            print(f"      • LoRA参数总量: {total_params:,}")
            print(f"      • 通信效率: 99%+ (仅传输LoRA参数)")

        # 权重分布分析
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weights)  # 信息熵
        max_entropy = np.log(len(weights))  # 最大熵（均匀分布）
        diversity_ratio = weight_entropy / max_entropy

        print(f"      • 权重多样性: {diversity_ratio:.2%} (100%为完全均匀)")

        if diversity_ratio > 0.9:
            print(f"      • 决策类型: 民主化聚合 (权重分布均匀)")
        elif diversity_ratio > 0.7:
            print(f"      • 决策类型: 平衡聚合 (权重适度集中)")
        else:
            print(f"      • 决策类型: 精英聚合 (权重高度集中)")

        return aggregated_model

    def _parse_scores(self, response, expected_length, default_score=1.0):
        """解析LLM返回的评分"""
        try:
            import re
            array_match = re.search(r'\[([\d\.,\s]+)\]', response)
            if array_match:
                scores_str = array_match.group(1)
                scores = [float(x.strip()) for x in scores_str.split(',')]
                if len(scores) == expected_length:
                    scores = [max(0.1, min(3.0, score)) for score in scores]
                    return scores

            self.logger.warning(f"评分解析失败，使用默认评分: {response[:100]}")
            return [default_score] * expected_length

        except Exception as e:
            self.logger.error(f"评分解析异常: {e}")
            return [default_score] * expected_length

    def _call_llm(self, prompt: str) -> str:
        """调用LLM获取响应"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )

            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content

            return content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise

    def _adjust_dimension_weights(self, round_idx):
        """根据训练进展动态调整维度权重（5个专家）"""
        if round_idx < 5:
            # 初期更重视数据质量和模型性能
            self.dimension_weights = {
                'model_performance': 0.40,
                'data_quality': 0.30,
                'spatial_distribution': 0.15,
                'temporal_stability': 0.10,
                'traffic_pattern': 0.05
            }
        elif round_idx < 15:
            # 中期平衡各维度，加强空间感知
            self.dimension_weights = {
                'model_performance': 0.35,
                'data_quality': 0.25,
                'spatial_distribution': 0.20,
                'temporal_stability': 0.15,
                'traffic_pattern': 0.05
            }
        else:
            # 后期更重视长期稳定性和流量模式
            self.dimension_weights = {
                'model_performance': 0.30,
                'data_quality': 0.20,
                'spatial_distribution': 0.20,
                'temporal_stability': 0.20,
                'traffic_pattern': 0.10
            }

    def _get_stats_hash(self, client_stats):
        """生成客户端统计信息的哈希值用于缓存"""
        key_data = []
        for stats in client_stats:
            key_data.append(f"{stats.client_id}_{stats.loss:.4f}")
        return hashlib.md5('_'.join(key_data).encode()).hexdigest()[:8]

    def get_dimension_analysis_summary(self, client_stats, round_idx):
        """获取维度分析摘要（用于调试和监控）"""
        dimension_scores = self._get_enhanced_dimension_scores(client_stats, round_idx)

        summary = {
            'round': round_idx,
            'dimension_weights': self.dimension_weights.copy(),
            'dimension_scores': {},
            'top_clients': {},
            'insights': []
        }

        # 各维度得分统计
        for dim, scores in dimension_scores.items():
            summary['dimension_scores'][dim] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'scores': [float(s) for s in scores]
            }

            # 找出该维度最高分客户端
            best_idx = np.argmax(scores)
            summary['top_clients'][dim] = {
                'client_id': str(client_stats[best_idx].client_id),
                'score': float(scores[best_idx])
            }

        # 生成洞察
        final_scores = self._calculate_weighted_scores(dimension_scores)
        best_overall_idx = np.argmax(final_scores)
        worst_overall_idx = np.argmin(final_scores)

        summary['insights'] = [
            f"最佳综合表现: 客户端 {client_stats[best_overall_idx].client_id} (得分: {final_scores[best_overall_idx]:.3f})",
            f"需要关注: 客户端 {client_stats[worst_overall_idx].client_id} (得分: {final_scores[worst_overall_idx]:.3f})",
            f"数据质量最高: 客户端 {summary['top_clients']['data_quality']['client_id']}",
            f"空间分布最佳: 客户端 {summary['top_clients']['spatial_distribution']['client_id']}",
            f"模型性能最佳: 客户端 {summary['top_clients']['model_performance']['client_id']}"
        ]

    def _print_expert_analysis(self, dimension: str, scores: List[float], client_stats: List):
        """打印专家分析结果"""
        print(f"   📈 {dimension.replace('_', ' ').title()}专家评分结果:")

        # 显示每个客户端的评分
        for i, (score, stats) in enumerate(zip(scores, client_stats)):
            client_id = stats.client_id
            print(f"      基站 {client_id}: {score:.3f}")

        # 统计分析
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)

        print(
            f"   📊 统计信息: 平均={avg_score:.3f}, 最高={max_score:.3f}, 最低={min_score:.3f}, 标准差={std_score:.3f}")

        # 找出最佳和最差客户端
        best_idx = np.argmax(scores)
        worst_idx = np.argmin(scores)

        print(f"   🏆 最佳: 基站 {client_stats[best_idx].client_id} ({scores[best_idx]:.3f})")
        print(f"   ⚠️  关注: 基站 {client_stats[worst_idx].client_id} ({scores[worst_idx]:.3f})")

        # 维度特定的详细分析
        self._print_dimension_specific_insights(dimension, scores, client_stats)
        print()

    def _print_dimension_specific_insights(self, dimension: str, scores: List[float], client_stats: List):
        """打印维度特定的洞察"""
        if dimension == 'model_performance':
            # 分析损失分布
            losses = [stats.loss for stats in client_stats]
            print(f"   💡 性能洞察: 损失范围 {min(losses):.4f}-{max(losses):.4f}")

        elif dimension == 'data_quality':
            # 分析数据质量分布
            cvs = [stats.traffic_stats.get('coefficient_of_variation', 0) for stats in client_stats]
            stable_count = sum(1 for cv in cvs if cv < 0.5)
            print(f"   💡 质量洞察: {stable_count}/{len(cvs)} 个基站数据稳定 (CV<0.5)")

        elif dimension == 'spatial_distribution':
            # 分析地理分布
            locations = [(stats.coordinates['lng'], stats.coordinates['lat']) for stats in client_stats]
            diversity = self._calculate_spatial_diversity(locations)
            print(f"   💡 空间洞察: 地理分散度 {diversity:.3f} (0-1, 越高越分散)")

        elif dimension == 'temporal_stability':
            # 分析时序稳定性
            trends = [stats.traffic_stats.get('trend', 'unknown') for stats in client_stats]
            stable_count = sum(1 for trend in trends if trend == 'stable')
            print(f"   💡 稳定性洞察: {stable_count}/{len(trends)} 个基站趋势稳定")

        elif dimension == 'traffic_pattern':
            # 分析流量模式
            means = [stats.traffic_stats.get('mean', 0) for stats in client_stats]
            ideal_count = sum(1 for mean in means if 50 <= mean <= 500)
            print(f"   💡 模式洞察: {ideal_count}/{len(means)} 个基站流量在理想范围 (50-500)")

    def _print_comprehensive_analysis(self, dimension_scores: Dict, client_stats: List, round_idx: int):
        """打印综合分析结果"""
        print(f"🎯 综合分析与决策")
        print(f"{'=' * 50}")

        # 计算加权总分
        final_scores = self._calculate_weighted_scores(dimension_scores)
        aggregation_weights = self._scores_to_weights(final_scores)

        print(f"📋 客户端综合评估排名:")
        # 创建排名
        ranked_indices = np.argsort(final_scores)[::-1]  # 从高到低排序

        for rank, idx in enumerate(ranked_indices, 1):
            client_id = client_stats[idx].client_id
            score = final_scores[idx]
            weight = aggregation_weights[idx]

            # 找出该客户端的强项
            strengths = []
            for dim, scores in dimension_scores.items():
                if scores[idx] > np.mean(scores) + 0.1:  # 高于平均值
                    strengths.append(dim.replace('_', ' '))

            strength_str = ', '.join(strengths[:2]) if strengths else '平衡型'

            print(f"   {rank:2d}. 基站 {client_id}: 综合分 {score:.3f} → 聚合权重 {weight:.3%} | 强项: {strength_str}")

        # 决策摘要
        print(f"\n🔍 决策摘要:")
        best_client_idx = ranked_indices[0]
        worst_client_idx = ranked_indices[-1]

        print(
            f"   • 最佳表现: 基站 {client_stats[best_client_idx].client_id} (权重: {aggregation_weights[best_client_idx]:.3%})")
        print(
            f"   • 需要关注: 基站 {client_stats[worst_client_idx].client_id} (权重: {aggregation_weights[worst_client_idx]:.3%})")

        # 权重分布分析
        max_weight = max(aggregation_weights)
        min_weight = min(aggregation_weights)
        weight_ratio = max_weight / min_weight if min_weight > 0 else float('inf')

        print(f"   • 权重集中度: 最高/最低 = {weight_ratio:.1f}x")

        if weight_ratio > 5:
            print(f"   ⚠️  权重分布较为集中，少数客户端主导聚合")
        elif weight_ratio < 2:
            print(f"   ✅ 权重分布较为均匀，民主化聚合")
        else:
            print(f"   📊 权重分布适中，平衡的聚合策略")

        # 专家一致性分析
        self._analyze_expert_consensus(dimension_scores, client_stats)

        print(f"{'=' * 50}")

    def _analyze_expert_consensus(self, dimension_scores: Dict, client_stats: List):
        """分析专家一致性"""
        print(f"\n🤝 专家意见一致性分析:")

        # 计算每个客户端在各维度的排名
        client_rankings = {}
        for i, stats in enumerate(client_stats):
            client_id = str(stats.client_id)
            rankings = []

            for dim, scores in dimension_scores.items():
                # 计算该客户端在此维度的排名（1为最好）
                sorted_indices = np.argsort(scores)[::-1]
                rank = np.where(sorted_indices == i)[0][0] + 1
                rankings.append(rank)

            client_rankings[client_id] = rankings

            # 计算排名方差（一致性指标）
            rank_variance = np.var(rankings)
            consensus_level = "高" if rank_variance < 2 else "中" if rank_variance < 5 else "低"

            print(f"   基站 {client_id}: 排名方差={rank_variance:.1f} → 专家一致性: {consensus_level}")

        # 找出专家意见最一致和最分歧的客户端
        variances = {}
        for client_id, rankings in client_rankings.items():
            variances[client_id] = np.var(rankings)

        most_consistent = min(variances.items(), key=lambda x: x[1])
        most_controversial = max(variances.items(), key=lambda x: x[1])

        print(f"   🎯 专家最认同: 基站 {most_consistent[0]} (方差: {most_consistent[1]:.1f})")
        print(f"   🤔 专家最分歧: 基站 {most_controversial[0]} (方差: {most_controversial[1]:.1f})")