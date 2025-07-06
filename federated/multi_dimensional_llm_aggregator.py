# -*- coding: utf-8 -*-
"""
多维度LLM辅助联邦聚合器
"""

import json
import copy
import numpy as np
from typing import List, Dict, Optional
import torch
import logging
import hashlib
from .aggregation import FedAvgAggregator, LoRAFedAvgAggregator


class MultiDimensionalLLMAggregator:
    """多维度专家评分LLM聚合器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "DeepSeek-R1",
                 cache_rounds: int = 1,
                 min_confidence: float = 0.7,
                 is_lora_mode: bool = False,
                 dimensions: List[str] = None,
                 server_instance=None):  # 添加服务器实例引用
        """
        Args:
            dimensions: 评分维度列表，默认['performance', 'geographic', 'traffic', 'trend']
        """
        self.api_key = api_key
        self.model_name = model_name
        self.cache_rounds = cache_rounds
        self.min_confidence = min_confidence
        self.is_lora_mode = is_lora_mode

        # 设置评分维度
        self.dimensions = dimensions or ['performance', 'geographic', 'traffic', 'trend']

        # 维度权重（可根据训练阶段动态调整）
        self.dimension_weights = {
            'performance': 0.4,  # 性能权重最高
            'geographic': 0.25,  # 地理分布权重
            'traffic': 0.25,  # 流量特征权重
            'trend': 0.1  # 趋势权重
        }

        # 缓存和日志
        self.score_cache = {}
        self.logger = logging.getLogger(__name__)

        # 服务器实例引用（用于访问历史数据）
        self.server_instance = server_instance

        # 设置备用聚合器
        self.fallback_aggregator = LoRAFedAvgAggregator() if is_lora_mode else FedAvgAggregator()

        # 初始化LLM客户端
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            if self.api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=self.api_key,
                    base_url="http://10.2.8.77:3000/v1"
                )
                self.logger.info("成功初始化多维度LLM客户端")
            else:
                self.logger.warning("未提供API密钥，将使用备用聚合器")
        except Exception as e:
            self.logger.error(f"LLM客户端初始化失败: {e}")

    def aggregate(self, client_models: List[Dict], client_info: List[Dict] = None,
                  client_stats: List = None, round_idx: int = 0):
        """多维度评分聚合"""
        if not self.llm_client or not client_stats:
            self.logger.info(f"轮次 {round_idx}: 使用备用聚合器")
            return self.fallback_aggregator.aggregate(client_models)

        try:
            # 动态调整维度权重
            self._adjust_dimension_weights(round_idx)

            # 获取各维度评分
            dimension_scores = self._get_dimension_scores(client_stats, round_idx)

            # 计算加权总分
            final_scores = self._calculate_weighted_scores(dimension_scores)

            # 转换为聚合权重
            aggregation_weights = self._scores_to_weights(final_scores)

            self.logger.info(f"轮次 {round_idx}: 使用多维度LLM智能权重进行聚合")

            # 执行聚合
            return self._weighted_aggregate(client_models, aggregation_weights)

        except Exception as e:
            self.logger.error(f"多维度聚合失败: {e}")
            return self.fallback_aggregator.aggregate(client_models)

    def _get_dimension_scores(self, client_stats, round_idx):
        """获取各维度评分"""
        dimension_scores = {}

        for dimension in self.dimensions:
            try:
                # 检查缓存
                cache_key = f"{dimension}_{round_idx}_{self._get_stats_hash(client_stats)}"
                if cache_key in self.score_cache:
                    dimension_scores[dimension] = self.score_cache[cache_key]
                    self.logger.info(f"{dimension}维度评分(缓存): {[f'{s:.2f}' for s in dimension_scores[dimension]]}")
                    continue

                # 调用对应的评分函数
                scores = getattr(self, f'_score_{dimension}')(client_stats, round_idx)
                dimension_scores[dimension] = scores

                # 缓存结果
                self.score_cache[cache_key] = scores

                self.logger.info(f"{dimension}维度评分: {[f'{s:.2f}' for s in scores]}")

            except Exception as e:
                self.logger.warning(f"{dimension}维度评分失败: {e}")
                # 使用均匀评分作为备用
                dimension_scores[dimension] = [1.0] * len(client_stats)

        return dimension_scores

    def _score_performance(self, client_stats, round_idx):
        """性能维度评分"""
        # 准备简洁的性能数据
        performance_data = []
        for i, stats in enumerate(client_stats):
            performance_data.append(f"基站{i}: 损失={stats.loss:.4f}")

        prompt = f"""你是联邦学习性能评估专家。请根据{len(client_stats)}个基站的训练损失给出性能评分。该评分会用于联邦聚合时的参考。

性能数据：
{chr(10).join(performance_data)}

评分原则：
- 损失越低，评分越高
- 评分范围：0.1-2.0
- 避免极端评分，保持相对平衡

请直接给出评分数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_geographic(self, client_stats, round_idx):
        """地理维度评分"""
        # 准备地理位置数据
        geo_data = []
        for i, stats in enumerate(client_stats):
            lng, lat = stats.coordinates['lng'], stats.coordinates['lat']
            geo_data.append(f"基站{i}: ({lng:.1f}, {lat:.1f})")

        prompt = f"""你是地理分布专家。请根据{len(client_stats)}个基站的位置给出地理多样性评分。该评分会用于联邦聚合时的参考。

位置数据：
{chr(10).join(geo_data)}

评分原则：
- 位置分散的基站评分高
- 位置聚集的基站评分略低
- 评分范围：0.5-1.5
- 促进地理多样性

请直接给出评分数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_traffic(self, client_stats, round_idx):
        """流量特征维度评分"""
        # 准备流量统计数据
        traffic_data = []
        for i, stats in enumerate(client_stats):
            traffic_info = stats.traffic_stats
            mean_traffic = traffic_info.get('mean', 0)
            traffic_trend = traffic_info.get('trend', 'stable')
            traffic_data.append(f"基站{i}: 均值={mean_traffic:.0f}, 趋势={traffic_trend}")

        prompt = f"""你是流量模式专家。请根据{len(client_stats)}个基站的流量特征给出代表性评分。该评分会用于联邦聚合时的参考。

流量数据：
{chr(10).join(traffic_data)}

评分原则：
- 流量稳定且在合理范围内的基站评分高
- 流量异常（过高/过低）的基站评分低
- 评分范围：0.3-1.8
- 优先代表性强的基站

请直接给出评分数组：[分数1, 分数2, ...]

评分："""

        response = self._call_llm(prompt)
        return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _score_trend(self, client_stats, round_idx):
        """趋势维度评分 - 完整实现"""
        # 准备趋势数据
        trend_data = []
        trend_scores = []

        for i, stats in enumerate(client_stats):
            client_id = str(stats.client_id)

            # 从服务器获取完整的趋势分析
            if self.server_instance:
                trend_summary = self.server_instance.get_client_trend_summary(client_id)
                trend_description = trend_summary['description']
                base_score = trend_summary['score']

                # 添加详细信息到数据中
                details = trend_summary.get('details', {})
                participation = details.get('participation_count', 0)

                trend_data.append(f"基站{i}: 趋势={trend_description}, 参与={participation}轮, 评分={base_score:.2f}")
                trend_scores.append(base_score)
            else:
                # 回退到简化计算
                trend = self._calculate_loss_trend_simple(stats, round_idx)
                trend_data.append(f"基站{i}: 趋势={trend}")

                # 简单评分映射
                if trend == "improving":
                    trend_scores.append(1.3)
                elif trend == "deteriorating":
                    trend_scores.append(0.7)
                else:
                    trend_scores.append(1.0)

        # 如果有服务器实例，使用更智能的LLM评分
        if self.server_instance and any(score != 1.0 for score in trend_scores):
            prompt = f"""你是学习趋势专家。请根据{len(client_stats)}个基站的详细趋势分析给出最终评分。该评分会用于联邦聚合时的参考。

趋势分析：
{chr(10).join(trend_data)}

评分原则：
- strongly_improving: 1.4-1.6分
- improving: 1.2-1.4分
- recently_improving: 1.1-1.3分
- stable_good: 1.0-1.2分
- stable: 0.9-1.1分
- unstable: 0.7-0.9分
- deteriorating: 0.4-0.7分
- 参与轮数多的客户端评分可适当上调

请直接给出评分数组：[分数1, 分数2, ...]

评分："""

            response = self._call_llm(prompt)
            llm_scores = self._parse_scores(response, len(client_stats), default_score=1.0)

            # 结合计算得分和LLM得分
            final_scores = []
            for i in range(len(client_stats)):
                # 加权平均：70%计算得分 + 30%LLM得分
                combined_score = 0.7 * trend_scores[i] + 0.3 * llm_scores[i]
                final_scores.append(combined_score)

            return final_scores
        else:
            # 使用简化的LLM prompt
            prompt = f"""你是学习趋势专家。请根据{len(client_stats)}个基站的学习趋势给出评分。

趋势数据：
{chr(10).join(trend_data)}

评分原则：
- 学习趋势向好的基站评分高
- 学习停滞的基站评分低
- 评分范围：0.4-1.6
- 奖励持续改进的基站

请直接给出评分数组：[分数1, 分数2, ...]

评分："""

            response = self._call_llm(prompt)
            return self._parse_scores(response, len(client_stats), default_score=1.0)

    def _calculate_loss_trend_simple(self, stats, round_idx):
        """简化的损失趋势计算（备用方法）"""
        # 这里是简化实现，当没有服务器实例时使用
        if hasattr(stats, 'loss_history') and len(stats.loss_history) > 1:
            recent_trend = stats.loss_history[-1] - stats.loss_history[-2]
            if recent_trend < -0.01:
                return "improving"
            elif recent_trend > 0.01:
                return "deteriorating"
            else:
                return "stable"
        else:
            return "stable"

    def _calculate_weighted_scores(self, dimension_scores):
        """计算加权总分"""
        num_clients = len(next(iter(dimension_scores.values())))
        final_scores = [0.0] * num_clients

        total_weight = 0.0
        for dimension, scores in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 0.25)
            total_weight += weight

            for i in range(num_clients):
                final_scores[i] += scores[i] * weight

        # 归一化（如果权重总和不为1）
        if total_weight != 1.0:
            final_scores = [score / total_weight for score in final_scores]

        self.logger.info(f"加权总分: {[f'{s:.3f}' for s in final_scores]}")
        return final_scores

    def _scores_to_weights(self, scores):
        """将评分转换为聚合权重（使用softmax）"""
        # 应用softmax转换
        scores_array = np.array(scores)

        # 调整温度参数来控制权重分布的平滑程度
        temperature = 1.5  # 温度越高，权重分布越平均
        scores_array = scores_array / temperature

        # Softmax计算
        exp_scores = np.exp(scores_array - np.max(scores_array))  # 数值稳定性
        weights = exp_scores / np.sum(exp_scores)

        self.logger.info(f"Softmax权重: {[f'{w:.3f}' for w in weights]}")
        return weights.tolist()

    def _weighted_aggregate(self, client_models: List[Dict], weights: List[float]) -> Dict:
        """使用给定权重进行模型聚合"""
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        print(f"  使用多维度LLM权重进行聚合: {[f'{w:.3f}' for w in weights]}")

        # 初始化聚合结果
        aggregated_model = {}

        # 对每个参数进行加权聚合
        for key in client_models[0].keys():
            aggregated_model[key] = torch.zeros_like(client_models[0][key])
            for i, client_model in enumerate(client_models):
                if key in client_model:
                    aggregated_model[key] += client_model[key] * weights[i]

        # 如果是LoRA模式，添加相关统计信息
        if self.is_lora_mode:
            total_params = sum(param.numel() for param in aggregated_model.values())
            print(f"  聚合的LoRA参数数量: {total_params:,}")

        return aggregated_model

    def _parse_scores(self, response, expected_length, default_score=1.0):
        """解析LLM返回的评分"""
        try:
            import re

            # 尝试提取数组格式的评分
            array_match = re.search(r'\[([\d\.,\s]+)\]', response)
            if array_match:
                scores_str = array_match.group(1)
                scores = [float(x.strip()) for x in scores_str.split(',')]

                if len(scores) == expected_length:
                    # 确保评分在合理范围内
                    scores = [max(0.1, min(2.0, score)) for score in scores]
                    return scores

            # 如果解析失败，返回默认评分
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

            # 收集流式响应
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content

            return content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise

    def _adjust_dimension_weights(self, round_idx):
        """根据训练进展动态调整维度权重"""
        if round_idx < 5:
            # 初期更重视性能
            self.dimension_weights = {
                'performance': 0.5,
                'geographic': 0.3,
                'traffic': 0.15,
                'trend': 0.05
            }
        elif round_idx < 15:
            # 中期平衡各维度
            self.dimension_weights = {
                'performance': 0.4,
                'geographic': 0.25,
                'traffic': 0.25,
                'trend': 0.1
            }
        else:
            # 后期更重视趋势和稳定性
            self.dimension_weights = {
                'performance': 0.35,
                'geographic': 0.2,
                'traffic': 0.25,
                'trend': 0.2
            }

    def _calculate_loss_trend(self, stats, round_idx):
        """计算损失趋势（完整实现）"""
        client_id = str(stats.client_id)

        # 如果有服务器实例，使用完整的趋势分析
        if self.server_instance:
            trend_summary = self.server_instance.get_client_trend_summary(client_id)
            return trend_summary['description']
        else:
            # 回退到简化实现
            return self._calculate_loss_trend_simple(stats, round_idx)

    def _get_stats_hash(self, client_stats):
        """生成客户端统计信息的哈希值用于缓存"""
        key_data = []
        for stats in client_stats:
            key_data.append(f"{stats.client_id}_{stats.loss:.4f}")

        return hashlib.md5('_'.join(key_data).encode()).hexdigest()[:8]