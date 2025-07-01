# -*- coding: utf-8 -*-
"""
LLM辅助的联邦聚合器
"""

import json
import copy
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
from .aggregation import FedAvgAggregator
import logging


class ClientStatistics:
    """客户端统计信息"""

    def __init__(self, client_id: str, coordinates: Dict,
                 loss: float, grad_norm: float = None,
                 traffic_stats: Dict = None):
        self.client_id = client_id
        self.coordinates = coordinates  # {'lng': x, 'lat': y}
        self.loss = loss
        self.grad_norm = grad_norm
        self.traffic_stats = traffic_stats or {}

    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.client_id,
            'coord': [self.coordinates.get('lng', 0), self.coordinates.get('lat', 0)],
            'loss': round(self.loss, 4),
            'grad_norm': round(self.grad_norm, 4) if self.grad_norm else None,
            'avg_traffic': self.traffic_stats.get('mean', 0),
            'std_traffic': self.traffic_stats.get('std', 0),
            'trend': self.traffic_stats.get('trend', 'stable')
        }


class LLMAggregator:
    """LLM辅助的联邦聚合器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "gemini-pro",
                 fallback_aggregator=None,
                 cache_rounds: int = 5,
                 min_confidence: float = 0.7):
        """
        Args:
            api_key: LLM API密钥
            model_name: 使用的模型名称
            fallback_aggregator: 备用聚合器
            cache_rounds: 缓存轮数，避免每轮都调用LLM
            min_confidence: 最小置信度阈值
        """
        self.api_key = api_key
        self.model_name = model_name
        self.fallback_aggregator = fallback_aggregator or FedAvgAggregator()
        self.cache_rounds = cache_rounds
        self.min_confidence = min_confidence

        # 缓存机制
        self.cached_weights = {}
        self.cache_counter = 0
        self.last_statistics = []

        # 日志
        self.logger = logging.getLogger(__name__)

        # 初始化LLM客户端
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            # 这里可以根据实际使用的LLM服务进行调整
            if self.api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.llm_client = genai.GenerativeModel(self.model_name)
                self.logger.info(f"成功初始化LLM客户端: {self.model_name}")
            else:
                self.llm_client = None
                self.logger.warning("未提供API密钥，将使用备用聚合器")
        except Exception as e:
            self.logger.error(f"LLM客户端初始化失败: {e}")
            self.llm_client = None

    def aggregate(self, client_models: List[Dict], client_info: List[Dict],
                  client_stats: List[ClientStatistics] = None, round_idx: int = 0):
        """
        使用LLM辅助进行模型聚合

        Args:
            client_models: 客户端模型参数
            client_info: 客户端基本信息
            client_stats: 客户端详细统计信息
            round_idx: 当前轮次

        Returns:
            aggregated_model: 聚合后的模型
        """
        # 决定是否使用LLM
        use_llm = self._should_use_llm(round_idx, client_stats)

        if use_llm and client_stats:
            try:
                # 使用LLM计算权重
                weights = self._get_llm_weights(client_stats, round_idx)
                if weights:
                    self.logger.info(f"轮次 {round_idx}: 使用LLM计算的权重进行聚合")
                    return self._weighted_aggregate(client_models, weights)
            except Exception as e:
                self.logger.error(f"LLM聚合失败: {e}")

        # 使用备用聚合器
        self.logger.info(f"轮次 {round_idx}: 使用备用聚合器")
        return self.fallback_aggregator.aggregate(client_models, client_info)

    def _should_use_llm(self, round_idx: int, client_stats: List[ClientStatistics]) -> bool:
        """判断是否应该使用LLM"""
        # 没有LLM客户端
        if not self.llm_client:
            return False

        # 没有统计信息
        if not client_stats:
            return False

        # 缓存机制：每cache_rounds轮使用一次LLM
        if self.cache_counter < self.cache_rounds and self.cached_weights:
            self.cache_counter += 1
            return False

        # 重置缓存计数器
        self.cache_counter = 0
        return True

    def _get_llm_weights(self, client_stats: List[ClientStatistics], round_idx: int) -> Optional[List[float]]:
        """使用LLM计算聚合权重"""
        try:
            # 构造prompt
            prompt = self._construct_prompt(client_stats, round_idx)

            # 调用LLM
            response = self._call_llm(prompt)

            # 解析权重
            weights = self._parse_weights(response, len(client_stats))

            if weights and len(weights) == len(client_stats):
                # 缓存权重
                self.cached_weights = {stats.client_id: w for stats, w in zip(client_stats, weights)}
                self.last_statistics = client_stats
                return weights

        except Exception as e:
            self.logger.error(f"LLM权重计算失败: {e}")

        return None

    def _construct_prompt(self, client_stats: List[ClientStatistics], round_idx: int) -> str:
        """构造LLM prompt"""
        # 转换统计信息为JSON格式
        stats_data = [stats.to_dict() for stats in client_stats]

        prompt = f"""
你是一个联邦学习系统的聚合决策专家。请分析以下基站客户端的统计信息，为每个客户端分配聚合权重。

## 任务背景
- 这是第{round_idx}轮联邦学习
- 目标：预测无线基站流量
- 客户端数量：{len(client_stats)}个

## 客户端统计信息
{json.dumps(stats_data, indent=2, ensure_ascii=False)}

## 权重分配原则
1. **模型质量优先**：loss越低的客户端权重应该更高
2. **空间代表性**：地理位置分散的客户端能提供更好的泛化能力
3. **训练稳定性**：梯度范数适中表示训练稳定
4. **数据质量**：流量统计合理的客户端更可靠

## 输出要求
请按以下JSON格式输出权重分配（权重总和应为1.0）：
{{
  "weights": [0.15, 0.22, 0.18, ...],
  "reasoning": "权重分配的主要考虑因素",
  "confidence": 0.85
}}

注意：
- weights数组长度必须等于客户端数量
- 权重值在0-1之间，总和为1.0
- confidence表示决策置信度(0-1)
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用LLM获取响应"""
        try:
            response = self.llm_client.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise

    def _parse_weights(self, response: str, expected_length: int) -> Optional[List[float]]:
        """解析LLM响应中的权重"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())

            weights = result.get('weights', [])
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')

            # 验证权重
            if len(weights) != expected_length:
                self.logger.warning(f"权重数量不匹配: 期望{expected_length}, 得到{len(weights)}")
                return None

            # 检查置信度
            if confidence < self.min_confidence:
                self.logger.warning(f"LLM置信度过低: {confidence}")
                return None

            # 归一化权重
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                return None

            self.logger.info(f"LLM决策推理: {reasoning}")
            self.logger.info(f"决策置信度: {confidence}")

            return weights.tolist()

        except Exception as e:
            self.logger.error(f"权重解析失败: {e}")
            return None

    def _weighted_aggregate(self, client_models: List[Dict], weights: List[float]) -> Dict:
        """使用给定权重进行模型聚合"""
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        # 初始化聚合模型
        aggregated_model = copy.deepcopy(client_models[0])

        # 加权聚合
        for key in aggregated_model.keys():
            aggregated_model[key] = torch.zeros_like(aggregated_model[key])
            for i, client_model in enumerate(client_models):
                aggregated_model[key] += client_model[key] * weights[i]

        return aggregated_model

    def get_cached_weights(self, client_ids: List[str]) -> Optional[List[float]]:
        """获取缓存的权重"""
        if not self.cached_weights:
            return None

        try:
            weights = [self.cached_weights.get(cid, 0.0) for cid in client_ids]
            if sum(weights) > 0:
                # 重新归一化
                weights = np.array(weights)
                weights = weights / weights.sum()
                return weights.tolist()
        except Exception as e:
            self.logger.error(f"缓存权重获取失败: {e}")

        return None


def create_client_statistics(client_id: str, coordinates: Dict,
                             loss: float, model_params: Dict = None,
                             traffic_data=None) -> ClientStatistics:
    """创建客户端统计信息的便捷函数"""

    # 计算梯度范数（如果提供了模型参数）
    grad_norm = None
    if model_params:
        total_norm = 0
        for param in model_params.values():
            if isinstance(param, torch.Tensor):
                total_norm += param.norm().item() ** 2
        grad_norm = total_norm ** 0.5

    # 计算流量统计（如果提供了流量数据）
    traffic_stats = {}
    if traffic_data is not None:
        traffic_stats = {
            'mean': float(np.mean(traffic_data)),
            'std': float(np.std(traffic_data)),
            'trend': 'increasing' if len(traffic_data) > 1 and traffic_data[-1] > traffic_data[0] else 'stable'
        }

    return ClientStatistics(
        client_id=client_id,
        coordinates=coordinates,
        loss=loss,
        grad_norm=grad_norm,
        traffic_stats=traffic_stats
    )