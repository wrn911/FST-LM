# -*- coding: utf-8 -*-
"""
简化版LLM辅助联邦聚合器
"""

import json
import copy
import numpy as np
from typing import List, Dict, Optional
import torch
import logging
from .aggregation import FedAvgAggregator, LoRAFedAvgAggregator


class SimpleLLMAggregator:
    """简化版LLM辅助聚合器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "gemini-2.5-flash",
                 fallback_aggregator=None,
                 cache_rounds: int = 1,  # 改为1，即每轮都调用
                 min_confidence: float = 0.7,
                 is_lora_mode: bool = False):
        """
        Args:
            api_key: Gemini API密钥
            model_name: 模型名称
            fallback_aggregator: 备用聚合器
            cache_rounds: 缓存轮数（设为1表示每轮都调用LLM）
            min_confidence: 最小置信度
            is_lora_mode: 是否为LoRA模式
        """
        self.api_key = api_key
        self.model_name = model_name
        self.cache_rounds = cache_rounds
        self.min_confidence = min_confidence
        self.is_lora_mode = is_lora_mode

        # 设置备用聚合器
        if fallback_aggregator is None:
            self.fallback_aggregator = LoRAFedAvgAggregator() if is_lora_mode else FedAvgAggregator()
        else:
            self.fallback_aggregator = fallback_aggregator

        # 缓存机制
        self.cached_weights = {}
        self.cache_counter = 0

        # 日志
        self.logger = logging.getLogger(__name__)

        # 初始化LLM客户端
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """初始化Gemini客户端"""
        try:
            if self.api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.llm_client = genai.GenerativeModel(self.model_name)
                self.logger.info(f"成功初始化Gemini客户端")
            else:
                self.logger.warning("未提供Gemini API密钥，将使用备用聚合器")
        except ImportError:
            self.logger.error("请安装google-generativeai: pip install google-generativeai")
        except Exception as e:
            self.logger.error(f"Gemini客户端初始化失败: {e}")

    def aggregate(self, client_models: List[Dict], client_info: List[Dict] = None,
                  client_stats: List = None, round_idx: int = 0):
        """
        使用LLM辅助进行模型聚合
        """
        # 决定是否使用LLM
        use_llm = self._should_use_llm(round_idx, client_stats)

        if use_llm and client_stats:
            try:
                # 使用LLM计算权重
                weights = self._get_llm_weights(client_stats, round_idx)
                if weights:
                    self.logger.info(f"轮次 {round_idx}: 使用LLM智能权重进行聚合")
                    return self._weighted_aggregate(client_models, weights)
            except Exception as e:
                self.logger.error(f"LLM聚合失败: {e}")

        # 使用备用聚合器
        self.logger.info(f"轮次 {round_idx}: 使用备用聚合器")

        # 根据聚合器类型传递正确的参数
        if hasattr(self.fallback_aggregator, '__class__') and 'Weighted' in self.fallback_aggregator.__class__.__name__:
            # WeightedFedAvgAggregator需要client_info
            if client_info:
                return self.fallback_aggregator.aggregate(client_models, client_info)
            else:
                # 如果没有client_info，创建简单的权重
                simple_weights = [1.0 / len(client_models)] * len(client_models)
                return self._weighted_aggregate(client_models, simple_weights)
        else:
            # 其他聚合器只需要client_models
            return self.fallback_aggregator.aggregate(client_models)

    def _should_use_llm(self, round_idx: int, client_stats: List) -> bool:
        """判断是否应该使用LLM - 每轮都调用"""
        if not self.llm_client or not client_stats:
            return False

        # 每轮都调用LLM（联邦大模型训练时间长，API调用时间相对较短）
        return True

    def _get_llm_weights(self, client_stats: List, round_idx: int) -> Optional[List[float]]:
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
                return weights

        except Exception as e:
            self.logger.error(f"LLM权重计算失败: {e}")

        return None

    def _construct_prompt(self, client_stats: List, round_idx: int) -> str:
        """构造LLM prompt - 修复JSON序列化问题"""
        # 转换统计信息，确保所有数值都是Python原生类型
        stats_data = []
        for i, stats in enumerate(client_stats):
            stats_dict = {
                'index': int(i),  # 确保是Python int
                'client_id': str(stats.client_id),
                'location': f"({float(stats.coordinates['lng']):.2f}, {float(stats.coordinates['lat']):.2f})",
                'loss': float(stats.loss),
                'avg_traffic': float(stats.traffic_stats.get('mean', 0)),
                'traffic_trend': str(stats.traffic_stats.get('trend', 'stable'))
            }
            stats_data.append(stats_dict)

        # 根据是否为LoRA模式调整prompt
        mode_description = ""
        if self.is_lora_mode:
            mode_description = """
## 技术背景
- 使用LoRA (Low-Rank Adaptation) 进行参数高效微调
- 只有约1%的参数参与训练，通信效率极高
- 每个客户端只传输少量LoRA参数（~3MB vs ~330MB完整模型）"""

        prompt = f"""你是专门用于联邦学习系统的聚合决策专家。请为第{round_idx}轮的{len(client_stats)}个无线基站客户端分配聚合权重。
{mode_description}

## 任务目标
预测各基站的无线流量，提升全局模型的泛化能力。

## 客户端信息
{json.dumps(stats_data, indent=2, ensure_ascii=False)}

## 权重分配原则（按重要性排序）
1. **模型质量优先**: 训练loss越低的客户端，其学习到的模式越可靠
2. **地理分布均衡**: 不同位置的基站提供互补的空间特征
3. **数据代表性**: 平均流量在合理范围内的客户端更具代表性
4. **趋势一致性**: 流量趋势稳定的客户端比剧烈波动的更可靠

## 特殊考虑
- 异常低loss可能表示过拟合，需要适当权衡
- 地理位置过于集中会降低泛化能力
- 流量异常高或异常低的基站权重应适当降低

请输出JSON格式的权重分配（总和严格等于1.0）：
{{
  "weights": [0.15, 0.22, 0.18, ...],
  "reasoning": "详细的决策理由，说明为什么这样分配权重",
  "confidence": 0.85
}}

注意：weights数组长度必须精确等于{len(client_stats)}个客户端。"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """调用Gemini获取响应"""
        try:
            response = self.llm_client.generate_content(prompt)
            return response.text
        except Exception as e:
            self.logger.error(f"Gemini调用失败: {e}")
            raise

    def _parse_weights(self, response: str, expected_length: int) -> Optional[List[float]]:
        """解析LLM响应中的权重"""
        try:
            # 提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                self.logger.warning("LLM响应中未找到JSON格式")
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

            self.logger.info(f"LLM决策: {reasoning[:100]}...")
            self.logger.info(f"置信度: {confidence:.2f}")

            return weights.tolist()

        except Exception as e:
            self.logger.error(f"权重解析失败: {e}")
            return None

    def _weighted_aggregate(self, client_models: List[Dict], weights: List[float]) -> Dict:
        """使用给定权重进行模型聚合 - 支持LoRA模式"""
        if not client_models:
            raise ValueError("客户端模型列表不能为空")

        print(f"  使用LLM权重进行聚合: {[f'{w:.3f}' for w in weights]}")

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


# 客户端统计信息类（简化版）
class ClientStatistics:
    """客户端统计信息"""

    def __init__(self, client_id: str, coordinates: Dict, loss: float, traffic_stats: Dict = None):
        self.client_id = client_id
        self.coordinates = coordinates
        self.loss = loss
        self.traffic_stats = traffic_stats or {}


def create_client_statistics(client_id: str, coordinates: Dict, loss: float,
                             model_params: Dict = None, traffic_data=None) -> ClientStatistics:
    """创建客户端统计信息 - 确保数据类型正确"""

    # 确保所有数值都是Python原生类型
    client_id = str(client_id)
    loss = float(loss)

    # 确保坐标是Python float
    coordinates = {
        'lng': float(coordinates.get('lng', 0)),
        'lat': float(coordinates.get('lat', 0))
    }

    traffic_stats = {}
    if traffic_data is not None:
        import numpy as np
        traffic_stats = {
            'mean': float(np.mean(traffic_data)),
            'std': float(np.std(traffic_data)),
            'trend': 'increasing' if len(traffic_data) > 1 and traffic_data[-1] > traffic_data[0] else 'stable'
        }

    return ClientStatistics(
        client_id=client_id,
        coordinates=coordinates,
        loss=loss,
        traffic_stats=traffic_stats
    )