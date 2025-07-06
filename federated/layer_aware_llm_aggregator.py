# -*- coding: utf-8 -*-
"""
层级感知的LLM辅助联邦聚合器
"""

import json
import copy
import numpy as np
from typing import List, Dict, Optional, Tuple
import torch
import logging
from collections import defaultdict


class LayerAwareLLMAggregator:
    """层级感知的LLM辅助聚合器"""

    def __init__(self,
                 api_key: str = None,
                 model_name: str = "DeepSeek-R1",
                 cache_rounds: int = 1,
                 min_confidence: float = 0.7,
                 is_lora_mode: bool = False,
                 layer_analysis_enabled: bool = True):
        """
        Args:
            layer_analysis_enabled: 是否启用层级分析和差异化权重
        """
        self.api_key = api_key
        self.model_name = model_name
        self.cache_rounds = cache_rounds
        self.min_confidence = min_confidence
        self.is_lora_mode = is_lora_mode
        self.layer_analysis_enabled = layer_analysis_enabled

        self.logger = logging.getLogger(__name__)
        self._init_llm_client()

        # 层级权重缓存
        self.layer_weights_cache = {}

        # 备用聚合器
        if is_lora_mode:
            from .aggregation import LoRAFedAvgAggregator
            self.fallback_aggregator = LoRAFedAvgAggregator()
        else:
            from .aggregation import FedAvgAggregator
            self.fallback_aggregator = FedAvgAggregator()

    def _init_llm_client(self):
        """初始化LLM客户端"""
        try:
            if self.api_key:
                from openai import OpenAI
                self.llm_client = OpenAI(
                    api_key=self.api_key,
                    base_url="http://10.2.8.77:3000/v1"
                )
                self.logger.info("成功初始化LLM客户端")
            else:
                self.logger.warning("未提供API密钥")
                self.llm_client = None
        except Exception as e:
            self.logger.error(f"LLM客户端初始化失败: {e}")
            self.llm_client = None

    def aggregate(self, client_models: List[Dict], client_info: List[Dict] = None,
                  client_stats: List = None, round_idx: int = 0):
        """
        层级感知的模型聚合
        """
        if not self.layer_analysis_enabled or not self.llm_client or not client_stats:
            # 回退到简单聚合
            self.logger.info("回退到简单聚合")
            return self.fallback_aggregator.aggregate(client_models)

        try:
            # 1. 分析模型层次结构
            layer_groups = self._analyze_model_structure(client_models[0])

            # 2. 分析客户端在不同层的表现差异
            layer_performance = self._analyze_layer_performance(client_models, client_stats)

            # 3. 使用LLM计算层级权重
            layer_weights = self._get_llm_layer_weights(
                layer_groups, layer_performance, client_stats, round_idx
            )

            # 4. 执行层级感知聚合
            if layer_weights:
                self.logger.info(f"轮次 {round_idx}: 使用层级感知LLM权重进行聚合")
                return self._layer_aware_aggregate(client_models, layer_weights)
            else:
                self.logger.info(f"轮次 {round_idx}: LLM权重获取失败，使用备用聚合器")
                return self.fallback_aggregator.aggregate(client_models)

        except Exception as e:
            self.logger.error(f"层级感知聚合失败: {e}")
            return self.fallback_aggregator.aggregate(client_models)

    def _analyze_model_structure(self, model_state: Dict) -> Dict[str, List[str]]:
        """
        分析模型的层次结构，将参数按功能分组
        """
        layer_groups = {
            'attention': [],  # 注意力层
            'feedforward': [],  # 前馈网络层
            'embedding': [],  # 嵌入层
            'normalization': [],  # 归一化层
            'output': [],  # 输出层
            'lora': [],  # LoRA参数
            'other': []  # 其他参数
        }

        for param_name in model_state.keys():
            param_name_lower = param_name.lower()

            # LoRA参数识别
            if any(lora_key in param_name for lora_key in ['lora_A', 'lora_B', 'lora_embedding']):
                layer_groups['lora'].append(param_name)
            # 注意力层识别
            elif any(attn_key in param_name_lower for attn_key in
                     ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj']):
                layer_groups['attention'].append(param_name)
            # 前馈网络识别
            elif any(ff_key in param_name_lower for ff_key in
                     ['mlp', 'ffn', 'feed_forward', 'c_fc', 'gate_proj', 'up_proj', 'down_proj']):
                layer_groups['feedforward'].append(param_name)
            # 嵌入层识别
            elif any(emb_key in param_name_lower for emb_key in
                     ['embed', 'embedding', 'wte', 'wpe', 'word_embeddings', 'position_embeddings']):
                layer_groups['embedding'].append(param_name)
            # 归一化层识别
            elif any(norm_key in param_name_lower for norm_key in
                     ['norm', 'ln', 'layer_norm', 'layernorm', 'rms_norm']):
                layer_groups['normalization'].append(param_name)
            # 输出层识别
            elif any(out_key in param_name_lower for out_key in
                     ['output', 'head', 'classifier', 'lm_head']):
                layer_groups['output'].append(param_name)
            else:
                layer_groups['other'].append(param_name)

        # 打印分组结果
        self.logger.info("模型层次结构分析:")
        for group_name, params in layer_groups.items():
            if params:
                self.logger.info(f"  {group_name}: {len(params)} 个参数")

        return layer_groups

    def _analyze_layer_performance(self, client_models: List[Dict],
                                   client_stats: List) -> Dict[str, Dict]:
        """
        分析客户端在不同层的表现差异
        """
        if not client_stats:
            return {}

        layer_performance = {}

        # 计算参数变化统计
        for i, client_model in enumerate(client_models):
            client_id = str(client_stats[i].client_id)
            layer_performance[client_id] = {}

            for param_name, param_tensor in client_model.items():
                if isinstance(param_tensor, torch.Tensor):
                    # 计算参数的统计特征
                    param_stats = {
                        'norm': float(torch.norm(param_tensor).item()),
                        'mean': float(torch.mean(param_tensor).item()),
                        'std': float(torch.std(param_tensor).item()),
                        'sparsity': float((param_tensor == 0).float().mean().item())
                    }
                    layer_performance[client_id][param_name] = param_stats

        return layer_performance

    def _get_llm_layer_weights(self, layer_groups: Dict, layer_performance: Dict,
                               client_stats: List, round_idx: int) -> Optional[Dict]:
        """
        使用LLM计算层级权重
        """
        try:
            # 构造层级感知的prompt
            prompt = self._construct_layer_aware_prompt(
                layer_groups, layer_performance, client_stats, round_idx
            )

            # 调用LLM
            response = self._call_llm(prompt)

            # 解析层级权重
            weights = self._parse_layer_weights(response, len(client_stats), layer_groups)

            return weights

        except Exception as e:
            self.logger.error(f"LLM层级权重计算失败: {e}")
            return None

    def _construct_layer_aware_prompt(self, layer_groups: Dict, layer_performance: Dict,
                                      client_stats: List, round_idx: int) -> str:
        """
        构造层级感知的prompt
        """
        # 准备客户端统计信息
        clients_data = []
        for i, stats in enumerate(client_stats):
            client_data = {
                'index': i,
                'client_id': str(stats.client_id),
                'location': f"({float(stats.coordinates['lng']):.2f}, {float(stats.coordinates['lat']):.2f})",
                'loss': float(stats.loss),
                'traffic_trend': str(stats.traffic_stats.get('trend', 'stable'))
            }
            clients_data.append(client_data)

        # 层级信息
        layer_info = {}
        for group_name, params in layer_groups.items():
            if params:
                layer_info[group_name] = len(params)

        prompt = f"""你是联邦学习系统的层级感知聚合专家。请为第{round_idx}轮的{len(client_stats)}个基站客户端设计层级差异化的聚合权重。

## 模型层次结构
{json.dumps(layer_info, indent=2, ensure_ascii=False)}

## 客户端信息
{json.dumps(clients_data, indent=2, ensure_ascii=False)}

## 层级差异化聚合原理

### 1. 注意力层 (attention)
- **功能**: 捕获时序数据中的长程依赖关系
- **敏感性**: 对数据质量敏感，需要稳定的客户端
- **策略**: 优先考虑loss低且数据稳定的客户端

### 2. 前馈网络层 (feedforward) 
- **功能**: 非线性特征变换和模式识别
- **鲁棒性**: 相对鲁棒，可以容忍一定的数据噪声
- **策略**: 平衡考虑地理分布和模型性能

### 3. 嵌入层 (embedding)
- **功能**: 将输入映射到高维表示空间
- **重要性**: 基础层，影响整个模型的表示能力
- **策略**: 保守聚合，避免过度变化

### 4. 归一化层 (normalization)
- **功能**: 训练稳定性和梯度流动
- **通用性**: 通常具有较好的通用性
- **策略**: 可以采用相对均匀的权重

### 5. LoRA参数 (lora) 
- **功能**: 任务特定的适应
- **灵活性**: 需要快速适应不同客户端的特征
- **策略**: 更多考虑最新的学习效果

## 权重分配任务
请为每个层级组合和客户端分配权重，格式如下：

```json
{{
  "layer_weights": {{
    "attention": [0.15, 0.25, 0.20, 0.18, 0.22],
    "feedforward": [0.18, 0.22, 0.19, 0.21, 0.20],
    "embedding": [0.19, 0.20, 0.20, 0.21, 0.20],
    "normalization": [0.20, 0.20, 0.20, 0.20, 0.20],
    "lora": [0.12, 0.28, 0.22, 0.16, 0.22],
    "other": [0.20, 0.20, 0.20, 0.20, 0.20]
  }},
      "reasoning": {{
    "attention": "注意力层优先考虑loss最低的客户端...",
    "feedforward": "前馈层平衡地理分布...",
    "embedding": "嵌入层采用保守策略...",
    "normalization": "归一化层使用均匀权重...",
    "lora": "LoRA参数重视最新学习效果...",
    "other": "其他参数使用默认策略..."
  }},
  "confidence": 0.85
}}
```

注意：每个层级的权重数组长度必须等于{len(client_stats)}，且每个数组的权重和必须等于1.0。
"""

        return prompt

    def _parse_layer_weights(self, response: str, num_clients: int,
                             layer_groups: Dict) -> Optional[Dict]:
        """
        解析LLM返回的层级权重
        """
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())
            layer_weights = result.get('layer_weights', {})
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', {})

            # 验证权重
            if confidence < self.min_confidence:
                self.logger.warning(f"层级权重置信度过低: {confidence}")
                return None

            # 验证每个层级的权重
            validated_weights = {}
            for group_name in layer_groups.keys():
                if layer_groups[group_name]:  # 只处理非空的层级
                    if group_name in layer_weights:
                        weights = np.array(layer_weights[group_name])
                        if len(weights) != num_clients:
                            self.logger.warning(f"层级{group_name}权重数量不匹配")
                            continue

                        # 归一化权重
                        if weights.sum() > 0:
                            weights = weights / weights.sum()
                            validated_weights[group_name] = weights.tolist()
                    else:
                        # 使用默认均匀权重
                        weights = [1.0 / num_clients] * num_clients
                        validated_weights[group_name] = weights

            self.logger.info(f"层级权重解析成功，置信度: {confidence:.2f}")
            for group_name, reasoning_text in reasoning.items():
                if reasoning_text:
                    self.logger.info(f"  {group_name}: {reasoning_text[:50]}...")

            return validated_weights

        except Exception as e:
            self.logger.error(f"层级权重解析失败: {e}")
            return None

    def _layer_aware_aggregate(self, client_models: List[Dict],
                               layer_weights: Dict) -> Dict:
        """
        执行层级感知的聚合
        """
        layer_groups = self._analyze_model_structure(client_models[0])
        aggregated_model = {}

        self.logger.info("开始层级感知聚合...")

        for group_name, param_names in layer_groups.items():
            if not param_names:
                continue

            # 获取该层级的权重
            if group_name in layer_weights:
                weights = layer_weights[group_name]
                self.logger.info(f"  {group_name}: 使用差异化权重 {[f'{w:.3f}' for w in weights]}")
            else:
                # 使用默认均匀权重
                weights = [1.0 / len(client_models)] * len(client_models)
                self.logger.info(f"  {group_name}: 使用均匀权重")

            # 聚合该层级的参数
            for param_name in param_names:
                aggregated_model[param_name] = torch.zeros_like(client_models[0][param_name])

                for i, client_model in enumerate(client_models):
                    if param_name in client_model:
                        aggregated_model[param_name] += client_model[param_name] * weights[i]

        return aggregated_model

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
                    print(chunk.choices[0].delta.content, end='')
            return content
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            raise