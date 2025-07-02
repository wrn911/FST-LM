联邦时空语言模型 (Federated Spatio-Temporal Language Model - FST-LM)

# LLM辅助联邦聚合使用指南

## 功能特点

1. **智能权重分配**：基于客户端位置、模型表现、流量特征的多维决策
2. **成本控制**：缓存机制，每5轮调用一次Gemini API
3. **稳定性保障**：LLM失败时自动回退到LoRA FedAvg
4. **可解释性**：Gemini提供聚合决策的推理过程

## 使用方法

### 1. 获取Gemini API密钥
访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取免费的Gemini API密钥。

### 2. 安装依赖
```bash
pip install google-generativeai
```

### 3. 基础使用
```bash
# 使用环境变量设置API密钥（推荐）
export GEMINI_API_KEY="your_gemini_api_key"

# 运行LLM聚合的联邦学习
python federated_train.py \
    --use_lora \
    --aggregation llm_fedavg \
    --llm_api_key "$GEMINI_API_KEY" \
    --lora_target_modules c_attn,c_proj
```

### 4. 完整配置示例
```bash
python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation llm_fedavg \
    --llm_api_key "your_api_key" \
    --llm_model "gemini-pro" \
    --llm_cache_rounds 5 \
    --llm_min_confidence 0.7 \
    --num_clients 20 \
    --rounds 50 \
    --lora_target_modules c_attn,c_proj
```

## 工作流程

### 1. 客户端训练
- 使用LoRA进行参数高效训练
- 记录训练损失等统计信息
- 只上传LoRA参数（通信效率高）

### 2. LLM智能决策
- 收集客户端位置、损失、流量等信息
- 构造结构化prompt发送给Gemini
- 解析返回的权重分配和推理过程

### 3. 智能聚合
- 使用LLM计算的权重进行加权聚合
- 每5轮调用一次LLM（成本控制）
- 失败时自动回退到标准聚合

## 示例输出

```
轮次 15: 选择了 6 个客户端
  训练客户端 3277: 本地损失 = 0.435
  训练客户端 8327: 本地损失 = 0.521
  ...
聚合 36 个LoRA参数
轮次 15: 使用LLM智能权重进行聚合
LLM决策: 优先考虑loss较低且地理位置分散的客户端...
置信度: 0.89
通信效率统计:
  每个客户端传输大小: 3.24 MB
  通信量减少: 99.0%
```

## 核心优势：LoRA + LLM 完美结合

### 📈 **参数效率**
- **LoRA微调**: 只训练1%的参数（81万 vs 8270万）
- **通信优化**: 每轮只传输3.2MB（vs 330MB完整模型）
- **存储节省**: 服务器端只需更新LoRA参数

### 🧠 **智能聚合**
- **多维决策**: 位置、损失、流量、趋势综合考虑
- **实时调整**: 每轮根据最新客户端表现动态分配权重
- **可解释性**: Gemini提供详细的决策推理过程

### ⚡ **效率平衡**
- **训练时间**: 联邦大模型训练耗时较长（分钟级）
- **API调用**: Gemini响应速度快（秒级）
- **整体收益**: API时间开销相对于训练时间微不足道，但聚合质量显著提升

## 使用方法

### 推荐配置（LoRA + LLM每轮聚合）
```bash
# 设置API密钥
export GEMINI_API_KEY="your_gemini_api_key"

# 运行LoRA + LLM联邦学习
python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation llm_fedavg \
    --llm_api_key "$GEMINI_API_KEY" \
    --llm_cache_rounds 1 \
    --lora_target_modules c_attn,c_proj \
    --num_clients 15 \
    --rounds 20
```

## 典型输出示例

```
=== 联邦学习训练 ===
设备: cuda
聚合算法: llm_fedavg
LoRA配置完成:
  可训练参数: 811,008
  总参数: 82,723,584  
  可训练参数比例: 0.98%

==================================================
联邦学习轮次: 5/20
轮次 5: 选择了 5 个客户端
  训练客户端 3277: 本地损失 = 0.389
  训练客户端 8327: 本地损失 = 0.445
  训练客户端 7575: 本地损失 = 0.321
  训练客户端 3831: 本地损失 = 0.402
  训练客户端 6960: 本地损失 = 0.278

聚合 36 个LoRA参数
轮次 5: 使用LLM智能权重进行聚合
LLM决策: 优先考虑loss最低的客户端6960(0.278)和7575(0.321)，同时平衡地理分布...
置信度: 0.92
  使用LLM权重进行聚合: ['0.156', '0.178', '0.245', '0.189', '0.232']
  聚合的LoRA参数数量: 811,008
通信效率统计:
  每个客户端传输大小: 3.24 MB
  通信量减少: 99.0%
本轮平均客户端损失: 0.367
```

## 技术优势总结

| 方面 | 传统FedAvg | LoRA FedAvg | **LoRA + LLM** |
|------|------------|-------------|-----------------|
| 参数效率 | 100% | 1% | **1%** ✅ |
| 通信成本 | 高 | 低 | **低** ✅ |
| 聚合智能 | 简单平均 | 简单平均 | **智能加权** ✅ |
| 空间感知 | 无 | 无 | **有** ✅ |
| 可解释性 | 无 | 无 | **强** ✅ |
| 适应性 | 静态 | 静态 | **动态** ✅ |

## 注意事项

1. **API密钥安全**: 使用环境变量，不要硬编码
2. **网络连接**: 确保服务器能访问Google服务
3. **置信度设置**: 低于阈值时自动回退
4. **客户端数量**: 过多客户端会增加prompt长度和成本

## 环境变量设置（推荐）

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export GEMINI_API_KEY="your_actual_api_key"

# 然后运行
python federated_train.py \
    --use_lora \
    --aggregation llm_fedavg \
    --llm_api_key "$GEMINI_API_KEY"
```策略
- 异常时自动回退

### 4. 备用方案
- LLM失败时使用FedAvg
- 网络问题时使用缓存权重
- 确保训练连续性

## 成本优化建议

1. **适当的缓存轮数**：建议5-10轮
2. **客户端数量控制**：过多客户端会增加prompt长度
3. **置信度阈值**：设置合理阈值避免低质量决策
4. **API配额管理**：监控API使用量

## 注意事项

1. **API密钥安全**：不要在代码中硬编码API密钥
2. **网络稳定性**：确保服务器网络可访问LLM服务
3. **隐私保护**：只上传统计特征，不包含原始数据
4. **成本控制**：合理设置缓存参数避免过度调用

## 环境配置

```bash
# 安装Google Generative AI库
pip install google-generativeai

# 设置环境变量（推荐）
export GEMINI_API_KEY="your_api_key"
```

## 示例输出

```
轮次 15: 使用LLM计算的权重进行聚合
LLM决策推理: 基于地理分布和模型表现，优先考虑loss较低且位置代表性强的客户端
决策置信度: 0.89
检测到 245 个LoRA参数, 82,478,339 个基础模型参数
```