#!/bin/bash

# 多维度LLM聚合训练脚本
python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation multi_dim_llm \
    --multi_dim_dimensions performance,geographic,traffic,trend \
    --performance_weight 0.4 \
    --geographic_weight 0.25 \
    --traffic_weight 0.25 \
    --trend_weight 0.1 \
    --llm_api_key "sk-93nWYhI8SrnXad5m9932CeBdDeDf4233B21d93D217095f22" \
    --llm_model "DeepSeek-R1" \
    --llm_cache_rounds 1 \
    --llm_min_confidence 0.7 \
    --num_clients 10 \
    --rounds 5 \
    --llm_model_name Qwen3 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj