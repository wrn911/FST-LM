#!/bin/bash

# 增强版多维度LLM聚合训练脚本（5个实用专家）
echo "=== 启动增强版多维度LLM联邦学习训练 ==="

python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation enhanced_multi_dim_llm \
    --enhanced_multi_dim_dimensions model_performance,data_quality,spatial_distribution,temporal_stability,traffic_pattern \
    --model_performance_weight 0.35 \
    --data_quality_weight 0.25 \
    --spatial_distribution_weight 0.15 \
    --temporal_stability_weight 0.15 \
    --traffic_pattern_weight 0.10 \
    --enable_dimension_analysis \
    --expert_temperature 1.2 \
    --quality_threshold 0.5 \
    --llm_api_key "118aea86606f4e2f82750c54d3bb380c.DxtxnaKQhFz5EHPY" \
    --llm_model "glm-4-flash-250414" \
    --llm_cache_rounds 1 \
    --llm_min_confidence 0.7 \
    --num_clients 50 \
    --rounds 20 \
    --local_epochs 5 \
    --llm_model_name Qwen3 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --eval_every 5

echo "=== 训练完成 ==="