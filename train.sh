#!/bin/bash

# 增强版多维度LLM聚合训练脚本（新增动态融合和约束机制）
echo "=== 启动增强版动态权重联邦学习训练 ==="

python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation enhanced_multi_dim_llm \
    --enhanced_multi_dim_dimensions model_performance,data_quality,spatial_distribution,temporal_stability,traffic_pattern \
    --alpha_max 0.9 \
    --alpha_min 0.2 \
    --decay_type sigmoid \
    --base_constraint 0.25 \
    --stability_weight 0.4 \
    --quality_weight 0.35 \
    --consistency_weight 0.25 \
    --min_safe_weight 0.05 \
    --llm_api_key "118aea86606f4e2f82750c54d3bb380c.DxtxnaKQhFz5EHPY" \
    --llm_model "glm-4-flash-250414" \
    --num_clients 50 \
    --rounds 20 \
    --local_epochs 5 \
    --llm_model_name Qwen3 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --eval_every 5

echo "=== 训练完成 ==="