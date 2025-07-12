#!/bin/bash

# 增强版多维度LLM聚合训练脚本（新增动态融合和约束机制）
echo "=== 启动增强版动态权重联邦学习训练 ==="

cd ../

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
    --eval_every 5 \
    --enable_augmentation \
    --mixup_prob 0.2 \
    --jittering_prob 0.15 \
    --scaling_prob 0.1 \
    --augmentation_ratio 0.5 \
    --similarity_threshold 0.6 \
    --candidate_pool_size 5 \
    --enable_regularization_constraints \
    --max_deviation_ratio 0.25 \
    --min_correlation_threshold 0.5 \
    --constraint_correction_weight 0.3 \
    --device cuda:3

echo "=== 训练完成 ==="