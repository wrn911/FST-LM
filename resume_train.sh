#!/bin/bash

# 从检查点恢复训练脚本
echo "=== 从检查点恢复联邦学习训练 ==="

# 检查点文件路径（请根据实际情况修改）
CHECKPOINT_PATH="results/checkpoint_round_3.pth"

# 检查文件是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT_PATH"
    echo "请检查文件路径或运行正常训练"
    exit 1
fi

echo "从检查点恢复: $CHECKPOINT_PATH"

python federated_train.py \
    --resume "$CHECKPOINT_PATH" \
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
    --eval_every 1 \
    --save_checkpoint \
    --checkpoint_interval 5 \
    --save_best_model \
    --enable_augmentation \
    --mixup_prob 0.2 \
    --jittering_prob 0.15 \
    --scaling_prob 0.1 \
    --augmentation_ratio 0.3 \
    --similarity_threshold 0.6 \
    --candidate_pool_size 5 \
    --enable_regularization_constraints \
    --max_deviation_ratio 0.25 \
    --min_correlation_threshold 0.5 \
    --constraint_correction_weight 0.3 \
    --device cuda:0

echo "=== 恢复训练完成 ==="