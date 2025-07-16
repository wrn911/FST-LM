#!/bin/bash

# FedAvg基线训练脚本（支持LoRA）
echo "=== 启动FedAvg基线训练 ==="

python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedavg \
    --num_clients 50 \
    --rounds 20 \
    --local_epochs 5 \
    --llm_model_name Qwen3 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --eval_every 1 \
    --save_checkpoint \
    --checkpoint_interval 1 \
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
    --device cuda:1 \
    --save_dir result_baseline_lora_fedavg

echo "=== FedAvg基线训练完成 ==="