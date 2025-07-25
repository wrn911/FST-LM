#!/bin/bash

echo "=== 启动完整版FedDA训练 ==="

python federated_train.py \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation fedda \
    --fedda_clusters 3 \
    --fedda_rho 0.1 \
    --fedda_gamma 0.01 \
    --fedda_enable_augmentation \
    --fedda_augment_ratio 0.01 \
    --num_clients 50 \
    --rounds 20 \
    --local_epochs 5 \
    --llm_model_name Qwen3 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --eval_every 1 \
    --save_checkpoint \
    --checkpoint_interval 5 \
    --save_best_model \
    --device cuda:3 \
    --save_dir result_baseline_lora_fedda

echo "=== 完整版FedDA训练完成 ==="