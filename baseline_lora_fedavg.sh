#!/bin/bash

# FedAvg基线训练脚本（支持LoRA）
echo "=== 启动FedAvg基线训练 ==="

echo "=============================================== milano: net ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type net\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/milano/net

echo "=============================================== milano: call ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type call\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/milano/call

echo "=============================================== milano: sms ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type sms\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/milano/sms

echo "=============================================== trento: net ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type net\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/trento/net

echo "=============================================== trento: call ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type call\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/trento/call

echo "=============================================== trento: sms ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type sms\
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
    --device cuda:1 \
    --save_dir result/baseline_lora_fedavg/trento/sms

echo "=== FedAvg基线训练完成 ==="