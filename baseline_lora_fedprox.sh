#!/bin/bash

# LoRA版FedProx基线训练脚本
echo "=== 启动LoRA版FedProx基线训练 ==="

echo "=============================================== milano: net ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type net\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/milano/net

echo "=============================================== milano: call ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type call\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/milano/call

echo "=============================================== milano: sms ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type sms\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/milano/sms

echo "=============================================== trento: net ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type net\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/trento/net

echo "=============================================== trento: call ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type call\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/trento/call

echo "=============================================== trento: sms ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type sms\
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --aggregation lora_fedprox \
    --use_fedprox \
    --fedprox_mu 1.0 \
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
    --device cuda:2 \
    --save_dir result/baseline_lora_fedprox/trento/sms

echo "=== LoRA版FedProx基线训练完成 ==="