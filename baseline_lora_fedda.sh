#!/bin/bash

echo "=== 启动完整版FedDA训练 ==="

echo "=============================================== milano: net ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type net\
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
    --save_dir result/baseline_lora_fedda/milano/net

echo "=============================================== milano: call ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type call\
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
    --save_dir result/baseline_lora_fedda/milano/call

echo "=============================================== milano: sms ==============================================="
python federated_train.py \
    --file_path milano.h5\
    --data_type sms\
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
    --save_dir result/baseline_lora_fedda/milano/sms

echo "=============================================== trento: net ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type net\
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
    --save_dir result/baseline_lora_fedda/trento/net

echo "=============================================== trento: call ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type call\
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
    --save_dir result/baseline_lora_fedda/trento/call

echo "=============================================== trento: sms ==============================================="
python federated_train.py \
    --file_path trento.h5\
    --data_type sms\
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
    --save_dir result/baseline_lora_fedda/trento/sms

echo "=== 完整版FedDA训练完成 ==="