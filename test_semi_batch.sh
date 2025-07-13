#!/bin/bash
# Test script for semi-batched judge evaluation

echo "Testing semi-batched judge evaluation..."

python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "gsm8k" \
    --evaluator "gsm8k" \
    --output_dir "gsm8k_semi_batch_test" \
    --num_train_iters 1000 \
    --eval_iterations 50 \
    --verbose \
    --save_steps 500 \
    --num_chains 4 \
    --gradient_accumulation_steps 4 \
    --use_semi_batch_judge \
    --use_batch_generation \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-gsm8k" 
