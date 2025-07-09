#!/bin/bash
# Test script for semi-batched judge evaluation

echo "Testing semi-batched judge evaluation..."

CUDA_VISIBLE_DEVICES=2 uv run python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate" \
    --evaluator "debate" \
    --output_dir "debate_semi_batch_test2" \
    --num_train_iters 250 \
    --eval_iterations 25 \
    --verbose \
    --save_steps 25 \
    --num_chains 4 \
    --gradient_accumulation_steps 4 \
    --use_semi_batch_judge \
    --use_batch_generation \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-debate" 
