#!/bin/bash
# Test script for contrastive GRPO training with judge evaluation

echo "Testing contrastive GRPO training with judge evaluation..."

CUDA_VISIBLE_DEVICES=2,3 uv run python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate" \
    --evaluator "debate" \
    --output_dir "debate_contrastive_test" \
    --num_train_iters 500 \
    --eval_iterations 25 \
    --verbose \
    --save_steps 100 \
    --num_chains 4 \
    --gradient_accumulation_steps 4 \
    --contrastive_training True \
    --contrastive_eval False \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-debate-contrastive"

echo "Contrastive GRPO training completed!"
echo "Check output directory: debate_contrastive_test"
echo "Look for '*_contrastive_generations.txt' files in training_logs/"