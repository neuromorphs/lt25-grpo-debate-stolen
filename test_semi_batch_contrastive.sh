#!/bin/bash
# Test script for contrastive GRPO training with semi-batched judge evaluation

echo "Testing contrastive GRPO training with semi-batched judge evaluation..."
echo "This will train using PRO vs CON cross-comparison as described in Multi-Prompt GRPO"

CUDA_VISIBLE_DEVICES=2,3 uv run python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate" \
    --evaluator "debate" \
    --output_dir "debate_contrastive_test" \
    --num_train_iters 250 \
    --eval_iterations 25 \
    --verbose \
    --save_steps 100 \
    --num_chains 8 \
    --gradient_accumulation_steps 4 \
    --use_semi_batch_judge \
    --use_batch_generation \
    --use_contrastive \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-debate-contrastive"

echo "Contrastive GRPO training completed!"
echo "Check output directory: debate_contrastive_test"
echo "Look for '*_contrastive_generations.txt' files in training_logs/"