#!/bin/bash
#SBATCH --gpus-per-node=a100:2 --mem=80G
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=1
# Test script for contrastive GRPO training with semi-batched judge evaluation

echo "Testing contrastive GRPO training with judge evaluation..."

uv run --no-sync python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate_code" \
    --evaluator "debate_code" \
    --output_dir "debate_contrastive_test" \
    --num_train_iters 1000 \
    --eval_iterations 50 \
    --verbose \
    --save_steps 2000 \
    --num_chains 6 \
    --gradient_accumulation_steps 4 \
    --contrastive_training True \
    --contrastive_eval True \
    --truth_comparison \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-debate-contrastive"

echo "Contrastive GRPO training completed!"
echo "Check output directory: debate_contrastive_test"
echo "Look for '*_contrastive_generations.txt' files in training_logs/"