#!/bin/bash
#SBATCH --gpus-per-node=a100:2 --mem=80G
#SBATCH --time=05:59:00
#SBATCH --cpus-per-task=1
# Test script for contrastive GRPO training with semi-batched judge evaluation


echo "Testing contrastive GRPO training with semi-batched judge evaluation..."
echo "This will train using PRO vs CON cross-comparison as described in Multi-Prompt GRPO"

uv run --no-sync python main.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate_code" \
    --evaluator "debate_code" \
    --output_dir "debate_contrastive_code" \
    --num_train_iters 1000 \
    --eval_iterations 25 \
    --verbose \
    --save_steps -1 \
    --num_chains 6 \
    --gradient_accumulation_steps 4 \
    --use_semi_batch_judge \
    --use_batch_generation \
    --use_contrastive \
    --enable_detailed_logging \
    --enable_wandb \
    --wandb_project "grpo-debate-contrastive" \
    --eval_type "pc" \
    --truth_optim \

echo "Contrastive GRPO training completed!"
echo "Check output directory: debate_contrastive_test"
echo "Look for '*_contrastive_generations.txt' files in training_logs/"