#!/bin/bash
uv run --no-sync python main.py \
    --output_dir "output_pc" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "debate" \
    --evaluator "debate" \
    --num_train_iters 165 \
    --eval_iterations 20 \
    --gradient_accumulation_steps 2 \
    --num_chains 3 \
    --contrastive_training True \
    --contrastive_eval True \
    --enable_wandb \
    --wandb_project "grpo-debate-new" \
    --debug \