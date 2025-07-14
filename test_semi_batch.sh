#!/bin/bash
#SBATCH --gpus-per-node=a100:2 --mem=80G
#SBATCH --time=02:59:00
#SBATCH --cpus-per-task=1


module load CUDA/12.4.0
module load  GCC/10.2.0
module load Python/3.13.1-GCCcore-14.2.0


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
