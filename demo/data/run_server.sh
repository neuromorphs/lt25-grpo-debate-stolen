#!/bin/bash
#SBATCH -w isl-gpu53
#SBATCH --job-name=sc35
#SBATCH --output=%x_%j.out
#SBATCH --partition=g80
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14

set -x

NO_PROXY=127.0.0.1 nohup python inference_server.py > gradio.log 2>&1 &

sleep 10

vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --task generate \
  --model-impl transformers \
  --host 127.0.0.1 --port 1210
