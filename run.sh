# python main.py --output_dir "final1" --verbose
# python main.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose --resume
# python plotter.py --log_dir "debate_gpt4o_mini_final_run_2"
# python separate_judge_eval.py --output_dir "debate_gpt4o_mini_final_run_2" --verbose
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --resume
# python main.py --output_dir "ld_gpt4o_mini_gpt_judge_llama_8b" --verbose --dataset_name "LD" --evaluator "LD" --judge_model_name gpt-4o-mini --model_name Qwen/Qwen2.5-7B-Instruct
# python main.py --output_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-7B" --verbose --dataset_name "chopped" --evaluator "chopped" --judge_model_name gpt-4o-mini --model_name Qwen/Qwen2.5-7B-Instruct --resume
# python main.py --output_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-1B" --verbose --dataset_name "chopped" --evaluator "chopped"
# python plotter.py --log_dir "ld_gpt4o_mini_gpt_judge_llama_8b"
# python plotter.py --log_dir "chopped_gpt4o_mini_gpt_judge_Qwen2.5-7B"

# CUDA_VISIBLE_DEVICES=0 uv run python main.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" --dataset_name "debate" --evaluator "debate" --output_dir "debate_test" --num_train_iters 50 --eval_iterations 20 --verbose
# CUDA_VISIBLE_DEVICES=2,3 uv run python main.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" --dataset_name "debate" --evaluator "debate" --output_dir "debate_test" --num_train_iters 50 --eval_iterations 20 --verbose --save_steps 5 
CUDA_VISIBLE_DEVICES=2,3 uv run python main.py --model_name "Qwen/Qwen2.5-1.5B-Instruct" --judge_model_name "Qwen/Qwen2.5-1.5B-Instruct" --compare_model_name "Qwen/Qwen2.5-1.5B-Instruct" --dataset_name "debate" --evaluator "debate" --output_dir "debate_reduceMem_test" --num_train_iters 50 --eval_iterations 20 --verbose --save_steps 5 --num_chains 8 --gradient_accumulation_steps 8 

