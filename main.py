"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface

import llms
import utils
import evaluator
import rldatasets

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable logging.")

def eval_on_test_set(
    all_models: dict,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set by comparing each model completion
    against a base model completion and having them judged.
    """
    print("Running evaluation on test set...")
    logger = logging.getLogger(__name__) if hasattr(args, 'enable_detailed_logging') and args.enable_detailed_logging else None
    
    total_scores = defaultdict(float)
    num_examples = 0
    total_wins = 0

    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        total_comparisons = 0
        total_wins = 0
        
        for question in tqdm(test_loader, desc="Evaluating on test set"):
            num_examples += 1

            # 1. Prepare prompting
            prompt = [
                {'role': 'system', 'content': test_loader.pre_prompt},
                {'role': 'user', 'content': question}
            ]
            prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)

            # Log Initial prompt 
            f.write("\n" + "="*80 + "\n")
            f.write(f"Example #{num_examples}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Prompt:\n")
            f.write(f"{prompt_text}\n\n")

            # Generate completions from trained model
            _, _, _, _, completions_text, _ = generate_completions(
                all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
            )

            # Generate completions for compare model using batched interface
            if args.use_batch_generation and hasattr(all_models["compare_model"], 'generate_batch'):
                # Use efficient batched generation for HuggingFace models (e.g., Qwen)
                compare_completions_text = all_models["compare_model"].generate_batch(
                    system_prompt=test_loader.pre_prompt,
                    user_prompt=question,
                    num_completions=args.num_chains,
                    max_new_tokens=args.max_completion_length,
                    temperature=args.temperature
                )
            else:
                # Fallback to sequential generation for API models or when batching is disabled
                compare_completions_text = []
                for _ in range(args.num_chains):
                    completion = all_models["compare_model"].generate(
                        system_prompt=test_loader.pre_prompt,
                        user_prompt=question,
                        max_new_tokens=args.max_completion_length,
                        temperature=args.temperature
                    )
                    compare_completions_text.append(completion)

            # Score completions to get reward metrics
            rewards_per_func, reward_metrics = eval_class.compute_rewards(
                input_prompt=question, 
                all_models=all_models, 
                train_model_completions=completions_text, 
                compare_model_completions=compare_completions_text,
                device=device,
                is_test=True,
                use_batched_eval=args.use_batch_judge,
                use_semi_batched_eval=args.use_semi_batch_judge
            )

            # Track total comparisons and wins
            comparisons_this_question = len(completions_text)
            total_comparisons += comparisons_this_question
            total_wins += reward_metrics['num_wins']

            # For each completion pair, log the results
            for i, (completion, compare_completion) in enumerate(zip(completions_text, compare_completions_text)):
                f.write(f"\nCompletion #{i+1}:\n")
                f.write("-"*40 + "\n\n")

                # Log trained model's response
                f.write("TRAINED MODEL RESPONSE:\n")
                f.write(f"Full response:\n{completion}\n\n")
                
                try:
                    trained_reasoning = completion.split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    trained_answer = completion.split("<answer>\n")[1].split("\n</answer>")[0]
                except:
                    trained_reasoning = "ERROR: Could not parse reasoning"
                    trained_answer = "ERROR: Could not parse answer"
                
                f.write(f"Parsed reasoning:\n{trained_reasoning}\n")
                f.write(f"Parsed answer:\n{trained_answer}\n\n")

                # Log compare model's response
                f.write("COMPARE MODEL RESPONSE:\n")
                f.write(f"Full response:\n{compare_completion}\n\n")
                
                try:
                    compare_reasoning = compare_completion.split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    compare_answer = compare_completion.split("<answer>\n")[1].split("\n</answer>")[0]
                except:
                    compare_reasoning = "ERROR: Could not parse reasoning"
                    compare_answer = "ERROR: Could not parse answer"
                
                f.write(f"Parsed reasoning:\n{compare_reasoning}\n")
                f.write(f"Parsed answer:\n{compare_answer}\n\n")

                # Log reward scores for this completion
                f.write("REWARD SCORES:\n")
                reward_breakdown = eval_class.get_reward_breakdown(rewards_per_func[i])
                for reward_name, reward_value in reward_breakdown.items():
                    f.write(f"{reward_name}: {reward_value:.4f}\n")
                f.write(f"Total reward: {rewards_per_func[i].sum().item():.4f}\n")

                # Log if trained model won this comparison
                trained_model_won = rewards_per_func[i,0] > 0
                f.write(f"\nOUTCOME: Trained model {'won' if trained_model_won else 'lost'} this comparison\n")
                f.write("-"*40 + "\n")

            # Log summary metrics for this question
            f.write("\nSUMMARY METRICS:\n")
            f.write(f"Win rate: {reward_metrics['win_rate']:.2%}\n")
            f.write(f"Number of wins: {reward_metrics['num_wins']}\n")
            f.write(f"Total comparisons: {reward_metrics.get('num_comparisons', reward_metrics.get('num_debates', 0))}\n")
            f.write(f"Average format scores:\n")
            f.write(f"  Strict format: {reward_metrics['rewards/strict_format']:.4f}\n")
            f.write(f"  Soft format: {reward_metrics['rewards/soft_format']:.4f}\n")
            f.write(f"  XML count: {reward_metrics['rewards/xml_count']:.4f}\n")

            # Update total scores
            for k, v in reward_metrics.items():
                if k.startswith('rewards/'):
                    total_scores[k] += v
        
        # Calculate final metrics
        win_rate = (total_wins / total_comparisons) * 100 if total_comparisons > 0 else 0
        avg_scores = {k: v/num_examples for k,v in total_scores.items()}

        # Save metrics
        metrics = {
            'win_rate': win_rate,
            'total_wins': total_wins,
            'total_comparisons': total_comparisons,
            'num_examples': num_examples,
            'average_scores': avg_scores
        }

        # Write summary results to file and optionally print
        f.write("\nFINAL EVALUATION RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Win Rate: {win_rate:.2f}%\n")
        f.write(f"Total Wins: {total_wins}\n") 
        f.write(f"Total Comparisons: {total_comparisons}\n")
        f.write("\nAverage Scores:\n")
        for metric, value in avg_scores.items():
            f.write(f"{metric:15s}: {value:.4f}\n")
        f.write("-" * 20 + "\n")

        if args.verbose:
            print("\nEvaluation Results:")
            print("-" * 20)
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Wins: {total_wins}")
            print(f"Total Comparisons: {total_comparisons}")
            print("\nAverage Scores:")
            for metric, value in avg_scores.items():
                print(f"{metric:15s}: {value:.4f}")
            print("-" * 20)
            
        # Also log to file
        if logger:
            logger.info(f"Test evaluation completed - Win Rate: {win_rate:.2f}%, Total Wins: {total_wins}, Total Comparisons: {total_comparisons}")
            for metric, value in avg_scores.items():
                logger.info(f"Test {metric}: {value:.4f}")

    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics, win_rate

def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    prompt_text: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        prompt_text: The input question/prompt to generate completions for - should be full prompt ready to be turned into token ids (i.e. chat template applied etc)
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """

    # Tokenize
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:]
    
    # Repeat for number of chains/generations
    prompt_ids = prompt_ids.repeat(args.num_chains, 1)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1)

    # Move tensors to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Set up generation config
    generation_config = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id
    )

    # Generate completions
    prompt_completion_ids = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        generation_config=generation_config
    )

    # Extract completion ids
    prompt_length = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # Do masking 
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text
    
def score_completions(
    completions_text: list[str],
    question: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=completions_text, 
        compare_model_completions=None,
        device=device, 
        is_test=False,
        use_batched_eval=args.use_batch_judge,
        use_semi_batched_eval=args.use_semi_batch_judge
    )
    rewards = rewards_per_func.sum(dim=1)


    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data

def score_contrastive_completions(
    pro_completions_text: list[str],
    con_completions_text: list[str],
    question: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
        },
        'pro_generations': [],
        'con_generations': []
    }

    # Format inputs as expected by evaluator
    # Get rewards and metrics from evaluator
    pro_first_rewards_per_func, pro_first_metrics, con_second_debate_score = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=pro_completions_text,
        compare_model_completions=con_completions_text, 
        device=device, 
        is_test=False,
        use_batched_eval=args.use_batch_judge,
        use_semi_batched_eval=args.use_semi_batch_judge
    )
    con_first_rewards_per_func, con_first_metrics, first_second_debate_score = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=con_completions_text, # TODO: the argument name from the function is confusing
        compare_model_completions=pro_completions_text, # TODO: the argument name from the function is confusing
        device=device, 
        is_test=False,
        use_batched_eval=args.use_batch_judge,
        use_semi_batched_eval=args.use_semi_batch_judge
    )
    pro_first_rewards_per_func[:, 0] += first_second_debate_score
    con_first_rewards_per_func[:, 0] += con_second_debate_score
    pro_rewards = pro_first_rewards_per_func.sum(dim=1) # shape: (num_completions,)
    con_rewards = con_first_rewards_per_func.sum(dim=1) # shape: (num_completions,)

    # NEEDS TO BE MOVED BECAUSE THE REWARD IS OUT OF THE LOOP
    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(pro_completions_text, pro_first_rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores), # TODO: why do we sum before to then breakdown?
                'total_reward': pro_rewards[i].item()
            }
        }
        log_data['pro_generations'].append(generation_data)
    for i, (completion, reward_scores) in enumerate(zip(con_completions_text, con_first_rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': con_rewards[i].item()
            }
        }
        log_data['con_generations'].append(generation_data)
    
    metrics = {**pro_first_metrics, **con_first_metrics}

    return pro_rewards, con_rewards, pro_first_rewards_per_func, con_first_rewards_per_func, metrics, log_data


def compute_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel, 
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        completion_ids: Token IDs for just the completion
        attention_mask: Attention mask for the full sequence
        completion_mask: Mask indicating which tokens are from the completion
        advantages: Advantage values for each sequence
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits
    logits_to_keep = completion_ids.size(1)

    # Get reference model logits
    with torch.inference_mode():
        ref_per_token_logps = utils.get_per_token_logps(base_model, prompt_completion_ids, attention_mask, logits_to_keep)

    # Get training model logits
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute KL divergence
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # Compute loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - args.kl_weight_beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    metrics["kl"] = mean_kl.item()

    return loss, metrics

def compute_contrastive_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    pro_prompt_completion_ids: torch.Tensor,
    pro_prompt_ids: torch.Tensor,
    pro_completion_ids: torch.Tensor,
    pro_attention_mask: torch.Tensor,
    pro_completion_mask: torch.Tensor,
    pro_advantages: torch.Tensor,
    con_prompt_completion_ids: torch.Tensor,
    con_prompt_ids: torch.Tensor,
    con_completion_ids: torch.Tensor,
    con_attention_mask: torch.Tensor,
    con_completion_mask: torch.Tensor,
    con_advantages: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the contrastive GRPO loss for cross-comparison setup between PRO and CON stances.
    
    Based on Multi-Prompt GRPO from GRPO-README.md, this computes loss for both PRO and CON
    completions that are scored against each other in cross-comparison.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        pro_prompt_completion_ids: Combined prompt and completion token IDs for PRO stance
        pro_prompt_ids: Token IDs for just the PRO prompt
        pro_completion_ids: Token IDs for just the PRO completion
        pro_attention_mask: Attention mask for the PRO full sequence
        pro_completion_mask: Mask indicating which tokens are from the PRO completion
        pro_advantages: Advantage values for each PRO sequence
        con_prompt_completion_ids: Combined prompt and completion token IDs for CON stance
        con_prompt_ids: Token IDs for just the CON prompt
        con_completion_ids: Token IDs for just the CON completion
        con_attention_mask: Attention mask for the CON full sequence
        con_completion_mask: Mask indicating which tokens are from the CON completion
        con_advantages: Advantage values for each CON sequence
        args: Training arguments
        
    Returns:
        loss: The computed contrastive GRPO loss (sum of PRO and CON losses)
        metrics: Dictionary containing additional metrics like KL divergence for both stances
    """
    
    # Compute PRO stance loss
    pro_logits_to_keep = pro_completion_ids.size(1)
    
    # Get reference model logits for PRO
    with torch.inference_mode():
        pro_ref_per_token_logps = utils.get_per_token_logps(base_model, pro_prompt_completion_ids, pro_attention_mask, pro_logits_to_keep)
    
    # Get training model logits for PRO
    pro_input_ids = torch.cat([pro_prompt_ids, pro_completion_ids], dim=1)
    pro_per_token_logps = utils.get_per_token_logps(model, pro_input_ids, pro_attention_mask, pro_logits_to_keep)
    
    # Compute KL divergence for PRO
    pro_per_token_kl = torch.exp(pro_ref_per_token_logps - pro_per_token_logps) - (pro_ref_per_token_logps - pro_per_token_logps) - 1
    
    # Compute loss with advantages for PRO
    pro_per_token_loss = torch.exp(pro_per_token_logps - pro_per_token_logps.detach()) * pro_advantages.unsqueeze(1)
    pro_per_token_loss = -(pro_per_token_loss - args.kl_weight_beta * pro_per_token_kl)
    pro_loss = ((pro_per_token_loss * pro_completion_mask).sum(dim=1) / pro_completion_mask.sum(dim=1)).mean()
    
    # Compute CON stance loss
    con_logits_to_keep = con_completion_ids.size(1)
    
    # Get reference model logits for CON
    with torch.inference_mode():
        con_ref_per_token_logps = utils.get_per_token_logps(base_model, con_prompt_completion_ids, con_attention_mask, con_logits_to_keep)
    
    # Get training model logits for CON
    con_input_ids = torch.cat([con_prompt_ids, con_completion_ids], dim=1)
    con_per_token_logps = utils.get_per_token_logps(model, con_input_ids, con_attention_mask, con_logits_to_keep)
    
    # Compute KL divergence for CON
    con_per_token_kl = torch.exp(con_ref_per_token_logps - con_per_token_logps) - (con_ref_per_token_logps - con_per_token_logps) - 1
    
    # Compute loss with advantages for CON
    con_per_token_loss = torch.exp(con_per_token_logps - con_per_token_logps.detach()) * con_advantages.unsqueeze(1)
    con_per_token_loss = -(con_per_token_loss - args.kl_weight_beta * con_per_token_kl)
    con_loss = ((con_per_token_loss * con_completion_mask).sum(dim=1) / con_completion_mask.sum(dim=1)).mean()
    
    # Combined loss as sum over both prompt groups (following Multi-Prompt GRPO formula)
    total_loss = pro_loss + con_loss
    
    # Additional metrics
    metrics = {}
    
    # PRO stance metrics
    pro_response_length = pro_completion_mask.sum(1).float().mean().item()
    metrics["pro_response_length"] = pro_response_length
    pro_mean_kl = ((pro_per_token_kl * pro_completion_mask).sum(dim=1) / pro_completion_mask.sum(dim=1)).mean()
    metrics["pro_kl"] = pro_mean_kl.item()
    
    # CON stance metrics  
    con_response_length = con_completion_mask.sum(1).float().mean().item()
    metrics["con_response_length"] = con_response_length
    con_mean_kl = ((con_per_token_kl * con_completion_mask).sum(dim=1) / con_completion_mask.sum(dim=1)).mean()
    metrics["con_kl"] = con_mean_kl.item()
    
    # Combined metrics
    metrics["total_response_length"] = (pro_response_length + con_response_length) / 2
    metrics["total_kl"] = (pro_mean_kl + con_mean_kl).item() / 2
    metrics["pro_loss"] = pro_loss.item()
    metrics["con_loss"] = con_loss.item()
    
    return total_loss, metrics

def grpo_loss(
        train_loader,
        all_models: dict,
        question: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """

    prompt = [
        {'role': 'system', 'content': test_loader.pre_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)

    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
    )
    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, eval_class, device, args
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    loss, loss_metrics = compute_loss(
        all_models["training_model"], all_models["base_model"], prompt_completion_ids, prompt_ids, completion_ids,
        attention_mask, completion_mask, advantages, args
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

def grpo_contrastive_loss(
        train_loader,
        all_models: dict,
        question: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute contrastive GRPO loss between PRO and CON stances.
    
    Args:
        train_loader: Training data loader
        all_models: Dictionary of all models
        question: Input question/prompt
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed contrastive GRPO loss
        metrics: Dictionary containing training metrics
    """

    prompt_candidate = [
        {'role': 'system', 'content': train_loader.pre_prompt},
        {'role': 'user', 'content': question + "Position: PRO"}
    ]
    prompt_opponent = [
        {'role': 'system', 'content': train_loader.pre_prompt},
        {'role': 'user', 'content': question + "Position: CON"}
    ]

    prompt_text_candidate = all_models["training_model_tokenizer"].apply_chat_template(prompt_candidate, tokenize=False)
    prompt_text_opponent  = all_models["training_model_tokenizer"].apply_chat_template(prompt_opponent, tokenize=False)
    # print(f"Generating completions for PRO prompt: {prompt_text_candidate}")
    # print(f"Generating completions for CON prompt: {prompt_text_opponent}")

    # Generate completions
    pro_prompt_completion_ids, pro_prompt_ids, pro_completion_ids, pro_attention_mask, pro_completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text_candidate, device, args
    ) # outputs 8 completions for PRO stance
    con_prompt_completion_ids, con_prompt_ids, con_completion_ids, con_attention_mask, con_completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text_opponent, device, args
    ) # outputs 8 completions for CON stance
    
    # Score completions (cross-comparison between PRO and CON)
    pro_rewards, con_rewards, pro_first_rewards_per_func, con_first_rewards_per_func, metrics, log_data = score_contrastive_completions(
        pro_completions_text, con_completions_text, prompt_text_candidate, eval_class, device, args
    )

    def compute_contrastive_advantages(
        pro_rewards: torch.Tensor, con_rewards: torch.Tensor,
        log_data: dict, metrics:dict
    ) -> tuple[torch.Tensor, dict]:
        mean_grouped_pro_rewards = pro_rewards.view(-1, args.num_chains).mean(dim=1)
        mean_grouped_con_rewards = con_rewards.view(-1, args.num_chains).mean(dim=1)
        # print(f"Pro rewards shape: {pro_rewards.shape}, Con rewards shape: {con_rewards.shape}")
        # print(f"Pro rewards view shape: {pro_rewards.view(-1, args.num_chains).shape}, Con rewards view shape: {con_rewards.view(-1, args.num_chains).shape}")
        print(f"\nPro rewards: {pro_rewards}")
        print(f"Con rewards: {con_rewards}")
        # print(f"Mean grouped pro rewards shape: {mean_grouped_pro_rewards.shape}, Mean grouped con rewards shape: {mean_grouped_con_rewards.shape}")
        # print(f"Mean grouped pro rewards: {mean_grouped_pro_rewards}")
        # print(f"Mean grouped con rewards: {mean_grouped_con_rewards}")  
        # Repeat mean rewards to match original shape
        mean_grouped_pro_rewards = mean_grouped_pro_rewards.repeat_interleave(args.num_chains, dim=0)
        mean_grouped_con_rewards = mean_grouped_con_rewards.repeat_interleave(args.num_chains, dim=0)   
        # print(f"Mean grouped pro rewards after repeat shape: {mean_grouped_pro_rewards.shape}")
        # print(f"Mean grouped pro rewards after repeat: {mean_grouped_pro_rewards}")
        # print(f"Mean grouped con rewards after repeat shape: {mean_grouped_con_rewards.shape}")
        # print(f"Mean grouped con rewards after repeat: {mean_grouped_con_rewards}")
        std_grouped_pro_rewards = pro_rewards.view(-1, args.num_chains).std(dim=1)
        std_grouped_con_rewards = con_rewards.view(-1, args.num_chains).std(dim=1)
        # print(f"Std grouped pro rewards shape: {std_grouped_pro_rewards.shape}, Std grouped con rewards shape: {std_grouped_con_rewards.shape}")
        # print(f"Std grouped pro rewards: {std_grouped_pro_rewards}")
        # print(f"Std grouped con rewards: {std_grouped_con_rewards}")
        # Repeat std rewards to match original shape
        std_grouped_pro_rewards = std_grouped_pro_rewards.repeat_interleave(args.num_chains, dim=0)
        std_grouped_con_rewards = std_grouped_con_rewards.repeat_interleave(args.num_chains, dim=0)
        # print(f"Std grouped pro rewards after repeat shape: {std_grouped_pro_rewards.shape}")
        # print(f"Std grouped pro rewards after repeat: {std_grouped_pro_rewards}")
        # print(f"Std grouped con rewards after repeat shape: {std_grouped_con_rewards.shape}")
        # print(f"Std grouped con rewards after repeat: {std_grouped_con_rewards}")
        # Compute advantages
        pro_advantages = (pro_rewards - mean_grouped_pro_rewards) / (std_grouped_pro_rewards + 1e-4)
        con_advantages = (con_rewards - mean_grouped_con_rewards) / (std_grouped_con_rewards + 1e-4)
        # print(f"Pro advantages shape: {pro_advantages.shape}")
        # print(f"Con advantages shape: {con_advantages.shape}")
        print(f"Pro advantages: {pro_advantages}")
        print(f"Con advantages: {con_advantages}")

        metrics["pro_reward_std"] = std_grouped_pro_rewards.mean().item()
        metrics["con_reward_std"] = std_grouped_con_rewards.mean().item()
        # print(f"Pro Reward std: {metrics['pro_reward_std']}")
        # print(f"Con Reward std: {metrics['con_reward_std']}")
        print(f"Metrics: {metrics}")

        # Store summary statistics
        log_data['pro_summary_stats'] = {
            'mean_rewards_per_group': mean_grouped_pro_rewards.tolist(),
            'std_rewards_per_group': std_grouped_pro_rewards.tolist(),
            'advantages': pro_advantages.tolist()
        }
        log_data['con_summary_stats'] = {
            'mean_rewards_per_group': mean_grouped_con_rewards.tolist(),
            'std_rewards_per_group': std_grouped_con_rewards.tolist(),
            'advantages': con_advantages.tolist()
        }
        return pro_advantages, con_advantages, log_data
        
    pro_advantages, con_advantages, log_data = compute_contrastive_advantages(
        pro_rewards, con_rewards, log_data, metrics
    )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_contrastive_generations.txt')
    utils.write_contrastive_generation_log(log_data, log_file)

    # Compute contrastive loss
    pro_completion_mask = pro_attention_mask[:, pro_prompt_ids.size(1):]
    con_completion_mask = con_attention_mask[:, con_prompt_ids.size(1):]
    
    loss, loss_metrics = compute_contrastive_loss(
        all_models["training_model"], all_models["base_model"], 
        pro_prompt_completion_ids, pro_prompt_ids, pro_completion_ids, 
        pro_attention_mask, pro_completion_mask, pro_advantages,
        con_prompt_completion_ids, con_prompt_ids, con_completion_ids,
        con_attention_mask, con_completion_mask, con_advantages, args
    )

    # Combine metrics
    combined_metrics = {**metrics, **loss_metrics}

    return loss, combined_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--judge_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name of model to use as judge")
    parser.add_argument("--compare_model_name", type=str, default="gpt-4o-mini", help="Name of model to use for comparison")
    parser.add_argument("--dataset_name", type=str, default="debate", choices=["debate", "ld", "chopped"], help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="debate", choices=["debate", "ld", "chopped"], help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--enable_detailed_logging", action="store_true", help="Enable detailed logging throughout training")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-debate", help="Wandb project name")
    parser.add_argument("--wandb_key_file", type=str, default="wandb_key.txt", help="Path to file containing wandb API key")
    parser.add_argument("--save_steps", type=int, default=80, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=40, help="Number of iterations for evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for warmup")
    parser.add_argument("--update_ref_model", action="store_true", help="Whether to update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="How often to update reference model")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Alpha parameter for reference model mixup")


    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")
    parser.add_argument("--use_batch_generation", action="store_true", help="Use batched generation for compare model (more efficient for HuggingFace models)")
    parser.add_argument("--use_batch_judge", action="store_true", help="Use batched evaluation for judge model (much faster for HuggingFace judge models)")
    parser.add_argument("--use_semi_batch_judge", action="store_true", help="Use semi-batched evaluation for judge model (batch inner loop only, reduces memory usage)")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")
    parser.add_argument("--use_contrastive", action="store_true", help="Use contrastive GRPO training (PRO vs CON)")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_detailed_logging:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting GRPO training")
        logger.info(f"Arguments: {vars(args)}")
    else:
        # Setup minimal logging
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
    
    # Setup Weights & Biases logging
    if args.enable_wandb and WANDB_AVAILABLE:
        print("Setting up Weights & Biases logging...")
        try:
            # Read wandb key from file
            if os.path.exists(args.wandb_key_file):
                print(f"Reading wandb API key from {args.wandb_key_file}")
                with open(args.wandb_key_file, 'r') as f:
                    wandb_key = f.read().strip()
                os.environ['WANDB_API_KEY'] = wandb_key
                print("✓ API key loaded successfully")
                
                # Initialize wandb
                run_name = f"{args.model_name.split('/')[-1]}_{args.dataset_name}_{args.evaluator}"
                print(f"Initializing wandb project: {args.wandb_project}")
                print(f"Run name: {run_name}")
                
                wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    resume="allow" if args.resume else None
                )
                print("✓ Wandb initialized successfully!")
                print(f"🔗 View your run at: https://wandb.ai/{wandb.run.entity}/{args.wandb_project}/runs/{wandb.run.id}")
                
                if args.enable_detailed_logging:
                    logger.info(f"Initialized wandb project: {args.wandb_project}, run: {run_name}")
            else:
                print(f"❌ Error: Wandb key file {args.wandb_key_file} not found.")
                print("Please create the file and add your wandb API key to enable logging.")
                args.enable_wandb = False
        except Exception as e:
            print(f"❌ Error: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            args.enable_wandb = False
    elif args.enable_wandb and not WANDB_AVAILABLE:
        print("❌ Error: wandb requested but not installed.")
        print("Install with: pip install wandb")
        args.enable_wandb = False
    elif not args.enable_wandb:
        print("Wandb logging disabled (use --enable_wandb to enable)")
    else:
        print("Wandb logging enabled - ERROR")

    # Seed everything 
    utils.seed_everything(args.seed)
    if args.enable_detailed_logging:
        logger.info(f"Random seed set to {args.seed}")

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.enable_detailed_logging:
        logger.info(f"Using device: {device}")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    ## Set which model to train 
    if args.enable_detailed_logging:
        logger.info(f"Loading training model: {args.model_name}")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    if args.enable_detailed_logging:
        logger.info(f"Using same model for base model: {args.model_name}")
    base_model = model

    # Get judge and compare models using the new interfaces
    if args.enable_detailed_logging:
        logger.info(f"Using same model for judge model: {args.model_name}")
    judge_model = llms.get_judge_model(args.model_name, device)
    if args.enable_detailed_logging:
        logger.info(f"Loading compare model: {args.compare_model_name}")
    compare_model = judge_model
    
    # Simplified all_models dictionary
    all_models = {
        "training_model": model,
        "training_model_tokenizer": tokenizer,
        "base_model": base_model,
        "base_model_tokenizer": tokenizer,
        "judge_model": judge_model,
        "compare_model": compare_model
    }
    if args.enable_detailed_logging:
        logger.info("All models loaded successfully")

    ## Set which data set 
    if args.enable_detailed_logging:
        logger.info(f"Loading dataset: {args.dataset_name}")
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name, args.use_contrastive,)
    if args.enable_detailed_logging:
        logger.info(f"Dataset loaded successfully")

    ## Set which evaluation criteria to use 
    if args.enable_detailed_logging:
        logger.info(f"Loading evaluator: {args.evaluator}")
    eval_class = evaluator.get_evaluator(args.evaluator, contrastive=args.use_contrastive)
    if args.enable_detailed_logging:
        logger.info(f"Evaluator loaded successfully")


    # Setup logging directories and save arguments
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    if args.enable_detailed_logging:
        logger.info(f"Saved training arguments to {args_path}")
    
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.enable_detailed_logging:
        logger.info(f"Created output directories: {args.output_dir}")

    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        all_models["training_model"].parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    if args.enable_detailed_logging:
        logger.info(f"Optimizer initialized with lr={args.learning_rate}, weight_decay={args.weight_decay}")

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)
    if args.enable_detailed_logging:
        logger.info(f"Scheduler initialized with {warmup_steps} warmup steps ({args.warmup_percent:.0%} of {args.num_train_iters} total steps)")

    # Resume from checkpoint if requested
    start_round = 0
    if args.resume:
        checkpoints = sorted([int(f.split('_')[1].split('.')[0]) for f in os.listdir(checkpoint_dir) if f.startswith('step_')])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{latest_checkpoint}.pt')
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_round = checkpoint['round_num'] + 1
            train_metrics_total = checkpoint['train_metrics_total']
            if args.enable_detailed_logging:
                logger.info(f"Resuming from checkpoint at step {latest_checkpoint}")
        else:
            if args.enable_detailed_logging:
                logger.info("No checkpoints found, starting from scratch")
            train_metrics_total = {}
    else:
        if args.enable_detailed_logging:
            logger.info("Starting fresh training (no resume)")
        train_metrics_total = {}

    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    
    if args.enable_detailed_logging:
        logger.info(f"Starting training loop: {start_round} to {args.num_train_iters-1}")
        logger.info(f"Total training steps: {args.num_train_iters - start_round}")
        logger.info(f"Evaluation every {args.eval_iterations} steps")
        logger.info(f"Saving checkpoints every {args.save_steps} steps")

    #=======================================================================================================================
    #==============================================      TRAINING LOOP       ===============================================
    #=======================================================================================================================
    for round_num in tqdm(range(start_round, args.num_train_iters), desc="Training Progress"):
        if args.enable_detailed_logging and round_num % 100 == 0:  # Log every 100 rounds to avoid spam
            logger.info(f"Training round {round_num}/{args.num_train_iters-1}")
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0 and round_num > 0:
            if args.enable_detailed_logging:
                logger.info(f"Starting evaluation at round {round_num}")
            eval_metrics, eval_accuracy = eval_on_test_set(
                all_models=all_models,
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            if args.enable_detailed_logging:
                logger.info(f"Evaluation completed - Win rate: {eval_accuracy:.2f}%")
                logger.info(f"Evaluation metrics: {eval_metrics}")
            
            # Log evaluation to wandb
            if args.enable_wandb:
                eval_wandb_log = {
                    "eval/win_rate": eval_accuracy,
                    "eval/total_wins": eval_metrics.get("total_wins", 0),
                    "eval/total_comparisons": eval_metrics.get("total_comparisons", 0),
                    "eval/num_examples": eval_metrics.get("num_examples", 0),
                    "step": round_num
                }
                
                # Add average scores
                avg_scores = eval_metrics.get("average_scores", {})
                for key, value in avg_scores.items():
                    eval_wandb_log[f"eval/{key}"] = value
                
                wandb.log(eval_wandb_log)
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)
            if args.enable_detailed_logging:
                logger.info(f"Evaluation results saved to {metrics_path}")

        # Save checkpoint
        if (round_num + 1) % args.save_steps == 0 or round_num == args.num_train_iters - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics_total': train_metrics_total
            }, checkpoint_path)
            if args.enable_detailed_logging:
                logger.info(f"Checkpoint saved at round {round_num}: {checkpoint_path}")

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data
            if args.enable_detailed_logging:
                logger.info(f"Reference model updated at round {round_num} with alpha={args.ref_model_mixup_alpha}")

        # Get next question
        question = next(train_loader)

        # Clear cache before GRPO
        torch.cuda.empty_cache()
        
        #=======================================================================================================================
        #==============================================      TRAINING CALL       ===============================================
        #=======================================================================================================================
        print(question)
        # Do GRPO - generate chains, score, compute advantage, compute loss 
        if args.use_contrastive:
            total_loss, train_metrics = grpo_contrastive_loss(train_loader, all_models, question, eval_class, 
                                                            device, round_num, train_log_dir, args)
        else:
            total_loss, train_metrics = grpo_loss(train_loader, all_models, question, eval_class, 
                                                device, round_num, train_log_dir, args)
        
        # Gradient accumulation
        total_loss = total_loss
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()
        
        # Log training metrics periodically
        if args.enable_detailed_logging and round_num % 50 == 0:
            logger.info(f"Round {round_num}: loss={total_loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            if args.enable_detailed_logging and round_num % 100 == 0:
                logger.info(f"Gradient step taken at round {round_num}, grad_norm={grad_norm:.4f}")    

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
            
        # Log detailed metrics periodically
        if args.enable_detailed_logging and round_num % 100 == 0 and 'rewards/strict_format' in train_metrics:
            logger.info(f"Round {round_num} detailed metrics: ")
            for key, value in train_metrics.items():
                if key.startswith('rewards/'):
                    logger.info(f"  {key}: {value:.4f}")
        
        # Log to wandb
        if args.enable_wandb:
            wandb_log = {
                "train/loss": train_metrics.get("loss", 0),
                "train/learning_rate": train_metrics.get("learning_rate", 0),
                "train/grad_norm": train_metrics.get("grad_norm", 0),
                "train/kl": train_metrics.get("kl", 0),
                "train/response_length": train_metrics.get("response_length", 0),
                "train/reward_std": train_metrics.get("reward_std", 0),
                "step": round_num
            }
            
            # Add reward metrics if available
            for key, value in train_metrics.items():
                if key.startswith('rewards/'):
                    wandb_log[f"train/{key}"] = value
            
            wandb.log(wandb_log)

        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
        
        # Additional memory cleanup
        if round_num % 10 == 0:  # Every 10 rounds
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    if args.enable_detailed_logging:
        logger.info("Training completed successfully!")
    
    # Finish wandb run
    if args.enable_wandb:
        wandb.finish()
    
