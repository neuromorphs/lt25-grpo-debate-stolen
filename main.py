"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface
import random

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
    
    total_scores = defaultdict(float)
    num_examples = 0
    total_wins = 0

    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        total_comparisons = 0
        total_wins = 0
        
        for (question, positions) in tqdm(test_loader, desc="Evaluating on test set"):
            num_examples += 1
            id_pos = random.choice([0, 1]) # TODO
            # 1. Prepare prompting
            prompt = [
                {'role': 'system', 'content': test_loader.pre_prompt},
                {'role': 'user', 'content': question + f"\nPosition you have to defend: {positions[id_pos]}"}
            ]
            prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)
            if args.contrastive_eval:
                prompt_2 = prompt.copy()
                prompt_2[1]['content'] = question + f"\nPosition you have to defend: {positions[1-id_pos]}"
                prompt_text_2 = all_models["training_model_tokenizer"].apply_chat_template(prompt_2, tokenize=False)


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

            # Generate completions for compare model using the interface
            compare_completions_text = []
            for _ in range(args.num_chains):
                completion = all_models["compare_model"].generate(
                    system_prompt=test_loader.pre_prompt,
                    user_prompt=prompt_2[1]['content'] if args.contrastive_eval else prompt[1]['content'],
                    max_new_tokens=args.max_completion_length,
                    temperature=args.temperature
                )
                compare_completions_text.append(completion)

            # Score completions to get reward metrics
            rewards_per_func, reward_metrics = eval_class.compute_rewards(
                input_prompt=prompt_2[1]['content'] if args.contrastive_eval else prompt[1]['content'], 
                all_models=all_models, 
                train_model_completions=completions_text, 
                compare_model_completions=compare_completions_text,
                device=device,
                is_test=True
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

                # Log judge response for this specific comparison
                if hasattr(eval_class, 'last_judge_responses') and eval_class.last_judge_responses and i < len(eval_class.last_judge_responses):
                    f.write(f"\nJUDGE RESPONSE:\n{eval_class.last_judge_responses[i]}\n")

                # Log if trained model won this comparison
                trained_model_won = rewards_per_func[i,0] > 0
                f.write(f"\nOUTCOME: Trained model {'won' if trained_model_won else 'lost'} this comparison\n")
                f.write("-"*40 + "\n")

            # Log summary metrics for this question
            f.write("\nSUMMARY METRICS:\n")
            f.write(f"Win rate: {reward_metrics['win_rate']:.2%}\n")
            f.write(f"Number of wins: {reward_metrics['num_wins']}\n")
            #f.write(f"Total comparisons: {reward_metrics['num_comparisons']}\n")
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

def compute_contrastive_advantages(
    rewards: torch.Tensor, rewards_2: torch.Tensor,
    log_data: dict, metrics:dict
) -> tuple[torch.Tensor, dict]:
    mean_grouped_rewards   = rewards.view(-1, args.num_chains).mean(dim=1)
    mean_grouped_rewards_2 = rewards_2.view(-1, args.num_chains).mean(dim=1)
    print(f"\nPro rewards: {rewards}")
    print(f"Con rewards: {rewards_2}")
    # Repeat mean rewards to match original shape
    mean_grouped_rewards   = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    mean_grouped_rewards_2 = mean_grouped_rewards_2.repeat_interleave(args.num_chains, dim=0)   
    std_grouped_rewards   = rewards.view(-1, args.num_chains).std(dim=1)
    std_grouped_rewards_2 = rewards_2.view(-1, args.num_chains).std(dim=1)
    # Repeat std rewards to match original shape
    std_grouped_rewards   = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards_2 = std_grouped_rewards_2.repeat_interleave(args.num_chains, dim=0)
    # Compute advantages
    advantages   = (rewards   - mean_grouped_rewards  ) / (std_grouped_rewards   + 1e-4)
    advantages_2 = (rewards_2 - mean_grouped_rewards_2) / (std_grouped_rewards_2 + 1e-4)
    print(f"advantages: {advantages}")
    print(f"advantages_2: {advantages_2}")

    metrics["reward_std"] = std_grouped_rewards.mean().item()
    metrics["reward_std_2"] = std_grouped_rewards_2.mean().item()
    # print(f"Pro Reward std: {metrics['pro_reward_std']}")
    # print(f"Con Reward std: {metrics['con_reward_std']}")
    print(f"Metrics: {metrics}")

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }
    log_data['summary_stats_2'] = {
        'mean_rewards_per_group': mean_grouped_rewards_2.tolist(),
        'std_rewards_per_group': std_grouped_rewards_2.tolist(),
        'advantages': advantages_2.tolist()
    }
    return advantages, advantages_2, log_data

def score_contrastive_completions(
    completions_text: list[str],
    completions_text_2: list[str],
    input_prompt: str,
    positions: list[str],
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': input_prompt['question'],
        },
        'pro_generations': [],
        'con_generations': []
    }

    # Compute both scenarios and average debate scores
    scenarios = [   
        (completions_text,   completions_text_2, True,  'one_first'), 
        (completions_text_2, completions_text,   False, 'two_first')
    ]

    rewards_data = {}
    for train_comps, comp_comps, one_first, order in scenarios:
        rewards_per_func, stance_metrics, cross_score = eval_class.compute_rewards(
            input_prompt=input_prompt, all_models=all_models,
            train_model_completions=train_comps, compare_model_completions=comp_comps,
            device=device, is_test=False, contrastive=True, one_first=one_first
        )
        
        # Store for cross-averaging
        rewards_data[order] = (rewards_per_func, stance_metrics, cross_score)

    # Average debate scores
    rewards_per_func,   metrics,   _ = rewards_data['one_first']
    rewards_per_func_2, metrics_2, _ = rewards_data['two_first']

    if args.dataset_name == "debate_code" and args.truth_optim:
        win_rate   = (rewards_per_func[:, 0]   + 1 - rewards_data['two_first'][2]) / 2
        win_rate_2 = (rewards_per_func_2[:, 0] + 1 - rewards_data['one_first'][2]) / 2
    else:
        win_rate   = (rewards_per_func[:, 0]   + rewards_data['two_first'][2]) / 2
        win_rate_2 = (rewards_per_func_2[:, 0] + rewards_data['one_first'][2]) / 2
    
    rewards_per_func[:, 0]   = (2 * win_rate   - 1) * 1.5 
    rewards_per_func_2[:, 0] = (2 * win_rate_2 - 1) * 1.5

    metrics['rewards/mean_debate_score'] = rewards_per_func[:, 0].mean().item()
    metrics_2['rewards/mean_debate_score'] = rewards_per_func_2[:, 0].mean().item()

    metrics['mean_rewards'] = rewards_per_func.mean().item()
    metrics_2['mean_rewards'] = rewards_per_func_2.mean().item()

    # Final data
    data_map = ((completions_text, rewards_per_func, positions[0]), 
                (completions_text_2, rewards_per_func_2, positions[1]))

    for _, (completions, rewards_per_func, pos) in enumerate(data_map):
        rewards = rewards_per_func.sum(dim=1)
        log_data[f'{pos}_generations'] = [{
            'response': comp,
            'scores': {**eval_class.get_reward_breakdown(scores), 'total_reward': rewards[i].item()}
        } for i, (comp, scores) in enumerate(zip(completions, rewards_per_func))]

    truth_metrics={'train/truth_win_rate': rewards_per_func[:, 0].mean().item()/1.5}

    # modify every key of pro_metrics with a pro_ prefix
    metrics   = {f'{positions[0]}_{k}': v for k, v in metrics.items()}
    metrics_2 = {f'{positions[1]}_{k}': v for k, v in metrics_2.items()}
    metrics = {**metrics, **metrics_2, **truth_metrics}
    rewards, rewards_2 = rewards_per_func.sum(dim=1), rewards_per_func_2.sum(dim=1)

    # Compute advantages
    advantages, advantages_2, log_data = compute_contrastive_advantages(
        rewards, rewards_2, log_data, metrics
    )

    return rewards, advantages, rewards_per_func, rewards_2, advantages_2, rewards_per_func_2, metrics, log_data

def compute_contrastive_loss(
    model: PreTrainedModel,
    base_model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    prompt_completion_ids_2: torch.Tensor,
    prompt_ids_2: torch.Tensor,
    completion_ids_2: torch.Tensor,
    attention_mask_2: torch.Tensor,
    completion_mask_2: torch.Tensor,
    advantages_2: torch.Tensor,
    args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the contrastive GRPO loss for cross-comparison setup between PRO and CON stances.
    
    Based on Multi-Prompt GRPO from GRPO-README.md, this computes loss for both PRO and CON
    completions that are scored against each other in cross-comparison.
    
    Args:
        model: The current model being trained
        base_model: The reference model to compare against
        prompt_completion_ids: Combined prompt and completion token IDs for PRO stance
        prompt_ids: Token IDs for just the PRO prompt
        completion_ids: Token IDs for just the PRO completion
        attention_mask: Attention mask for the PRO full sequence
        completion_mask: Mask indicating which tokens are from the PRO completion
        advantages: Advantage values for each PRO sequence
        prompt_completion_ids_2: Combined prompt and completion token IDs for CON stance
        prompt_ids_2: Token IDs for just the CON prompt
        completion_ids_2: Token IDs for just the CON completion
        attention_mask_2: Attention mask for the CON full sequence
        completion_mask_2: Mask indicating which tokens are from the CON completion
        advantages_2: Advantage values for each CON sequence
        args: Training arguments
        
    Returns:
        loss: The computed contrastive GRPO loss (sum of PRO and CON losses)
        metrics: Dictionary containing additional metrics like KL divergence for both stances
    """
    
    loss, metrics, intermediary = compute_loss(model, base_model, prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, advantages, args)
    loss_2, metrics_2, intermediary_2 = compute_loss(model, base_model, prompt_completion_ids_2, prompt_ids_2, completion_ids_2, attention_mask_2, completion_mask_2, advantages_2, args)
        
    # Combined loss as sum over both prompt groups (following Multi-Prompt GRPO formula)
    total_loss = loss + loss_2
    
    # Additional metrics
    metrics = {}
    
    # PRO stance metrics
    response_length = completion_mask.sum(1).float().mean().item()
    print(f"response length: {response_length}")
    # print(f"PRO PER TOKEN KL: {per_token_kl}")
    metrics["response_length"] = response_length
    mean_kl = ((intermediary['per_token_kl'] * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    print(f"PRO mean KL: {mean_kl}")
    metrics["kl"] = mean_kl.item()
    
    # CON stance metrics  
    response_length_2 = completion_mask_2.sum(1).float().mean().item()
    print(f"response length 2: {response_length_2}")
    # print(f"CON PER TOKEN KL: {per_token_kl_2}")
    metrics["response_length_2"] = response_length_2
    mean_kl_2 = ((intermediary['per_token_kl'] * completion_mask_2).sum(dim=1) / completion_mask_2.sum(dim=1)).mean()
    print(f"mean KL 2: {mean_kl_2}")
    metrics["kl_2"] = mean_kl_2.item()
    
    # Combined metrics
    metrics["mean_response_length"] = (response_length + response_length_2) / 2
    metrics["mean_kl"] = (mean_kl + mean_kl_2).item() / 2
    metrics["loss"] = loss.item()
    metrics["loss_2"] = loss_2.item()
    
    return total_loss, metrics


def score_completions(
    completions_text: list[str],
    input_prompt: str,
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
        input_prompt=input_prompt,
        all_models=all_models, 
        train_model_completions=completions_text, 
        compare_model_completions=None,
        device=device, 
        is_test=False,
        contrastive=False,
        one_first=None
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
        
        # Log judge response for this specific comparison
        if hasattr(eval_class, 'last_judge_responses') and eval_class.last_judge_responses and i < len(eval_class.last_judge_responses):
            generation_data['judge_response'] = eval_class.last_judge_responses[i]
            
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

    return loss, metrics, {'per_token_kl': per_token_kl, 'response_length': response_length, 'mean_kl': mean_kl}

def grpo_loss(
        train_loader,
        all_models: dict,
        question: str,
        positions: list[str],
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

    
    # position_id = 0 if args.dataset_name == "debate_code" else random.choice([0,1]) 
    
    prompt = [
        {'role': 'system', 'content': test_loader.pre_prompt},
        {'role': 'user', 'content': question + f"\nPosition you have to defend: {positions[0]}"}
    ]
    prompt_text = all_models["training_model_tokenizer"].apply_chat_template(prompt, tokenize=False)

    # Generate completions
    prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, _ = generate_completions(
        all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
    )
    input_prompt = {
        'question': question,
        'positions': positions,
    }
    if args.contrastive_training: 
        # raise NotImplementedError("Contrastive training not implemented in this function")
        prompt_2 = prompt.copy()
        prompt_2[1]['content'] = question + f"\nPosition you have to defend: {positions[1]}"
        prompt_text_2 = all_models["training_model_tokenizer"].apply_chat_template(prompt_2, tokenize=False)
        prompt_completion_ids_2, prompt_2_ids, completion_ids_2, attention_mask_2, completions_text_2, _ = generate_completions(
            all_models["training_model"], all_models["training_model_tokenizer"], prompt_text, device, args
        )

        rewards, advantages, rewards_per_func, rewards_2, advantages_2, rewards_per_func_2, metrics, log_data = score_contrastive_completions(
        completions_text, completions_text_2, input_prompt, positions, eval_class, device, args
    ) 
    else: 
        # Score completions
        rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
            completions_text, input_prompt, eval_class, device, args
        )

    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    if args.contrastive_training:
        completion_mask = attention_mask[:, prompt_ids.size(1):]
        completion_mask_2 = attention_mask_2[:, prompt_2_ids.size(1):]
        # raise NotImplementedError("Contrastive training not implemented in this function")
        loss, loss_metrics = compute_contrastive_loss(
            all_models["training_model"], all_models["base_model"],
            prompt_completion_ids,   prompt_ids,   completion_ids,   attention_mask,   completion_mask,   advantages,
            prompt_completion_ids_2, prompt_2_ids, completion_ids_2, attention_mask_2, completion_mask_2, advantages_2, args
        )
    else:
        completion_mask = attention_mask[:, prompt_ids.size(1):]
        loss, loss_metrics, _ = compute_loss(
            all_models["training_model"], all_models["base_model"], prompt_completion_ids, prompt_ids, completion_ids,
            attention_mask, completion_mask, advantages, args
        )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics

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

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")

    # new parameters
    parser.add_argument("--contrastive_training", type=str, default="false", choices=["true", "false", "True", "False"], help="Whether to use contrastive training (PRO vs CON)")
    parser.add_argument("--contrastive_eval", type=str, default="false", choices=["true", "false", "True", "False"], help="Whether to use contrastive training (PRO vs CON)")
    parser.add_argument("--truth_optim", action="store_true", help="Use truth optimization for debate_code")
    parser.add_argument("--truth_comparison", action="store_true", help="Use truth comparison in the judge prompt")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for additional logging and checks")
    parser.add_argument("--enable_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-debate", help="Wandb project name")
    parser.add_argument("--wandb_key_file", type=str, default="wandb_key.txt", help="Path to file containing wandb API key")
    
    args = parser.parse_args()
    
    # Convert string arguments to boolean
    args.contrastive_training = args.contrastive_training.lower() == "true"

    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    if args.dataset_name == "debate_code" and args.contrastive_training == False:
        raise ValueError("Contrastive training must be enabled for debate_code dataset")

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
                run_name = f"{args.model_name.split('/')[-1]}_{args.truth_optim}"
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

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    base_model, _ = llms.get_llm_tokenizer(args.model_name, device)

    # Get judge and compare models using the new interfaces
    judge_model = llms.get_judge_model(args.judge_model_name, device)
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

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name, debug=args.debug)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)


    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        all_models["training_model"].parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)

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
            print(f"Resuming from checkpoint at step {latest_checkpoint}")
        else:
            print("No checkpoints found, starting from scratch")
            train_metrics_total = {}
    else:
        train_metrics_total = {}

    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()

    for round_num in tqdm(range(start_round, args.num_train_iters), desc="Training Progress"): #TLOOP
        print(f"Round {round_num}")
        # Evaluate on test set every so often 
        if round_num % args.eval_iterations == 0 and round_num > 0: #TODO: eval
            eval_metrics, eval_accuracy = eval_on_test_set(
                all_models=all_models,
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )

            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

        # Save checkpoint
        if (round_num + 1) % args.save_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'step_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics_total': train_metrics_total
            }, checkpoint_path)

        # Slowly update ref model
        if args.update_ref_model and (round_num+1) % args.update_ref_model_freq == 0:
            with torch.no_grad():
                for param, ref_param in zip(model.parameters(), base_model.parameters()):
                    ref_param.data = args.ref_model_mixup_alpha * param.data + (1 - args.ref_model_mixup_alpha) * ref_param.data

        # Get next question
        question, positions = next(train_loader)

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(train_loader, all_models, question, positions, eval_class, device, round_num, train_log_dir, args) # TCALL
        
        # Gradient accumulation
        total_loss = total_loss
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    

        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)

        if args.enable_wandb:
            wandb_log = {
            }
            if args.contrastive_training:
                wandb_log.update({
                f"train/{positions[0]}_rewards/mean_win_rate_first": train_metrics.get(f"{positions[0]}_rewards/mean_win_rate_first", 0),
                f"train/{positions[0]}_rewards/mean_win_rate_second": train_metrics.get(f"{positions[0]}_rewards/mean_win_rate_second", 0),
                f"train/{positions[0]}_rewards/strict_format": train_metrics.get(f"{positions[0]}_rewards/strict_format", 0),
                f"train/{positions[0]}_rewards/soft_format": train_metrics.get(f"{positions[0]}_rewards/soft_format", 0),
                f"train/{positions[0]}_rewards/xml_count": train_metrics.get(f"{positions[0]}_rewards/xml_count", 0),
                f"train/{positions[0]}_reward": train_metrics.get(f"{positions[0]}_reward", 0),
                f"train/{positions[1]}_rewards/mean_win_rate_first": train_metrics.get(f"{positions[1]}_rewards/mean_win_rate_first", 0),
                f"train/{positions[1]}_rewards/mean_win_rate_second": train_metrics.get(f"{positions[1]}_rewards/mean_win_rate_second", 0),
                f"train/{positions[1]}_rewards/strict_format": train_metrics.get(f"{positions[1]}_rewards/strict_format", 0),
                f"train/{positions[1]}_rewards/soft_format": train_metrics.get(f"{positions[1]}_rewards/soft_format", 0),
                f"train/{positions[1]}_rewards/xml_count": train_metrics.get(f"{positions[1]}_rewards/xml_count", 0),
                f"train/{positions[1]}_reward": train_metrics.get(f"{positions[1]}_reward", 0),
                })
            
            # # Add reward metrics if available (including prefixed ones)
            # for key, value in train_metrics.items():
            #     if key.startswith('rewards/') or key.startswith('pro_rewards/') or key.startswith('con_rewards/'):
            #         wandb_log[f"train/{key}"] = value
            
            wandb.log(wandb_log)


        # Add after each major operation in the training loop
        torch.cuda.empty_cache()
    
