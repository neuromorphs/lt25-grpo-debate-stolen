import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional

import re

####################
## MISC FUNCTIONS ##
####################

def clean_spaces_preserve_newlines(text):
    # Replace multiple spaces with a single space, but preserve newlines
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [" ".join(re.split(r"\s+", line)).strip() for line in lines]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines



def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: Dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, 'w') as f:
        # Write prompt section
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")

        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} ####\n\n")
            f.write("RESPONSE:\n")
            f.write(gen['response'] + "\n\n")
            
            # Parse XML sections if present
            try:
                reasoning = gen['response'].split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                answer = gen['response'].split("<answer>\n")[1].split("\n</answer>")[0]
                f.write("PARSED SECTIONS:\n")
                f.write(f"Reasoning:\n{reasoning}\n")
                f.write(f"Answer:\n{answer}\n\n")
            except:
                f.write("ERROR: Could not parse XML sections\n\n")
            
            # Write reward scores
            f.write("REWARD SCORES:\n")
            for reward_name, reward_value in gen['scores'].items():
                f.write(f"{reward_name}: {reward_value:.4f}\n")
            # Total reward is sum of individual scores
            total_reward = sum(gen['scores'].values())
            f.write(f"Total reward: {total_reward:.4f}\n\n")
            f.write("-"*40 + "\n\n")


def write_contrastive_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write contrastive generation log data (PRO vs CON) to a text file.
    Updated to match the new grpo_contrastive_loss structure with single log_data containing both PRO and CON.

    Args:
        log_data: Dictionary containing both PRO and CON prompt and generation data with contrastive rewards
        log_file: Path to output log file
    """
    with open(log_file, 'w') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("CONTRASTIVE GRPO GENERATION LOG - PRO vs CON Cross-Comparison\n")
        f.write("=" * 80 + "\n\n")
        
        # Write base question
        f.write("###### BASE QUESTION #####\n\n")
        # Extract base question from chat template
        prompt_text = log_data['prompt']['text']
        
        # Extract the debate topic from the chat template
        if "Debate Topic:" in prompt_text:
            # Find the line with "Debate Topic:"
            lines = prompt_text.split('\n')
            for line in lines:
                if "Debate Topic:" in line:
                    # Extract just the topic part
                    topic_line = line.strip()
                    # Remove any position information
                    topic_line = topic_line.replace("Position: PRO", "").replace("Position: CON", "")
                    base_question = topic_line.strip()
                    break
            else:
                base_question = "Could not extract debate topic"
        else:
            base_question = prompt_text
            
        f.write(base_question + "\n\n")
        
        f.write("NOTE: In this contrastive setup, PRO completions are scored against CON completions\n")
        f.write("and CON completions are scored against PRO completions for cross-comparison.\n\n")

        # Write PRO section
        f.write("=" * 35 + " PRO STANCE (vs CON) " + "=" * 35 + "\n\n")
        f.write("###### PRO PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")

        # Write PRO generations with contrastive scoring
        if 'pro_generations' in log_data:
            for i, gen in enumerate(log_data['pro_generations'], 1):
                f.write(f"#### PRO GENERATION {i} (scored vs CON) ####\n\n")
                f.write("RESPONSE:\n")
                f.write(gen['response'] + "\n\n")
                
                # Parse XML sections if present
                try:
                    reasoning = gen['response'].split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    answer = gen['response'].split("<answer>\n")[1].split("\n</answer>")[0]
                    f.write("PARSED SECTIONS:\n")
                    f.write(f"Reasoning:\n{reasoning}\n")
                    f.write(f"Answer:\n{answer}\n\n")
                except:
                    f.write("ERROR: Could not parse XML sections\n\n")
                
                # Write contrastive reward scores (PRO vs CON comparison)
                f.write("CONTRASTIVE REWARD SCORES (PRO vs CON):\n")
                for reward_name, reward_value in gen['scores'].items():
                    if reward_name != 'total_reward':
                        f.write(f"{reward_name}: {reward_value:.4f}\n")
                f.write(f"Total contrastive reward: {gen['scores']['total_reward']:.4f}\n")
                f.write("\n")
                f.write("-"*40 + "\n\n")

        # Write CON section
        f.write("=" * 35 + " CON STANCE (vs PRO) " + "=" * 35 + "\n\n")
        f.write("###### CON PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")

        # Write CON generations with contrastive scoring
        if 'con_generations' in log_data:
            for i, gen in enumerate(log_data['con_generations'], 1):
                f.write(f"#### CON GENERATION {i} (scored vs PRO) ####\n\n")
                f.write("RESPONSE:\n")
                f.write(gen['response'] + "\n\n")
                
                # Parse XML sections if present
                try:
                    reasoning = gen['response'].split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                    answer = gen['response'].split("<answer>\n")[1].split("\n</answer>")[0]
                    f.write("PARSED SECTIONS:\n")
                    f.write(f"Reasoning:\n{reasoning}\n")
                    f.write(f"Answer:\n{answer}\n\n")
                except:
                    f.write("ERROR: Could not parse XML sections\n\n")
                
                # Write contrastive reward scores (CON vs PRO comparison)
                f.write("CONTRASTIVE REWARD SCORES (CON vs PRO):\n")
                for reward_name, reward_value in gen['scores'].items():
                    if reward_name != 'total_reward':
                        f.write(f"{reward_name}: {reward_value:.4f}\n")
                f.write(f"Total contrastive reward: {gen['scores']['total_reward']:.4f}\n")
                f.write("\n")
                f.write("-"*40 + "\n\n")

        # Write detailed contrastive summary statistics
        if 'pro_summary_stats' in log_data and 'con_summary_stats' in log_data:
            f.write("=" * 25 + " CONTRASTIVE ADVANTAGE STATISTICS " + "=" * 25 + "\n\n")
            
            f.write("PRO STANCE CONTRASTIVE STATS (PRO rewards vs CON):\n")
            pro_stats = log_data['pro_summary_stats']
            if 'advantages' in pro_stats:
                f.write(f"Final Advantages: {[f'{x:.4f}' for x in pro_stats['advantages']]}\n")
            if 'mean_rewards_per_group' in pro_stats:
                f.write(f"Mean Contrastive Rewards: {[f'{x:.4f}' for x in pro_stats['mean_rewards_per_group']]}\n")
            if 'std_rewards_per_group' in pro_stats:
                f.write(f"Std Contrastive Rewards: {[f'{x:.4f}' for x in pro_stats['std_rewards_per_group']]}\n")
            f.write("\n")
            
            f.write("CON STANCE CONTRASTIVE STATS (CON rewards vs PRO):\n")
            con_stats = log_data['con_summary_stats']
            if 'advantages' in con_stats:
                f.write(f"Final Advantages: {[f'{x:.4f}' for x in con_stats['advantages']]}\n")
            if 'mean_rewards_per_group' in con_stats:
                f.write(f"Mean Contrastive Rewards: {[f'{x:.4f}' for x in con_stats['mean_rewards_per_group']]}\n")
            if 'std_rewards_per_group' in con_stats:
                f.write(f"Std Contrastive Rewards: {[f'{x:.4f}' for x in con_stats['std_rewards_per_group']]}\n")
            f.write("\n")
            
            # Add explanation of contrastive advantage computation
            f.write("CONTRASTIVE ADVANTAGE COMPUTATION:\n")
            f.write("- PRO and CON completions are cross-compared for contrastive rewards\n")
            f.write("- Advantages = (rewards - mean_rewards) / (std_rewards + 1e-4)\n")
            f.write("- This creates a competitive setup where PRO and CON stances compete\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF CONTRASTIVE GENERATION LOG\n")
        f.write("=" * 80 + "\n")


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
