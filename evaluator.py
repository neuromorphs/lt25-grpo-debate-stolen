"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
from model_interface import ModelInterface

from tqdm import tqdm

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str,
        gold_answer: Any = None,  # Add this parameter
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str, contrastive: bool = False, truth_comparison: bool = False) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        contrastive: Whether to use contrastive mode for debate evaluation
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    n = name.lower()
    if n == "gsm8k":
        return GSM8KEvaluator()
    elif n in ["debate", "debate_code"]:
        return DebateEvaluator(contrastive=contrastive, truth_comparison=truth_comparison)
    elif n == "ld":
        return LDEvaluator()
    elif n == "chopped":
        return ChoppedEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class DebateEvaluator(RewardEvaluator):
    """
    Reward evaluator for debate responses using two different approaches:
    1. For training: round-robin tournament scoring between generated responses
    2. For testing: head-to-head debates against the base model
    """
    
    def __init__(self, contrastive: bool = False, truth_comparison: bool = False) -> None:
        self.num_reward_functions = 4  # debate score + 3 format rewards
        self.contrastive = contrastive  # If True, no position is assigned, just the topic
        judge_constrastive_prompt = "opposite sides (PRO vs CON)" if contrastive else "the same side (PRO or CON)" 
        judge_constrastive_prompt = judge_constrastive_prompt if not truth_comparison else "for different answers"
        self.judge_prompt = f"""\n
        You are an impartial debate judge. You will be shown two debate responses on the same topic, arguing {judge_constrastive_prompt}. 
        Your task is to determine which argument was more compelling based on:
        1. Logical reasoning and evidence
        2. Clear structure and organization
        3. Effective use of examples
        4. Respectful tone
        5. Addressing potential counterarguments
        
        Topic: {{topic}}
        
        Argument 1 {{arg1_response}}
        
        Argument 2 {{arg2_response}}
        
        Which response was more compelling? Respond with EXACTLY one of these options:
        - ARGUMENT_1_WINS
        - ARGUMENT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        """
        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        # print(completions[0])
        # print(bool(re.match(pattern, completions[0], re.DOTALL)))  # Add re.DOTALL here
        matches = [bool(re.match(pattern, r, re.DOTALL)) for r in completions]  # And here
        return [0.5 if m else 0.0 for m in matches]

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
        # print(completions[0])
        # print(bool(re.match(pattern, completions[0], re.DOTALL)))  # Add re.DOTALL here
        matches = [bool(re.match(pattern, r, re.DOTALL)) for r in completions]  # And here
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Penalize if answer is less than 250 characters
            if "<answer>" in text and "</answer>" in text:
                answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
                if len(answer) < 250:
                    count -= (0.5 + (250 - len(answer)) / 125)  # Scale penalty by length
                elif len(answer) > 500:
                    count -= (0.5 + (len(answer) - 500) / 125)
                else:
                    count += 0.5
            
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    
    def _compute_train_rewards(
        self,
        input_prompt: Dict[str, str], # TODO: check why the signature needs 2 strings
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Semi-batched round-robin tournament scoring - batch inner loop only."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        topic = input_prompt['question']
        
        # Batch the inner loop for each completion
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            # Collect all comparisons for completion i
            judge_prompts = []
            comparison_indices = []
            
            for j in range(i + 1, num_completions):
                response1 = self._extract_xml_answer(train_model_completions[i])
                response2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    topic=topic,
                    arg1_response=response1,
                    arg2_response=response2
                )
                
                judge_prompts.append(judge_prompt)
                comparison_indices.append(j)
            
            # Skip if no comparisons for this completion
            if not judge_prompts:
                continue
            
            # Batch evaluate all comparisons for completion i
            if hasattr(all_models["judge_model"], 'generate_batch_prompts'):
                judge_responses = all_models["judge_model"].generate_batch_prompts(
                    system_prompt="You are an impartial debate judge.",
                    user_prompts=judge_prompts,
                    max_new_tokens=50,
                    temperature=0.1
                )
            else:
                # Fallback to sequential evaluation
                judge_responses = []
                for judge_prompt in judge_prompts:
                    response = all_models["judge_model"].generate(
                        system_prompt="You are an impartial debate judge.",
                        user_prompt=judge_prompt,
                        max_new_tokens=50,
                        temperature=0.1
                    )
                    judge_responses.append(response)
            
            # Process judge responses for completion i
            for j, judge_response in zip(comparison_indices, judge_responses):
                judge_response = judge_response.strip().upper()
                
                if "ARGUMENT_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                elif "ARGUMENT_2_WINS" in judge_response:
                    wins[j] += 1
                    losses[i] += 1
        
        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1  # number of matches per completion
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        debate_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Clean role prefixes that chat template might have added
        cleaned_completions = []
        for completion in train_model_completions:
            # Remove common role prefixes
            completion = completion.strip()
            if completion.startswith(('user\n', 'system\n', 'assistant\n')):
                completion = completion.split('\n', 1)[1] if '\n' in completion else completion
            cleaned_completions.append(completion)

        #print(f"Cleaned completions: {cleaned_completions[:5]}")  # Debugging output


        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(cleaned_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(cleaned_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(cleaned_completions), 
            device=device
        )
        # print(f"{strict_format=}, {soft_format=}, {xml_count=}")

        # Combine all rewards
        rewards_per_func[:, 0] = debate_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/debate_score": debate_scores.mean().item(),
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics
    
    def _compute_train_contrastive_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        first_model_completions: List[str],
        second_model_completions: List[str],
        device: str,
        pro_first: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        """Semi-batched round-robin tournament scoring - batch inner loop only."""
        num_completions = len(first_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros((num_completions, num_completions), device=device)
        
        topic = input_prompt['question']
        
        # Batch the inner loop for each completion
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            # Collect all comparisons for completion i
            judge_prompts = []
            comparison_indices = []
            
            for j in range(num_completions):
                response1 = self._extract_xml_answer(first_model_completions[i])
                response2 = self._extract_xml_answer(second_model_completions[j])
                
                if self.contrastive:
                    if pro_first:
                        response1 = 'defending the answer ' + input_prompt['pro_position'] + ':\n' + response1
                        response2 = 'defending the answer ' + input_prompt['con_position'] + ':\n' + response2
                    else:
                        response1 = 'defending the answer ' + input_prompt['con_position'] + ':\n' + response1
                        response2 = 'defending the answer ' + input_prompt['pro_position'] + ':\n' + response2
                else:
                    response1 = ':\n' + response1
                    response2 = ':\n' + response2

                judge_prompt = self.judge_prompt.format(
                    topic=topic,
                    arg1_response=response1, # gather the answer and add it
                    arg2_response=response2 
                )
                
                judge_prompts.append(judge_prompt)
                comparison_indices.append(j)
            
            # Skip if no comparisons for this completion
            if not judge_prompts:
                continue
            
            # Batch evaluate all comparisons for completion i
            if hasattr(all_models["judge_model"], 'generate_batch_prompts'):
                judge_responses = all_models["judge_model"].generate_batch_prompts(
                    system_prompt="You are an impartial debate judge.",
                    user_prompts=judge_prompts,
                    max_new_tokens=50,
                    temperature=0.1
                )
            else:
                # Fallback to sequential evaluation
                judge_responses = []
                for judge_prompt in judge_prompts:
                    response = all_models["judge_model"].generate(
                        system_prompt="You are an impartial debate judge.",
                        user_prompt=judge_prompt,
                        max_new_tokens=50,
                        temperature=0.1
                    )
                    judge_responses.append(response)
            
            # Process judge responses for completion i
            for j, judge_response in zip(comparison_indices, judge_responses):
                judge_response = judge_response.strip().upper()
                
                if "ARGUMENT_1_WINS" in judge_response:
                    wins[i, j] = 1
        # print(judge_prompt)
        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions  # number of matches per completion
        wins_first = wins.sum(dim=1)  # Wins for PRO model
        wins_second = wins.sum(dim=0)  # Wins for CON model
        first_win_rate = wins_first / total_matches
        second_win_rate = wins_second / total_matches
        first_debate_scores = first_win_rate #* 1.5  # Scale to desired range
        second_debate_scores = second_win_rate #* 1.5  # Scale to desired range

        # Clean role prefixes that chat template might have added
        cleaned_completions = []
        for completion in first_model_completions:
            # Remove common role prefixes
            completion = completion.strip()
            if completion.startswith(('user\n', 'system\n', 'assistant\n')):
                completion = completion.split('\n', 1)[1] if '\n' in completion else completion
            cleaned_completions.append(completion)

        #print(f"Cleaned completions: {cleaned_completions[:5]}")  # Debugging output


        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(cleaned_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(cleaned_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(cleaned_completions), 
            device=device
        )

        # print(strict_format)
        # print(soft_format)
        # print(xml_count)

        
        # Combine all rewards
        rewards_per_func[:, 0] = first_debate_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/debate_score": first_debate_scores.mean().item(),
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics, second_debate_scores


    def _compute_test_rewards(
        self,
        input_prompt: Dict[str, str],
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str,
        train_first: bool | None = None,
        train_pro: bool | None = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head debates against base model for testing."""
        num_debates = len(train_model_completions)
        # trained_spoke_first = torch.zeros(num_debates, device=device) # TODO: more granualar random choice
        rewards_per_func = torch.zeros(num_debates, self.num_reward_functions, device=device)
        wins = 0
        wins_defending_truth = 0
        wins_defending_false = 0
        get_truth_defending_truth = 0
        get_truth_defending_false = 0
        score = 0.0
        # total_defending_truth = 0
        # total_defending_false = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )

        topic = input_prompt['question']

        if train_first is None:
            raise ValueError("train_first must be specified")
        
        # Prepare all judge prompts at once
        judge_prompts = []
        for i in range(num_debates):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            compare_response = self._extract_xml_answer(compare_model_completions[i])
            
            if self.contrastive:
                # If contrastive, prepend PRO/CON based on position
                # check that train_first is not None
                if train_first: 
                    arg1 = 'defending the answer ' 
                    arg1 += input_prompt['pro_position'] + ':\n' if train_pro else input_prompt['con_position'] + ':\n'
                    arg1 += trained_response
                    arg2 = 'defending the answer ' 
                    arg2 += input_prompt['con_position'] + ':\n' if train_pro else input_prompt['pro_position'] + ':\n'
                    arg2 += compare_response
                else:
                    arg1 = 'defending the answer ' 
                    arg1 += input_prompt['con_position'] + ':\n' if train_pro else input_prompt['pro_position'] + ':\n'
                    arg1 += compare_response
                    arg2 = 'defending the answer ' 
                    arg2 += input_prompt['pro_position'] + ':\n' if train_pro else input_prompt['con_position'] + ':\n'
                    arg2 += trained_response
            else:
                if train_first is None:
                    arg1 = ':\n' +trained_response
                    arg2 = ':\n' +compare_response
                else:
                    arg1 = ':\n' +compare_response
                    arg2 = ':\n' +trained_response


            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                topic=topic,
                arg1_response=arg1,
                arg2_response=arg2
            )
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are an impartial debate judge.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()

            # persuasion metrics
            trained_win = False
            if train_first:
                if "ARGUMENT_1_WINS" in judge_response:
                    rewards_per_func[i, 0] = 1.0
                    wins += 1
                    trained_win = True
            else:
                if "ARGUMENT_2_WINS" in judge_response:
                    rewards_per_func[i, 0] = 1.0
                    wins += 1
                    trained_win = True
            # truth metrics
            if trained_win and train_pro:
                get_truth_defending_truth += 1
            if not(trained_win) and not train_pro:
                get_truth_defending_false += 1


            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_debates
        metrics = {
            "win_rate": win_rate,
            "truth_rate": (get_truth_defending_truth + wins_defending_truth) / num_debates,
            "truth_rate_defending_truth": get_truth_defending_truth / (num_debates/2),
            "truth_rate_defending_false": get_truth_defending_false / (num_debates/2),
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_debates": num_debates,
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False,
        train_first: Optional[bool] = None,
        train_pro: Optional[bool] = None,
        pro_first: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            # print("Computing test rewards using head-to-head debates")
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, 
                                                device, train_first=train_first, train_pro=train_pro)
        else:
            # print("Using semi-batched evaluation for training rewards")
            if self.contrastive: 
                return self._compute_train_contrastive_rewards(input_prompt, all_models, 
                                                                train_model_completions, 
                                                                compare_model_completions, 
                                                                device, pro_first=pro_first)
            else:
                return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "debate_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


class LDEvaluator(RewardEvaluator):
    """
    Reward evaluator for Larry David-style roasts using two different approaches:
    1. For training: round-robin tournament scoring between generated responses
    2. For testing: head-to-head comparisons against the base model
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # humor score + 3 format rewards
        self.judge_prompt = """You are a comedy judge. You will be shown two comedy bits in the style of Larry David making fun of something.

        Your only job is to pick which one is funnier. Two critical rules:

        1. Pick the funniest bit regardless of length or structure - a messy, extremely short bit could win if it's funnier
        2. AUTOMATIC LOSS: If a bit doesn't speak in first person AS Larry David (e.g. if it says "this is what Larry David might say" or describes what he would say)

        Subject: {topic}

        Bit 1:
        {arg1_response}

        Bit 2:
        {arg2_response}

        Which bit was funnier? Respond with EXACTLY one of these options:
        - BIT_1_WINS
        - BIT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        Remember: Any bit that doesn't speak AS Larry David in first person automatically loses. For bits that both speak as Larry, pick the funnier one regardless of length or structure."""

        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Penalize if answer is less than 250 characters
            if "<answer>" in text and "</answer>" in text:
                answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
                if len(answer) < 250:
                    count -= 0.25
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Round-robin tournament scoring for training + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        # Get humor scores using round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            for j in range(i + 1, num_completions):
                topic = input_prompt.split("Roast Subject: ")[1]
                response1 = self._extract_xml_answer(train_model_completions[i])
                response2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    topic=topic,
                    arg1_response=response1,
                    arg2_response=response2
                )
                
                # Get judge's decision using the interface
                judge_response = all_models["judge_model"].generate(
                    system_prompt="You are a comedy judge specializing in Larry David's style of humor.",
                    user_prompt=judge_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                ).strip().upper()
                
                if "BIT_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                else:
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        humor_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        # Combine all rewards
        rewards_per_func[:, 0] = humor_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/humor_score": humor_scores.mean().item(),
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics

    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head comparisons against base model for testing."""
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        wins = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        topic = prompt.split("Roast Subject: ")[1]
        
        for i in range(num_comparisons):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            
            # Get compare model's response
            compare_response = self._extract_xml_answer(compare_model_completions[i])     

            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                topic=topic,
                arg1_response=trained_response,
                arg2_response=compare_response
            )
            
            # Get judge's decision using the interface
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are a comedy judge specializing in Larry David's style of humor.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()
            
            if "BIT_1_WINS" in judge_response:
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1

            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_comparisons
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_comparisons": num_comparisons,
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False,
        use_batched_eval: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "humor_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


class ChoppedEvaluator(RewardEvaluator):
    """
    Reward evaluator for Chopped-style recipe generation using two different approaches:
    1. For training: round-robin tournament scoring between generated recipes
    2. For testing: head-to-head comparisons against the base model
    """
    
    def __init__(self):
        self.num_reward_functions = 4  # recipe score + 3 format rewards
        self.judge_prompt = """You are a Chopped judge evaluating two recipes that use the same mystery basket ingredients.
        Your task is to determine which recipe would taste better based on:
        1. Flavor balance and harmony
        2. Creative use of mystery ingredients
        3. Technical execution and timing
        4. Overall appeal and presentation
        5. Practicality and replicability

        Mystery Basket:
        {basket}

        Recipe 1:
        {recipe1}

        Recipe 2:
        {recipe2}

        Which recipe would taste better? Respond with EXACTLY one of these options:
        - RECIPE_1_WINS
        - RECIPE_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        Focus purely on which recipe would taste better and make better use of the mystery ingredients.
        """
        
    def _extract_xml_answer(self, text: str) -> str:
        """Extract the answer portion from XML tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except:
            return text  # Fallback if format is incorrect
   
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<reasoning>" in text: count += 0.125
            if "</reasoning>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            
            # Penalize if answer is less than 250 characters
            if "<answer>" in text and "</answer>" in text:
                answer = text.split("<answer>")[-1].split("</answer>")[0].strip()
                if len(answer) < 250:
                    count -= 0.25
            
            # Only penalize actual content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
            
        return [count_xml(r) for r in completions]
        
    def _compute_train_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Round-robin tournament scoring for training + format rewards."""
        num_completions = len(train_model_completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Track wins/losses for each completion
        wins = torch.zeros(num_completions, device=device)
        losses = torch.zeros(num_completions, device=device)
        
        # Get recipe scores using round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating completions", leave=False):
            for j in range(i + 1, num_completions):
                basket = input_prompt.split("Mystery Basket:\n")[1].strip()
                recipe1 = self._extract_xml_answer(train_model_completions[i])
                recipe2 = self._extract_xml_answer(train_model_completions[j])
                
                judge_prompt = self.judge_prompt.format(
                    basket=basket,
                    recipe1=recipe1,
                    recipe2=recipe2
                )
                
                # Get judge's decision using the interface
                judge_response = all_models["judge_model"].generate(
                    system_prompt="You are a Chopped judge evaluating recipes.",
                    user_prompt=judge_prompt,
                    max_new_tokens=50,
                    temperature=0.1
                ).strip().upper()
                
                if "RECIPE_1_WINS" in judge_response:
                    wins[i] += 1
                    losses[j] += 1
                else:
                    wins[j] += 1
                    losses[i] += 1

        # Calculate normalized scores (-1.5 to 1.5 range)
        total_matches = num_completions - 1
        win_rate = wins / total_matches
        loss_rate = losses / total_matches
        recipe_scores = (win_rate - loss_rate) * 1.5  # Scale to desired range

        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        # Combine all rewards
        rewards_per_func[:, 0] = recipe_scores
        rewards_per_func[:, 1] = strict_format
        rewards_per_func[:, 2] = soft_format
        rewards_per_func[:, 3] = xml_count
        
        metrics = {
            "rewards/recipe_score": recipe_scores.mean().item(),
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(),
            "rewards/xml_count": xml_count.mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item()
        }
        
        return rewards_per_func, metrics

    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: List[str],
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Head-to-head comparisons against base model for testing."""
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        wins = 0
        
        # Get format rewards
        strict_format = torch.tensor(
            self._strict_format_reward(train_model_completions), 
            device=device
        )
        soft_format = torch.tensor(
            self._soft_format_reward(train_model_completions), 
            device=device
        )
        xml_count = torch.tensor(
            self._xml_count_reward(train_model_completions), 
            device=device
        )
        
        basket = prompt.split("Mystery Basket:\n")[1].strip()
        
        for i in range(num_comparisons):
            # Get trained model's response
            trained_response = self._extract_xml_answer(train_model_completions[i])
            
            # Get compare model's response
            compare_response = self._extract_xml_answer(compare_model_completions[i])     

            # Format judge prompt
            judge_prompt = self.judge_prompt.format(
                basket=basket,
                recipe1=trained_response,
                recipe2=compare_response
            )
            
            # Get judge's decision using the interface
            judge_response = all_models["judge_model"].generate(
                system_prompt="You are a Chopped judge evaluating recipes.",
                user_prompt=judge_prompt,
                max_new_tokens=50,
                temperature=0.1
            ).strip().upper()
            
            if "RECIPE_1_WINS" in judge_response:
                score = 1.0
                rewards_per_func[i, 0] = score
                wins += 1

            # Add format rewards
            rewards_per_func[i, 1] = strict_format[i]
            rewards_per_func[i, 2] = soft_format[i]
            rewards_per_func[i, 3] = xml_count[i]

        win_rate = wins / num_comparisons
        metrics = {
            "win_rate": win_rate,
            "reward": rewards_per_func.mean().item(),
            "num_wins": wins,
            "num_comparisons": num_comparisons,
            "rewards/strict_format": strict_format.mean().item(),
            "rewards/soft_format": soft_format.mean().item(), 
            "rewards/xml_count": xml_count.mean().item()
        }
        
        return rewards_per_func, metrics

    def compute_rewards(
        self,
        input_prompt: str,
        all_models: Dict[str, Any],
        train_model_completions: List[str],
        compare_model_completions: Optional[List[str]] = None,
        device: str = "cuda",
        is_test: bool = False,
        use_batched_eval: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards - different behavior for training vs testing."""
        if is_test:
            return self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            return self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
            
    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """Convert raw reward scores to a labeled dictionary."""
        return {
            "recipe_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item()
        }


# ──────────────────────────────────────────────────────────────────────────────
#  GSM-8K  (math-reasoning)  evaluator
# ──────────────────────────────────────────────────────────────────────────────
class GSM8KEvaluator(RewardEvaluator):
    """
    Reward evaluator for grade-school math problems (GSM-8K).

    Main reward: pair-wise ‘math correctness’ decided by a judge LLM.
    + three format rewards identical to the debate evaluator.
    """

    def __init__(self) -> None:
        # math score + strict + soft + xml_count
        self.num_reward_functions = 4

        self.judge_prompt = """You are an expert elementary-math teacher.
You will be shown ONE word problem and TWO candidate solutions, each wrapped
in <reasoning> … </reasoning> and <answer> … </answer> tags.

Evaluate each solution on:
1. Whether the arithmetic steps are valid.
2. Whether the final numeric answer follows from the steps.
3. Clarity and conciseness of the reasoning.

Rules for deciding the winner:
- If only one solution is fully correct, that solution wins.
- If both are correct, choose the clearer / more rigorous explanation.
- If both are wrong, choose the one that comes *closer* (fewer errors or
  a smaller numeric mistake).

Problem:
{question}

Solution 1:
{sol1}

Solution 2:
{sol2}

Respond with **EXACTLY** one of:
- SOLUTION_1_WINS
- SOLUTION_2_WINS

YOU MUST CHOOSE A WINNER – ties are not allowed."""

    # ── helpers ────────────────────────────────────────────────────────────
    def _extract_xml_answer(self, text: str) -> str:
        try:
            return text.split("<answer>")[-1].split("</answer>")[0].strip()
        except Exception:
            return text.strip()

    def _strict_format_reward(self, completions) -> list[float]:
        pat = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$"
        return [0.5 if re.match(pat, c, re.DOTALL) else 0.0 for c in completions]

    def _soft_format_reward(self, completions) -> list[float]:
        pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        return [0.5 if re.match(pat, c, re.DOTALL) else 0.0 for c in completions]

    def _xml_count_reward(self, completions) -> list[float]:
        def score(txt: str) -> float:
            s = 0.0
            for tag in ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]:
                if tag in txt:
                    s += 0.125
            if "</answer>" in txt:
                s -= 0.001 * len(txt.split("</answer>")[-1].strip())
            return s
        return [score(c) for c in completions]

    # ── core round-robin reward used for *training* ────────────────────────
    def _compute_train_rewards(
        self,
        prompt: str,
        all_models: dict[str, Any],
        completions: list[str],
        device: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        n = len(completions)
        r = torch.zeros(n, self.num_reward_functions, device=device)
        wins, losses = torch.zeros(n, device=device), torch.zeros(n, device=device)

        # grab the original question (prompt ends with it)
        question = prompt.split("Question:\n", 1)[-1]

        # enumerate all pairs
        pairs, judge_prompts = [], []
        for i in range(n):
            for j in range(i + 1, n):
                p = self.judge_prompt.format(
                    question=question,
                    sol1=completions[i],
                    sol2=completions[j],
                )
                pairs.append((i, j))
                judge_prompts.append(p)

        # batch or sequential – whatever interface supports
        if hasattr(all_models["judge_model"], "generate_batch_prompts"):
            judge_resps = all_models["judge_model"].generate_batch_prompts(
                system_prompt="You are an expert elementary-math teacher.",
                user_prompts=judge_prompts,
                max_new_tokens=30,
                temperature=0.1,
            )
        else:
            judge_resps = [
                all_models["judge_model"].generate(
                    system_prompt="You are an expert elementary-math teacher.",
                    user_prompt=jp,
                    max_new_tokens=30,
                    temperature=0.1,
                )
                for jp in judge_prompts
            ]

        # update wins / losses
        for (i, j), resp in zip(pairs, judge_resps):
            resp = resp.strip().upper()
            if "SOLUTION_1_WINS" in resp:
                wins[i] += 1
                losses[j] += 1
            else:
                wins[j] += 1
                losses[i] += 1

        total_matches = n - 1
        math_scores = (wins / total_matches - losses / total_matches) * 1.5

        strict = torch.tensor(self._strict_format_reward(completions), device=device)
        soft   = torch.tensor(self._soft_format_reward(completions),   device=device)
        xml    = torch.tensor(self._xml_count_reward(completions),    device=device)

        r[:, 0], r[:, 1], r[:, 2], r[:, 3] = math_scores, strict, soft, xml

        metrics = {
            "rewards/math_score": math_scores.mean().item(),
            "rewards/strict_format": strict.mean().item(),
            "rewards/soft_format": soft.mean().item(),
            "rewards/xml_count": xml.mean().item(),
            "reward": r.sum(dim=1).mean().item(),
        }
        return r, metrics

    # # For now we reuse training logic for test; customise if needed later.
    # def _compute_test_rewards(
    #     self, prompt, all_models, completions, compare_completions, device
    # ):
    #     return self._compute_train_rewards(prompt, all_models, completions, device)

    # public dispatcher ------------------------------------------------------
    def compute_rewards(
        self,
        input_prompt: str,
        all_models: dict[str, Any],
        train_model_completions: list[str],
        compare_model_completions=None,
        device: str = "cuda",
        is_test: bool = False,
        gold_answer: str = None,  # Add parameter
        **kwargs,
    ):
        if is_test:
            return self._compute_test_rewards(
                input_prompt,
                all_models,
                train_model_completions,
                compare_model_completions,
                device,
                gold_answer=gold_answer,  # Pass it through
            )
        # Training path unchanged - still uses judge
        return self._compute_train_rewards(
            input_prompt, all_models, train_model_completions, device
        )
    # ───────────────────────────────────────────────────────────────
    #  test-time head-to-head vs a baseline model
    # ───────────────────────────────────────────────────────────────
    def _compute_test_rewards(
        self,
        prompt: str,
        all_models: dict[str, Any],
        trained_completions: list[str],
        baseline_completions: list[str],
        device: str,
        gold_answer: str = None,  # Add gold answer parameter
    ):
        """Evaluate based on correctness against gold answer during testing."""
        n = len(trained_completions)
        rewards = torch.zeros(n, self.num_reward_functions, device=device)
        wins = 0

        if gold_answer is None:
            print("ANSWER NOT PROVIDED")
            # Fall back to judge-based evaluation if no gold answer
            return self._compute_test_rewards_judge_based(
                prompt, all_models, trained_completions, baseline_completions, device
            )

        # Extract answers and check correctness
        for i in range(n):
            # Extract answer from trained model's completion
            trained_answer = self._extract_xml_answer(trained_completions[i]).strip()

            # Check if the trained model got it correct
            #print("USING ACCURACY BASED CHECKING MECHANISM")
            trained_correct = self._check_answer_correctness(trained_answer, gold_answer)

            # Extract answer from baseline model's completion  
            baseline_answer = self._extract_xml_answer(baseline_completions[i]).strip()
            baseline_correct = self._check_answer_correctness(baseline_answer, gold_answer)

            # Assign win based on correctness
            if trained_correct and not baseline_correct:
                wins += 1
                rewards[i, 0] = 1.0
            elif trained_correct and baseline_correct:
                # Both correct - could use judge as tiebreaker or give partial credit
                rewards[i, 0] = 0.5  # Or use judge to break tie
            # If trained is wrong, it gets 0 (default)

            # XML-format rewards remain the same
            rewards[i, 1] = self._strict_format_reward([trained_completions[i]])[0]
            rewards[i, 2] = self._soft_format_reward([trained_completions[i]])[0]
            rewards[i, 3] = self._xml_count_reward([trained_completions[i]])[0]

        win_rate = wins / n
        metrics = {
            "num_wins": wins,
            "num_comparisons": n,
            "win_rate": win_rate,
            "rewards/strict_format": rewards[:, 1].mean().item(),
            "rewards/soft_format": rewards[:, 2].mean().item(),
            "rewards/xml_count": rewards[:, 3].mean().item(),
            "reward": rewards.mean().item(),
        }
        return rewards, metrics

    def _check_answer_correctness(self, predicted: str, gold: str) -> bool:
        """Check if predicted answer matches gold answer."""
        # Normalize both answers
        predicted = predicted.strip().lower()
        gold = gold.strip().lower()
        
        # Remove common formatting
        for char in [",", "$", "%"]:
            predicted = predicted.replace(char, "")
            gold = gold.replace(char, "")
        
        # Direct match
        if predicted == gold:
            return True
        
        # Try numeric comparison
        try:
            pred_num = float(predicted)
            gold_num = float(gold)
            return abs(pred_num - gold_num) < 1e-6
        except:
            return False

    # Keep the old judge-based method as fallback
    def _compute_test_rewards_judge_based(
            self,
            prompt: str,
            all_models: dict[str, Any],
            trained_completions: list[str],
            baseline_completions: list[str],
            device: str,
        ):
            """Judge each trained-vs-baseline pair once and add win-metrics keys."""
            n = len(trained_completions)
            rewards = torch.zeros(n, self.num_reward_functions, device=device)
            wins = 0

            question = prompt.split("Question:\n", 1)[-1]

            for i in range(n):
                judge_prompt = self.judge_prompt.format(
                    question=question,
                    sol1=trained_completions[i],
                    sol2=baseline_completions[i],
                )

                decision = all_models["judge_model"].generate(
                    system_prompt="You are an expert elementary-math teacher.",
                    user_prompt=judge_prompt,
                    max_new_tokens=30,
                    temperature=0.1,
                ).strip().upper()

                if "SOLUTION_1_WINS" in decision:
                    wins += 1
                    rewards[i, 0] = 1.0   # +1 for a win

                # XML-format rewards for this completion
                rewards[i, 1] = self._strict_format_reward([trained_completions[i]])[0]
                rewards[i, 2] = self._soft_format_reward  ([trained_completions[i]])[0]
                rewards[i, 3] = self._xml_count_reward   ([trained_completions[i]])[0]

            win_rate = wins / n
            metrics = {
                "num_wins": wins,
                "num_comparisons": n,
                "win_rate": win_rate,
                "rewards/strict_format": rewards[:, 1].mean().item(),
                "rewards/soft_format":  rewards[:, 2].mean().item(),
                "rewards/xml_count":    rewards[:, 3].mean().item(),
                "reward": rewards.mean().item(),
            }
            return rewards, metrics

    def get_reward_breakdown(self, rewards: torch.Tensor) -> dict[str, float]:
        return {
            "math_score": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item(),
        }
