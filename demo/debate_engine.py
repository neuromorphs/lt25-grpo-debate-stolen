"""
Core debate engine for the Gradio demo.
Handles topic selection, argument generation, and judge evaluation.
"""

import re
import random
import sys
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path to import from main codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelManager, DemoConfig

@dataclass
class DebateRound:
    """Represents a single debate round."""
    topic: str
    stance: str  # "PRO" or "CON"
    user_name: str
    user_argument: str
    ai_argument: str
    judge_decision: str
    judge_explanation: str
    user_won: bool
    user_format_score: float
    ai_format_score: float

class DebateTopics:
    """Manages debate topics from the original codebase."""
    
    TOPICS = [
        "Video games should be taught as a school sport",
        "All schools should have mandatory cooking classes",
        "Homework should be replaced with project-based learning",
        "Every city should have a night market",
        "Movie theaters should have special quiet showings",
        "All schools should teach sign language",
        "Restaurants should offer smaller portion options",
        "Public spaces should have musical instruments",
        "All high schools should start after 9am",
        "Zoos should focus only on local wildlife",
        "Libraries should have recording studios",
        "Every workplace should allow pets",
        "Schools should teach financial literacy",
        "All restaurants should show calorie counts",
        "Museums should be open late on weekends",
        "Cities should have designated graffiti walls",
        "Schools should teach basic coding",
        "Grocery stores should have recipe stations",
        "All buildings should have rooftop gardens",
        "Cafes should have board game nights",
        "Libraries should offer virtual reality rooms",
        "Parks should have outdoor movie screens",
        "Schools should teach meditation",
        "Restaurants should compost food waste",
        "Cities should have more water fountains",
        "All schools should have maker spaces",
        "Gyms should offer childcare",
        "Libraries should loan art pieces",
        "Hotels should adopt shelter pets",
        "Schools should teach gardening",
        "Airports should have sleeping pods",
        "Malls should have indoor gardens",
        "Restaurants should grow their own herbs",
        "Cities should have free music venues",
        "Schools should teach public speaking",
        "Offices should have nap rooms",
        "Supermarkets should have tasting stations",
        "Libraries should have podcast studios",
        "Parks should have outdoor chess tables",
        "Schools should teach time management",
        "Restaurants should offer cooking classes",
        "Cities should have stargazing areas",
        "Beaches should have free sunscreen",
        "Schools should teach digital citizenship",
        "Hotels should have community spaces",
        "Parks should have fruit trees",
        "Libraries should offer language exchanges",
        "Theaters should have subtitle options",
        "Schools should teach environmental science",
        "Cities should have interactive art installations"
    ]
    
    @classmethod
    def get_random_topic(cls) -> str:
        """Get a random debate topic."""
        return random.choice(cls.TOPICS)
    
    @classmethod
    def get_random_stance(cls) -> str:
        """Get a random stance (PRO or CON)."""
        return random.choice(["PRO", "CON"])

class DebateEngine:
    """Core engine for running debate rounds."""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
        # Debate prompt adapted from rldatasets.py
        self.debate_prompt_template = """You will be given a debate topic and your position (PRO or CON). You should reason carefully about the position, then provide your argument.
It is very important that you put your reasoning process inside <reasoning> tags and your final argument inside <answer> tags, like this:

<reasoning>
Your step-by-step reasoning process here, considering key points and potential counterarguments
</reasoning>
<answer>
Your clear, concise 2-3 sentence debate position
</answer>

All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>."""

        # Judge prompt adapted from evaluator.py
        self.judge_prompt_template = """You are an impartial debate judge. You will be shown two debate responses on the same topic, arguing the same side (PRO or CON). Your task is to determine which argument was more compelling based on:
1. Logical reasoning and evidence
2. Clear structure and organization
3. Effective use of examples
4. Respectful tone
5. Addressing potential counterarguments

Topic: {topic}
Position: {stance}

Argument 1 (Human):
{human_argument}

Argument 2 (AI):
{ai_argument}

Which response was more compelling? First provide a brief explanation of your reasoning, then respond with EXACTLY one of these options:
- ARGUMENT_1_WINS (Human wins)
- ARGUMENT_2_WINS (AI wins)

YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED. End your response with either ARGUMENT_1_WINS or ARGUMENT_2_WINS."""

    def extract_xml_content(self, text: str, tag: str) -> str:
        """Extract content from XML tags."""
        try:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            if start_tag in text and end_tag in text:
                start_idx = text.find(start_tag) + len(start_tag)
                end_idx = text.find(end_tag)
                return text[start_idx:end_idx].strip()
            return text.strip()
        except:
            return text.strip()

    def calculate_format_score(self, text: str) -> float:
        """Calculate format score based on XML structure (adapted from evaluator.py)."""
        score = 0.0
        
        # Check for strict format (similar to _strict_format_reward)
        pattern = r"<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>"
        if re.search(pattern, text, re.DOTALL):
            score += 0.5
        else:
            # Check for relaxed format (similar to _soft_format_reward)
            pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
            if re.search(pattern, text, re.DOTALL):
                score += 0.3
        
        # XML tag counting (adapted from _xml_count_reward)
        if "<reasoning>" in text: score += 0.1
        if "</reasoning>" in text: score += 0.1  
        if "<answer>" in text: score += 0.1
        if "</answer>" in text: score += 0.1
        
        # Penalize content after final tag
        if "</answer>" in text:
            extra_content = text.split("</answer>")[-1].strip()
            score -= len(extra_content) * 0.001
        
        return max(0.0, min(1.0, score))

    def generate_ai_argument(self, topic: str, stance: str) -> str:
        """Generate an AI argument for the given topic and stance."""
        user_prompt = f"Debate Topic: {topic}\nPosition you have to defend: {stance}"
        
        try:
            response = self.model_manager.generate_argument(
                system_prompt=self.debate_prompt_template,
                user_prompt=user_prompt
            )
            return response
        except Exception as e:
            print(f"Error generating AI argument: {e}")
            # Fallback response
            return f"""<reasoning>
I need to defend the {stance} position on "{topic}". Let me think about the key benefits and address potential counterarguments.
</reasoning>
<answer>
As an AI, I believe the {stance} position on "{topic}" has merit based on practical benefits and societal value. This approach would lead to positive outcomes for communities and individuals.
</answer>"""

    def judge_arguments(self, topic: str, stance: str, human_argument: str, ai_argument: str) -> Tuple[bool, str]:
        """
        Judge two arguments and return (human_won, explanation).
        Returns True if human wins, False if AI wins.
        """
        judge_prompt = self.judge_prompt_template.format(
            topic=topic,
            stance=stance,
            human_argument=human_argument,
            ai_argument=ai_argument
        )
        
        try:
            response = self.model_manager.judge_debate(
                system_prompt="You are an impartial debate judge.",
                user_prompt=judge_prompt
            )
            
            # Extract decision
            human_won = "ARGUMENT_1_WINS" in response.upper()
            
            return human_won, response
            
        except Exception as e:
            print(f"Error in judging: {e}")
            # Fallback decision (random)
            human_won = random.choice([True, False])
            explanation = f"Error occurred during judging. Random decision: {'Human' if human_won else 'AI'} wins."
            return human_won, explanation

    def run_debate_round(self, user_name: str, topic: str, stance: str, user_argument: str) -> DebateRound:
        """Run a complete debate round."""
        
        # Generate AI argument
        ai_argument = self.generate_ai_argument(topic, stance)
        
        # Extract clean arguments from XML
        clean_user_argument = self.extract_xml_content(user_argument, "answer")
        clean_ai_argument = self.extract_xml_content(ai_argument, "answer")
        
        # Judge the debate
        user_won, judge_response = self.judge_arguments(
            topic, stance, clean_user_argument, clean_ai_argument
        )
        
        # Calculate format scores
        user_format_score = self.calculate_format_score(user_argument)
        ai_format_score = self.calculate_format_score(ai_argument)
        
        # Create debate round object
        return DebateRound(
            topic=topic,
            stance=stance,
            user_name=user_name,
            user_argument=user_argument,
            ai_argument=ai_argument,
            judge_decision="ARGUMENT_1_WINS" if user_won else "ARGUMENT_2_WINS",
            judge_explanation=judge_response,
            user_won=user_won,
            user_format_score=user_format_score,
            ai_format_score=ai_format_score
        )

    def format_argument_for_display(self, argument: str) -> Dict[str, str]:
        """Format an argument for display, extracting reasoning and answer."""
        reasoning = self.extract_xml_content(argument, "reasoning")
        answer = self.extract_xml_content(argument, "answer")
        
        return {
            "reasoning": reasoning if reasoning != argument else "No structured reasoning provided",
            "answer": answer if answer != argument else argument,
            "full_text": argument
        }

    def get_format_feedback(self, argument: str) -> str:
        """Provide feedback on argument formatting."""
        score = self.calculate_format_score(argument)
        
        if score >= 0.8:
            return "✅ Excellent formatting! Your argument follows the XML structure perfectly."
        elif score >= 0.5:
            return "👍 Good formatting! Your argument has the main XML tags."
        elif score >= 0.3:
            return "⚠️ Partial formatting. Try to include both <reasoning> and <answer> tags."
        else:
            return "❌ Please format your argument with <reasoning> and <answer> tags as shown in the example."

if __name__ == "__main__":
    # Test the debate engine
    from config import DemoConfig, ModelManager
    
    config = DemoConfig()
    model_manager = ModelManager(config)
    engine = DebateEngine(model_manager)
    
    # Test topic selection
    topic = DebateTopics.get_random_topic()
    stance = DebateTopics.get_random_stance()
    print(f"Topic: {topic}")
    print(f"Stance: {stance}")
    
    # Test AI argument generation
    ai_arg = engine.generate_ai_argument(topic, stance)
    print(f"AI Argument: {ai_arg}")
    
    # Test format scoring
    test_arg = """<reasoning>
    This is my reasoning about the topic.
    </reasoning>
    <answer>
    This is my final answer.
    </answer>"""
    
    score = engine.calculate_format_score(test_arg)
    print(f"Format score: {score}")
    feedback = engine.get_format_feedback(test_arg)
    print(f"Feedback: {feedback}")