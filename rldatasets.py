"""
Hold all data sets 

"""

import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod
from typing import Tuple, Any
import pandas as pd


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This class defines the interface that all dataset loaders should implement.
    Specific dataset loaders should inherit from this class and implement the
    required methods.
    
    Attributes:
        random (bool): If True, returns items randomly; if False, returns sequentially
        current_index (int): Current position for sequential access
    """
    
    def __init__(self, random: bool = False) -> None:
        self.random = random
        self.current_index = 0
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        pass
        
    @abstractmethod
    def __iter__(self) -> 'DataLoader':
        """Return self as iterator."""
        return self
        
    @abstractmethod
    def __next__(self) -> Any:
        """Return the next item(s) in the dataset."""
        pass


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()



SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""




class DebateDataLoader(DataLoader):
    """
    A loader class that provides iteration over debate topics.
    
    This class implements both sequential and random access to debate topics through
    standard Python iterator protocols. For each topic, it randomly assigns PRO or CON
    position to create debate scenarios.

    Attributes:
        topics (list[str]): List of debate topics (here 43 topics for training and 7 for testing)
        pre_prompt (str): Instructional prompt for the model
        system_prompt (str): System prompt to guide the model's responses
    """
    
    def __init__(self, topics: list[str], random: bool = False, contrastive: bool = False) -> None:
        super().__init__(random)
        self.topics = topics
        self.contrastive = contrastive  # If True, no position is assigned, just the topic
        self.pre_prompt = """You will be given a debate topic and your position (PRO or CON). You should reason carefully about the position, then provide your argument.
            It is very important that you put your reasoning process inside <reasoning> tags and your final argument inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, considering key points and potential counterarguments
            </reasoning>
            <answer>
            Your clear, concise 2-3 sentence debate position
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT  # Using the same system prompt as GSM8K
            
    def __len__(self) -> int:
        return len(self.topics)
        
    def __iter__(self) -> 'DebateDataLoader':
        return self
        
    def __next__(self) -> tuple[str, str]:
        if self.current_index >= len(self.topics):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.topics) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        topic = self.topics[idx]
        formatted_question = f"Debate Topic: {topic}\n"
        if not self.contrastive: 
            position = random.choice(["PRO", "CON"])
            formatted_question += f"Position: {position}\n"
        
        # The "answer" in this case is the position, which is needed for evaluation
        return formatted_question

    def reset(self):
        self.current_index = 0



def build_debate_dataloaders(debug: bool = False) -> Tuple[DebateDataLoader, DebateDataLoader]:
    # Define debate topics - non-controversial but engaging topics
    topics = [
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
    # Split into train/test sets (85/15 split)
    total_topics = len(topics)

    test_size = 1 if debug else 6 # int(total_topics * 0.15)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_topics), test_size)
    test_indices_set = set(test_indices)
    
    # Split topics
    train_topics = [t for i, t in enumerate(topics) if i not in test_indices_set]
    test_topics = [topics[i] for i in test_indices]
    
    # Create data loaders
    trainloader = DebateDataLoader(train_topics, random=True)
    testloader = DebateDataLoader(test_topics, random=False)
    
    return trainloader, testloader

def build_debate_contrastive_dataloaders(debug: bool = False) -> Tuple[DebateDataLoader, DebateDataLoader]:
    # Define debate topics - non-controversial but engaging topics
    topics = [
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
    # Split into train/test sets (85/15 split)
    total_topics = len(topics)
    test_size = 1 if debug else 6 # int(total_topics * 0.15)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_topics), test_size)
    test_indices_set = set(test_indices)
    
    # Split topics
    train_topics = [t for i, t in enumerate(topics) if i not in test_indices_set]
    test_topics = [topics[i] for i in test_indices]
    
    # Create data loaders
    trainloader = DebateDataLoader(train_topics, random=True, contrastive=True)
    testloader = DebateDataLoader(test_topics, random=False, contrastive=True)
    
    return trainloader, testloader


class LDDataLoader(DataLoader):
    """
    A loader class that provides iteration over subjects for Larry David-style roasts.
    
    This class implements both sequential and random access to roast subjects through
    standard Python iterator protocols. For each subject, it generates a prompt for a
    Larry David-style roast.
    """
    
    def __init__(self, subjects: list[str], random: bool = False) -> None:
        super().__init__(random)
        self.subjects = subjects
        self.pre_prompt = """You will be given a subject to mock in Larry David's style from Curb Your Enthusiasm. 
            You should first reason about how to make it funny and creative, then provide your mocking.
            It is very important that you put your reasoning process inside <reasoning> tags and your actual roast inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, thinking about what makes Larry David's humor unique and how to apply it to this subject
            </reasoning>
            <answer>
            [Speak in first person as Larry David himself]
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT
            
    def __len__(self) -> int:
        return len(self.subjects)
        
    def __iter__(self) -> 'LDDataLoader':
        return self
        
    def __next__(self) -> str:
        if self.current_index >= len(self.subjects):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.subjects) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        subject = self.subjects[idx]
        
        # Format the question to include the subject
        formatted_question = f"Roast Subject: {subject}"
        
        return formatted_question

    def reset(self):
        self.current_index = 0


def build_ld_dataloaders() -> Tuple[LDDataLoader, LDDataLoader]:
    # Define roast subjects - mix of celebrities, social norms, and everyday situations
    subjects = [
        "People who clap when planes land",
        "Selfie sticks",
        "People who talk in movie theaters",
        "Social media influencers",
        "People who wear pajamas to the airport",
        "Reality TV shows",
        "People who take phone calls on speaker in public",
        "Food delivery apps",
        "People who post their entire lives on social media",
        "Modern art",
        "People who use emojis in work emails",
        "Smartphone addiction",
        "People who take photos of their food",
        "Online dating",
        "People who wear masks while driving alone",
        "Social media challenges",
        "People who use voice messages",
        "Modern architecture",
        "People who clap at the end of movies",
        "Food trucks",
        "People who use hashtags in regular conversation",
        "Modern music",
        "People who take selfies at funerals",
        "Social media filters",
        "People who use speakerphone in restaurants",
        "Modern fashion",
        "People who post workout videos",
        "Food delivery fees",
        "People who use voice assistants in public",
        "Modern technology",
        "People who take photos of everything",
        "Social media algorithms",
        "People who use emojis in texts",
        "Modern entertainment",
        "People who post their entire lives",
        "Food delivery services",
        "People who use voice messages",
        "Modern society",
        "People who take photos of everything",
        "Social media culture",
        "People who use emojis in emails",
        "Modern lifestyle",
        "People who post their entire lives",
        "Food delivery apps",
        "People who use voice assistants",
        "Modern culture",
        "People who take photos of everything",
        "Social media trends",
        "People who use emojis in texts",
        "Modern world"
    ]
    
    # Split into train/test sets (85/15 split)
    total_subjects = len(subjects)
    test_size = int(total_subjects * 0.15)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_subjects), test_size)
    test_indices_set = set(test_indices)
    
    # Split subjects
    train_subjects = [s for i, s in enumerate(subjects) if i not in test_indices_set]
    test_subjects = [subjects[i] for i in test_indices]
    
    # Create data loaders
    trainloader = LDDataLoader(train_subjects, random=True)
    testloader = LDDataLoader(test_subjects, random=False)
    
    return trainloader, testloader


class CodeComprehensionDataLoader(DataLoader):
    """
    A loader class that provides iteration over string manipulation tasks from the CodeComprehension dataset.
    
    This class implements both sequential and random access to code comprehension problems through
    standard Python iterator protocols. It focuses specifically on string manipulation tasks,
    filtering the dataset to include only problems that involve string operations.
    """
    
    def __init__(self, random: bool = False, max_samples: int = None) -> None:
        super().__init__(random)
        self.max_samples = max_samples
        self.string_examples = []
        self._load_string_manipulation_examples()
        self.pre_prompt = """You will be given a Python code snippet that performs string manipulation operations.
            You should trace through the code execution step by step, then determine the final result.
            It is very important that you put your reasoning process inside <reasoning> tags and your final answer inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, tracing through each line of code:
            1. Initial variable values
            2. Each operation and its result
            3. Final variable state
            </reasoning>
            <answer>
            The final result/value that gets printed or assigned to the result variable
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! 
            Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT
        
    def _load_string_manipulation_examples(self) -> None:
        """Load and filter examples that involve string manipulation."""
        from datasets import load_dataset
        
        dataset = load_dataset('imbue/code-comprehension')
        string_functions = [
            'str(', '.islower()', '.isdigit()', '.upper()', '.lower()', '.strip()', 
            '.replace(', '.split(', '.join(', 'len(', '.swapcase()', '.isalpha()', 
            '.startswith(', '.endswith(', '.find(', '.count(', '.capitalize()', 
            '.title(', '.isnumeric(', '.isalnum(', '.lstrip(', '.rstrip()'
        ]
        
        max_check = self.max_samples * 2 if self.max_samples else len(dataset['train'])
        
        for i in range(min(max_check, len(dataset['train']))):
            example = dataset['train'][i]
            question = example['question']
            
            # Check if this involves string manipulation
            if any(func in question for func in string_functions):
                self.string_examples.append(example)
                if self.max_samples and len(self.string_examples) >= self.max_samples:
                    break
                    
    def __len__(self) -> int:
        return len(self.string_examples)
        
    def __iter__(self) -> 'CodeComprehensionDataLoader':
        return self
        
    def __next__(self) -> tuple[str, list[str], int]:
        if self.current_index >= len(self.string_examples):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.string_examples) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        example = self.string_examples[idx]
        
        # Format the question - the dataset already includes the question format
        formatted_question = example['question']
        choices = example['choices']
        correct_answer = example['correct_answer']
        
        # Find the index of the correct answer in choices
        try:
            correct_idx = choices.index(correct_answer)
        except ValueError:
            # If exact match not found, set to 0 as fallback
            correct_idx = 0
        
        return formatted_question, choices, correct_idx

    def reset(self):
        self.current_index = 0
        
    def get_choices_and_answer(self, idx: int = None) -> tuple[list[str], str]:
        """Get the choices and correct answer for a specific example."""
        if idx is None:
            idx = self.current_index - 1 if self.current_index > 0 else 0
        
        if idx >= len(self.string_examples):
            raise IndexError("Index out of range")
            
        example = self.string_examples[idx]
        return example['choices'], example['correct_answer']


def build_codecomprehension_dataloaders(max_train_samples: int = 1000, max_test_samples: int = 200) -> Tuple[CodeComprehensionDataLoader, CodeComprehensionDataLoader]:
    """
    Build train and test data loaders for CodeComprehension string manipulation tasks.
    
    Args:
        max_train_samples: Maximum number of training samples to load
        max_test_samples: Maximum number of test samples to load
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from datasets import load_dataset
    
    dataset = load_dataset('imbue/code-comprehension')
    string_functions = [
        'str(', '.islower()', '.isdigit()', '.upper()', '.lower()', '.strip()', 
        '.replace(', '.split(', '.join(', 'len(', '.swapcase()', '.isalpha()', 
        '.startswith(', '.endswith(', '.find(', '.count(', '.capitalize()', 
        '.title(', '.isnumeric(', '.isalnum(', '.lstrip(', '.rstrip()'
    ]
    
    # Filter for string manipulation examples
    string_examples = []
    for i in range(len(dataset['train'])):
        example = dataset['train'][i]
        question = example['question']
        
        if any(func in question for func in string_functions):
            string_examples.append(example)
            
    # Split into train/test (80/20 split)
    total_examples = len(string_examples)
    test_size = min(max_test_samples, int(total_examples * 0.2))
    train_size = min(max_train_samples, total_examples - test_size)
    
    # Generate random indices for test set
    test_indices = random.sample(range(total_examples), test_size)
    test_indices_set = set(test_indices)
    
    # Split examples
    train_examples = [example for i, example in enumerate(string_examples) if i not in test_indices_set][:train_size]
    test_examples = [string_examples[i] for i in test_indices]
    
    # Create custom loaders with pre-filtered data
    trainloader = CodeComprehensionDataLoader(random=True)
    trainloader.string_examples = train_examples
    
    testloader = CodeComprehensionDataLoader(random=False)
    testloader.string_examples = test_examples
    
    return trainloader, testloader


class ChoppedDataLoader(DataLoader):
    """
    A loader class that provides iteration over mystery baskets for Chopped-style recipe generation.
    
    This class implements both sequential and random access to mystery baskets through
    standard Python iterator protocols. For each basket, it generates a prompt for creating
    a creative recipe using all the mystery ingredients.
    """
    
    def __init__(self, baskets: list[list[str]], random: bool = False) -> None:
        super().__init__(random)
        self.baskets = baskets
        self.pre_prompt = """You will be given 4 mystery ingredients from a Chopped-style cooking show. 
            You should first reason about how to create the best possible dish using these ingredients,
            then provide a detailed recipe. It is very important that you put your reasoning process inside 
            <reasoning> tags and your recipe inside <answer> tags, like this:

            <reasoning>
            Your step-by-step reasoning process here, thinking about:
            1. How to balance flavors and textures
            2. Cooking techniques that would work best
            3. How to make the dish cohesive despite unusual combinations
            4. Additional ingredients needed and why
            5. Timing and execution strategy
            </reasoning>
            <answer>
            A detailed recipe that:
            - Uses all mystery ingredients
            - Can be made in 40 minutes
            - Includes a list of additional ingredients needed
            - Has clear step-by-step instructions
            - Includes timing estimates
            </answer>

            All of your returned text should either be in the <reasoning> or <answer> tags - no text outside! 
            Start each response by immediately starting with <reasoning>. 
            """
        self.system_prompt = SYSTEM_PROMPT
            
    def __len__(self) -> int:
        return len(self.baskets)
        
    def __iter__(self) -> 'ChoppedDataLoader':
        return self
        
    def __next__(self) -> str:
        if self.current_index >= len(self.baskets):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.baskets) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
            
        basket = self.baskets[idx]
        
        # Format the question to include the mystery basket
        formatted_question = f"Mystery Basket:\n" + "\n".join(f"- {ingredient}" for ingredient in basket)
        
        return formatted_question

    def reset(self):
        self.current_index = 0


def build_chopped_dataloaders() -> Tuple[ChoppedDataLoader, ChoppedDataLoader]:
    # Define base ingredients for training - mix of common and unusual ingredients
    train_ingredients = [
        "Gummy bears",
        "Sardines",
        "Popcorn",
        "Hot sauce",
        "Marshmallows",
        "Kimchi",
        "Coffee grounds",
        "Coconut water",
        "Bacon",
        "Blue cheese",
        "Ginger root",
        "Lemon grass",
        "Mint leaves",
        "Pomegranate seeds",
        "Soy sauce",
        "Tofu",
        "Wasabi",
        "Yogurt",
        "Zucchini",
        "Quinoa"
    ]
    
    # Define separate ingredients for testing
    test_ingredients = [
        "Durian",
        "Seaweed",
        "Crickets",
        "Black garlic",
        "Goat cheese",
        "Mango",
        "Curry powder",
        "Pickled ginger",
        "Coconut milk",
        "Tempeh",
        "Star anise",
        "Lime leaves",
        "Cilantro",
        "Pomegranate molasses",
        "Fish sauce",
        "Miso paste",
        "Sesame oil",
        "Kombu",
        "Shiitake mushrooms",
        "Rice vinegar"
    ]
    
    # Generate training baskets (80 baskets)
    train_baskets = []
    for _ in range(80):
        basket = random.sample(train_ingredients, 4)
        train_baskets.append(basket)
    
    # Generate test baskets (20 baskets)
    test_baskets = []
    for _ in range(10):
        basket = random.sample(test_ingredients, 4)
        test_baskets.append(basket)
    
    # Create data loaders
    trainloader = ChoppedDataLoader(train_baskets, random=True)
    testloader = ChoppedDataLoader(test_baskets, random=False)
    
    return trainloader, testloader


def get_dataloaders(dataset_name: str, contrastive: bool = False, max_train_samples: int = None, max_test_samples: int = None, debug: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to get train and test data loaders for a specified dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load ('debate', 'ld', 'chopped', or 'codecomprehension' currently supported)
        contrastive (bool): Whether to use contrastive version (only for debate dataset)
        max_train_samples (int): Maximum number of training samples (only for codecomprehension dataset)
        max_test_samples (int): Maximum number of test samples (only for codecomprehension dataset)
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    print(f"Loading dataset: {dataset_name} (contrastive={contrastive})")
    if dataset_name.lower() == "gsm8k":
        return build_gsm8k_dataloaders()
    if dataset_name.lower() == 'debate':
        print("DEBUG MODE:", debug)
        return build_debate_contrastive_dataloaders(debug) if contrastive else build_debate_dataloaders(debug)
    elif dataset_name.lower() == 'ld':
        return build_ld_dataloaders()
    elif dataset_name.lower() == 'chopped':
        return build_chopped_dataloaders()
    elif dataset_name.lower() == 'debate_code':
        # Use default values if not specified
        test = False
        if test:
            train_samples = 5
            test_samples = 2
        else:
            train_samples = max_train_samples if max_train_samples is not None else 100
            test_samples = max_test_samples if max_test_samples is not None else 30
        return build_codecomprehension_dataloaders(train_samples, test_samples)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported. Currently 'debate', 'ld', 'chopped', 'gsm8k' and 'debate_code' are available.")

class GSM8KDataLoader(DataLoader):
    """
    Iterates over the GSM8K grade-school math dataset.

    Each __next__ returns a tuple:
        (prompt, gold_answer)

    prompt  – already *includes* the pre-prompt + the raw question
    gold_answer – the numeric answer string after the final '####'.
    """
    SPLITS = {
        "train": "main/train-00000-of-00001.parquet",
        "test":  "main/test-00000-of-00001.parquet",
    }

    def __init__(self, split: str = "train", random: bool = False) -> None:
        super().__init__(random)
        if split not in self.SPLITS:
            raise ValueError("split must be 'train' or 'test'")

        parquet_path = f"hf://datasets/openai/gsm8k/{self.SPLITS[split]}"
        df = pd.read_parquet(parquet_path)

        # HuggingFace parquet columns are `question` and `answer`
        self.examples = df[["question", "answer"]].to_dict("records")

        # ── prompts ────────────────────────────────────────────────
        self.pre_prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n"
            "... step-by-step derivation ...\n"
            "</reasoning>\n"
            "<answer>\n"
            "... final numeric answer ...\n"
            "</answer>"
        )
        self.system_prompt = SYSTEM_PROMPT

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, random: bool):
        obj = cls.__new__(cls)           # bypass normal __init__
        super(GSM8KDataLoader, obj).__init__(random)
        obj.examples = df[["question", "answer"]].to_dict("records")
        obj.pre_prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
        )
        obj.system_prompt = SYSTEM_PROMPT
        obj.current_index = 0
        return obj

    # ── required ABC methods ──────────────────────────────────────
    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> "GSM8KDataLoader":
        return self

    def __next__(self) -> tuple[str, str]:  # Return tuple instead of just str
        if self.current_index >= len(self.examples):
            raise StopIteration

        idx = random.randint(0, len(self.examples) - 1) if self.random else self.current_index
        if not self.random:
            self.current_index += 1

        ex = self.examples[idx]
        question = ex["question"]
        
        # Extract gold answer
        gold_answer = extract_hash_answer(ex["answer"])

        prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
            f"Question:\n{question}"
        )
        
        return prompt, gold_answer  # Return both prompt and gold answer

    # optional helper
    def reset(self) -> None:
        self.current_index = 0




def build_gsm8k_dataloaders(
    train_size = 500,
    test_size  = 50,
    seed = 42,
) -> tuple[GSM8KDataLoader, GSM8KDataLoader]:
    """
    Build GSM-8K DataLoaders with optional down-sampling.

    Args
    ----
    train_size : how many examples to keep from the 7 473-example train split.
                 None  ⟶ keep them all.
    test_size  : how many examples to keep from the 1 319-example test split.
                 None  ⟶ keep them all.
    seed       : RNG seed so the same subset is repeatable.

    Returns
    -------
    (trainloader, testloader)
    """
    rng = random.Random(seed)

    # ---------- load full parquet files ----------
    def load_parquet(rel_path: str) -> pd.DataFrame:
        return pd.read_parquet(f"hf://datasets/openai/gsm8k/{rel_path}")

    train_df = load_parquet(GSM8KDataLoader.SPLITS["train"])
    test_df  = load_parquet(GSM8KDataLoader.SPLITS["test"])

    # ---------- optional down-sample ----------
    if train_size is not None and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=seed).reset_index(drop=True)
    if test_size is not None and test_size < len(test_df):
        test_df  = test_df.sample(n=test_size,  random_state=seed+1).reset_index(drop=True)

    # feed the *dataframes* straight into the loader
    trainloader = GSM8KDataLoader.from_dataframe(train_df, random=True)
    testloader  = GSM8KDataLoader.from_dataframe(test_df,  random=False)
    return trainloader, testloader

class GSM8KDataLoader(DataLoader):
    """
    Iterates over the GSM8K grade-school math dataset.

    Each __next__ returns a tuple:
        (prompt, gold_answer)

    prompt  – already *includes* the pre-prompt + the raw question
    gold_answer – the numeric answer string after the final '####'.
    """
    SPLITS = {
        "train": "main/train-00000-of-00001.parquet",
        "test":  "main/test-00000-of-00001.parquet",
    }

    def __init__(self, split: str = "train", random: bool = False) -> None:
        super().__init__(random)
        if split not in self.SPLITS:
            raise ValueError("split must be 'train' or 'test'")

        parquet_path = f"hf://datasets/openai/gsm8k/{self.SPLITS[split]}"
        df = pd.read_parquet(parquet_path)

        # HuggingFace parquet columns are `question` and `answer`
        self.examples = df[["question", "answer"]].to_dict("records")

        # ── prompts ────────────────────────────────────────────────
        self.pre_prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n"
            "... step-by-step derivation ...\n"
            "</reasoning>\n"
            "<answer>\n"
            "... final numeric answer ...\n"
            "</answer>"
        )
        self.system_prompt = SYSTEM_PROMPT

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, random: bool):
        obj = cls.__new__(cls)           # bypass normal __init__
        super(GSM8KDataLoader, obj).__init__(random)
        obj.examples = df[["question", "answer"]].to_dict("records")
        obj.pre_prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
        )
        obj.system_prompt = SYSTEM_PROMPT
        obj.current_index = 0
        return obj

    # ── required ABC methods ──────────────────────────────────────
    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> "GSM8KDataLoader":
        return self

    def __next__(self) -> tuple[str, str]:  # Return tuple instead of just str
        if self.current_index >= len(self.examples):
            raise StopIteration

        idx = random.randint(0, len(self.examples) - 1) if self.random else self.current_index
        if not self.random:
            self.current_index += 1

        ex = self.examples[idx]
        question = ex["question"]
        
        # Extract gold answer
        gold_answer = extract_hash_answer(ex["answer"])

        prompt = (
            "Solve the following grade-school math word problem.\n"
            "Respond ONLY in this format:\n"
            "<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>\n\n"
            f"Question:\n{question}"
        )
        
        return prompt, gold_answer  # Return both prompt and gold answer

    # optional helper
    def reset(self) -> None:
        self.current_index = 0




def build_gsm8k_dataloaders(
    train_size = 500,
    test_size  = 50,
    seed = 42,
) -> tuple[GSM8KDataLoader, GSM8KDataLoader]:
    """
    Build GSM-8K DataLoaders with optional down-sampling.

    Args
    ----
    train_size : how many examples to keep from the 7 473-example train split.
                 None  ⟶ keep them all.
    test_size  : how many examples to keep from the 1 319-example test split.
                 None  ⟶ keep them all.
    seed       : RNG seed so the same subset is repeatable.

    Returns
    -------
    (trainloader, testloader)
    """
    rng = random.Random(seed)

    # ---------- load full parquet files ----------
    def load_parquet(rel_path: str) -> pd.DataFrame:
        return pd.read_parquet(f"hf://datasets/openai/gsm8k/{rel_path}")

    train_df = load_parquet(GSM8KDataLoader.SPLITS["train"])
    test_df  = load_parquet(GSM8KDataLoader.SPLITS["test"])

    # ---------- optional down-sample ----------
    if train_size is not None and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=seed).reset_index(drop=True)
    if test_size is not None and test_size < len(test_df):
        test_df  = test_df.sample(n=test_size,  random_state=seed+1).reset_index(drop=True)

    # feed the *dataframes* straight into the loader
    trainloader = GSM8KDataLoader.from_dataframe(train_df, random=True)
    testloader  = GSM8KDataLoader.from_dataframe(test_df,  random=False)
    return trainloader, testloader


if __name__ == "__main__": 
    # Example usage for CodeComprehension dataset
    print("=== CodeComprehension Dataset Example ===")
    trainloader, testloader = get_dataloaders('codecomprehension', max_train_samples=50, max_test_samples=10)
    
    print(f"Training samples: {len(trainloader)}")
    print(f"Test samples: {len(testloader)}")
    
    # Show a few training examples
    print("\n--- Training Examples ---")
    for i, (question, choices, correct_idx) in enumerate(trainloader):
        if i >= 2:
            break
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"Choices: {choices}")
        print(f"Correct Answer Index: {correct_idx}")
        print(f"Correct Answer: {choices[correct_idx]}")
        print("-" * 50)
    
    # Reset and show test example
    testloader.reset()
    print("\n--- Test Example ---")
    test_question, test_choices, test_correct_idx = next(testloader)
    print(f"Question: {test_question}")
    print(f"Choices: {test_choices}")
    print(f"Correct Answer Index: {test_correct_idx}")
    print(f"Correct Answer: {test_choices[test_correct_idx]}")
    
    # Show the pre_prompt
    print("\n--- Pre-prompt Template ---")
    print(trainloader.pre_prompt)