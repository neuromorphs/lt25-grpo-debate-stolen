"""
Abstract interface for language models and their implementations.
"""
import time
import torch
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text from the model given a system prompt and user prompt.
        
        Args:
            system_prompt: The system prompt/instructions
            user_prompt: The user's input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            str: The generated text
        """
        pass

class HuggingFaceModel(ModelInterface):
    """Implementation of ModelInterface for Hugging Face models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        return self._generate(system_prompt, user_prompt, **kwargs)
    
    def _generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Format prompt in chat template
        prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs
        )
        
        # Decode only the new tokens
        response = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

class OpenAIModel(ModelInterface):
    """Implementation of ModelInterface for OpenAI API models."""
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Translate common HF parameters to OpenAI parameters
        openai_kwargs = {}
        if 'max_new_tokens' in kwargs:
            openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        if 'temperature' in kwargs:
            openai_kwargs['temperature'] = kwargs.pop('temperature')
        if 'top_p' in kwargs:
            openai_kwargs['top_p'] = kwargs.pop('top_p')
            
        max_retries = 5
        base_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **openai_kwargs
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                    
                # Calculate exponential backoff delay
                delay = base_delay * (2 ** attempt)  # 1, 2, 4, 8, 16 seconds
                time.sleep(delay)
                continue

class AnthropicModel(ModelInterface):
    """Implementation of ModelInterface for Anthropic API models."""
    
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get('max_tokens', 4096),
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
        )
        return response.content[0].text.strip()

class VLLMModel(ModelInterface):
    """Implementation of ModelInterface for vLLM models with memory management."""
    
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.5, **kwargs):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is not installed. Install with: pip install vllm")
        
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        
        # Initialize vLLM with memory constraints
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Convert HF-style parameters to vLLM SamplingParams
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_new_tokens', 512),
            stop=kwargs.get('stop_sequences', None)
        )
        
        # Format as chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate response
        outputs = self.llm.chat(messages, sampling_params)
        
        return outputs[0].outputs[0].text.strip()