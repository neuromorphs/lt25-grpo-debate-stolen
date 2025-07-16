"""
Simplified model interfaces for API-based models only.
No local HuggingFace dependencies required.
"""

import os
import time
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class ModelInterface(ABC):
    """Abstract base class for language model interfaces."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate text from the model given a system prompt and user prompt."""
        pass

class OpenAIModel(ModelInterface):
    """Implementation of ModelInterface for OpenAI API models."""
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Translate common parameters to OpenAI parameters
        openai_kwargs = {}
        if 'max_new_tokens' in kwargs:
            openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        if 'temperature' in kwargs:
            openai_kwargs['temperature'] = kwargs.pop('temperature')
        if 'top_p' in kwargs:
            openai_kwargs['top_p'] = kwargs.pop('top_p')
            
        max_retries = 5
        base_delay = 1
        
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
                if attempt == max_retries - 1:
                    raise e
                    
                delay = base_delay * (2 ** attempt)
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
            max_tokens=kwargs.get('max_new_tokens', 4096),
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
        )
        return response.content[0].text.strip()

class OpenRouterModel(ModelInterface):
    """Implementation for OpenRouter API using OpenAI-compatible interface."""
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.model_name = model_name
    
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Translate parameters
        openai_kwargs = {}
        if 'max_new_tokens' in kwargs:
            openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        if 'temperature' in kwargs:
            openai_kwargs['temperature'] = kwargs.pop('temperature')
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **openai_kwargs
        )
        return response.choices[0].message.content.strip()

def create_model(model_name: str) -> ModelInterface:
    """Create a model interface based on the model name."""
    if model_name.startswith('gpt-'):
        return OpenAIModel(model_name)
    elif model_name.startswith('claude-'):
        return AnthropicModel(model_name)
    elif model_name.startswith('openrouter/'):
        # Remove 'openrouter/' prefix for actual model name
        actual_model_name = model_name.replace('openrouter/', '')
        return OpenRouterModel(actual_model_name)
    else:
        # Default to OpenAI if unclear
        return OpenAIModel(model_name)