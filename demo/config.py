"""
Configuration module for the Gradio debate demo.
Handles model setup and API key management for API-based LLM providers.
"""

import os
from typing import Optional, Dict
from dataclasses import dataclass

from simple_models import ModelInterface, create_model

@dataclass
class DemoConfig:
    """Configuration for the debate demo."""
    
    # Model settings
    judge_model_name: str = "gpt-4o-mini"
    arguing_model_name: str = "gpt-4o-mini"
    
    # Generation parameters
    temperature: float = 0.7
    max_new_tokens: int = 300
    judge_temperature: float = 0.1
    judge_max_tokens: int = 100
    
    # Demo settings
    max_argument_length: int = 300
    session_timeout_minutes: int = 60
    
    # UI settings
    theme: str = "soft"
    title: str = "🏛️ AI Debate Arena"
    description: str = "Debate against AI on fun topics! Make your best argument and see if you can convince the judge!"

class ModelManager:
    """Manages LLM models for the debate demo."""
    
    def __init__(self, config: DemoConfig):
        self.config = config
        self.judge_model: Optional[ModelInterface] = None
        self.arguing_model: Optional[ModelInterface] = None
        self._setup_models()
    
    def _setup_models(self):
        """Initialize the judge and arguing models."""
        try:
            # Setup judge model
            self.judge_model = self._create_model(self.config.judge_model_name)
            
            # Setup arguing model (can be same as judge)
            if self.config.arguing_model_name == self.config.judge_model_name:
                self.arguing_model = self.judge_model
            else:
                self.arguing_model = self._create_model(self.config.arguing_model_name)
                
            print(f"✅ Successfully loaded models:")
            print(f"   Judge: {self.config.judge_model_name}")
            print(f"   Arguing: {self.config.arguing_model_name}")
            
        except Exception as e:
            print(f"❌ Error setting up models: {e}")
            raise
    
    def _create_model(self, model_name: str) -> ModelInterface:
        """Create a model interface based on the model name."""
        if model_name.startswith('gpt-'):
            self._check_openai_key()
        elif model_name.startswith('claude-'):
            self._check_anthropic_key()
        elif model_name.startswith('openrouter/'):
            self._check_openrouter_key()
        else:
            # Default to OpenAI for unknown models
            self._check_openai_key()
            
        return create_model(model_name)
    
    
    def _check_openai_key(self):
        """Check if OpenAI API key is available."""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
    
    def _check_anthropic_key(self):
        """Check if Anthropic API key is available."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError(
                "Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable."
            )
    
    def _check_openrouter_key(self):
        """Check if OpenRouter API key is available."""
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable."
            )
    
    def generate_argument(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an argument using the arguing model."""
        return self.arguing_model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature
        )
    
    def judge_debate(self, system_prompt: str, user_prompt: str) -> str:
        """Judge a debate using the judge model."""
        return self.judge_model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature
        )

def load_config_from_env() -> DemoConfig:
    """Load configuration from environment variables."""
    return DemoConfig(
        judge_model_name=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
        arguing_model_name=os.getenv("ARGUING_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.7")),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "300")),
        judge_temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.1")),
        judge_max_tokens=int(os.getenv("JUDGE_MAX_TOKENS", "100")),
        max_argument_length=int(os.getenv("MAX_ARGUMENT_LENGTH", "300")),
    )

def get_available_models() -> Dict[str, list]:
    """Return a dictionary of available models by provider."""
    return {
        "OpenAI": [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ],
        "Anthropic": [
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229",
        ],
        "OpenRouter": [
            "openrouter/anthropic/claude-3-haiku",
            "openrouter/openai/gpt-4o-mini",
            "openrouter/google/gemini-pro",
            "openrouter/meta-llama/llama-3-8b-instruct",
        ]
    }

if __name__ == "__main__":
    # Test configuration
    config = load_config_from_env()
    print("Demo Configuration:")
    print(f"  Judge Model: {config.judge_model_name}")
    print(f"  Arguing Model: {config.arguing_model_name}")
    print(f"  Temperature: {config.temperature}")
    
    try:
        model_manager = ModelManager(config)
        print("✅ Model manager initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize model manager: {e}")