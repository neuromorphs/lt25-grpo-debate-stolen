"""
Test script for vLLM integration and memory management.
"""
import torch
from model_interface import VLLMModel, VLLM_AVAILABLE
from transformers import AutoTokenizer

def get_gpu_memory_info():
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    return allocated_memory, total_memory

def monitor_gpu_memory() -> dict:
    """Monitor current GPU memory usage."""
    allocated, total = get_gpu_memory_info()
    
    return {
        "allocated_gb": allocated,
        "total_gb": total,
        "utilization_percent": (allocated / total * 100) if total > 0 else 0,
        "free_gb": total - allocated
    }

def count_tokens(system_prompt: str, user_prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> int:
    """
    Count the number of tokens in system and user prompts using the specified model's tokenizer.
    Applies the chat template to get the actual token count that the model will see.
    
    Args:
        system_prompt: The system prompt/instructions
        user_prompt: The user's input prompt
        model_name: The model name to use for tokenization
        
    Returns:
        int: Number of tokens in the formatted prompt
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Format prompt in chat template (same as in HuggingFaceModel)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize the formatted prompt
        tokens = tokenizer.encode(prompt_text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def test_vllm_model(model_name: str = "microsoft/DialoGPT-small"):
    """Test vLLM model with memory constraints."""
    print(f"\n=== vLLM Model Test ({model_name}) ===")
    
    if not VLLM_AVAILABLE:
        print("vLLM not available, skipping test")
        return
    
    try:
        # Estimate memory split (assuming 2GB for training)
        model_size = 2.9  # GB
        allocated, total = get_gpu_memory_info()
        available = total - allocated
        safety_margin = 0.5
        vllm_util = max(0.1, (model_size / available) * (1 + safety_margin))
        print(f"Using vLLM memory utilization: {vllm_util:.2f}")
        
        # Initialize vLLM model with memory constraint
        model = VLLMModel(
            model_name=model_name,
            gpu_memory_utilization=vllm_util,
            max_model_len=1024  # Small context for testing
        )
        
        # Test generation
        # system_prompt = "You are a helpful assistant."
        # user_prompt = "What is 2+2?"
        system_prompt = "You are an impartial debate judge."
        user_prompt = """You are an impartial debate judge. You will be shown two debate responses on the same topic, 
        arguing the same side (PRO or CON). Your task is to determine which argument was more compelling based on:
        1. Logical reasoning and evidence
        2. Clear structure and organization
        3. Effective use of examples
        4. Respectful tone
        5. Addressing potential counterarguments
        
        Topic: {topic}
        
        Argument 1:
        {arg1_response}
        
        Argument 2:
        {arg2_response}
        
        Which response was more compelling? Respond with EXACTLY one of these options:
        - ARGUMENT_1_WINS
        - ARGUMENT_2_WINS

        YOU MUST CHOOSE A WINNER, A TIE IS NOT ALLOWED
        """
        user_prompt_ = user_prompt.format(
            topic="Should we ban AI?",
            arg1_response="",
            arg2_response="",
        )
        token_count = count_tokens(system_prompt, user_prompt_, model_name)
        print(f"Token count: {token_count}")

        user_prompt = user_prompt.format(
            topic="Should we ban AI?",
            arg1_response="I think we should ban AI." * 10,
            arg2_response="I think we should not ban AI." * 10,
        )
        token_count = count_tokens(system_prompt, user_prompt, model_name)
        print(f"Token count: {token_count}")
        
        print("Generating response...")
        response = model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_new_tokens=100,
        )
        
        print(f"Response: {response}")
        
        # Monitor memory after initialization
        memory_after = monitor_gpu_memory()
        print(f"Memory after vLLM init: {memory_after['utilization_percent']:.1f}%")
        
    except Exception as e:
        print(f"vLLM test failed: {e}")

if __name__ == "__main__":
    print("vLLM Integration Test")
    print("=" * 50)
    
    # Test vLLM model (uncomment if you have a small model available)
    test_vllm_model("Qwen/Qwen2.5-1.5B-Instruct")
    
    print("\nTest completed!")