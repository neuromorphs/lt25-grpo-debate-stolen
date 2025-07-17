"""
Test script for vLLM integration and memory management.
"""
import torch
from model_interface import VLLMModel, VLLM_AVAILABLE
from memory_utils import get_gpu_memory_info, estimate_memory_split, monitor_gpu_memory

def test_memory_utilities():
    """Test memory management utilities."""
    print("=== Memory Utilities Test ===")
    
    # Test GPU memory info
    allocated, total = get_gpu_memory_info()
    print(f"GPU Memory: {allocated:.1f}GB / {total:.1f}GB")
    
    # Test memory monitoring
    memory_info = monitor_gpu_memory()
    print(f"Memory utilization: {memory_info['utilization_percent']:.1f}%")
    
    # Test memory split estimation
    if total > 0:
        for training_size in [2.0, 4.0, 6.0]:
            try:
                vllm_util = estimate_memory_split(training_size)
                print(f"Training: {training_size}GB -> vLLM utilization: {vllm_util:.2f}")
            except RuntimeError as e:
                print(f"Training: {training_size}GB -> Error: {e}")

def test_vllm_model(model_name: str = "microsoft/DialoGPT-small"):
    """Test vLLM model with memory constraints."""
    print(f"\n=== vLLM Model Test ({model_name}) ===")
    
    if not VLLM_AVAILABLE:
        print("vLLM not available, skipping test")
        return
    
    try:
        # Estimate memory split (assuming 2GB for training)
        vllm_util = estimate_memory_split(2.0)
        print(f"Using vLLM memory utilization: {vllm_util:.2f}")
        
        # Initialize vLLM model with memory constraint
        model = VLLMModel(
            model_name=model_name,
            gpu_memory_utilization=vllm_util,
            max_model_len=512  # Small context for testing
        )
        
        # Test generation
        system_prompt = "You are a helpful assistant."
        user_prompt = "What is 2+2?"
        
        print("Generating response...")
        response = model.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.7,
            max_new_tokens=50
        )
        
        print(f"Response: {response}")
        
        # Monitor memory after initialization
        memory_after = monitor_gpu_memory()
        print(f"Memory after vLLM init: {memory_after['utilization_percent']:.1f}%")
        
    except Exception as e:
        print(f"vLLM test failed: {e}")

def simulate_concurrent_usage():
    """Simulate concurrent training and inference scenario."""
    print("\n=== Concurrent Usage Simulation ===")
    
    # Simulate training model allocation
    if torch.cuda.is_available():
        print("Simulating training model...")
        # Create a dummy tensor to simulate training model memory
        dummy_training = torch.randn(1000, 1000, device='cuda')
        
        memory_with_training = monitor_gpu_memory()
        print(f"Memory with simulated training: {memory_with_training['utilization_percent']:.1f}%")
        
        # Calculate remaining memory for vLLM
        remaining_gb = memory_with_training['free_gb']
        total_gb = memory_with_training['total_gb']
        vllm_util = max(0.1, remaining_gb / total_gb * 0.8)  # Use 80% of remaining
        
        print(f"Recommended vLLM utilization: {vllm_util:.2f}")
        print(f"This would use ~{vllm_util * total_gb:.1f}GB for vLLM")
        
        # Cleanup
        del dummy_training
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("vLLM Integration Test")
    print("=" * 50)
    
    # Test memory utilities
    test_memory_utilities()
    
    # Test concurrent usage simulation
    simulate_concurrent_usage()
    
    # Test vLLM model (uncomment if you have a small model available)
    # test_vllm_model()
    
    print("\nTest completed!")