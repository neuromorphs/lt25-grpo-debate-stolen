"""
Memory management utilities for concurrent training and vLLM inference.
"""
import torch
import psutil
from typing import Tuple, Optional

def get_gpu_memory_info() -> Tuple[float, float]:
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    
    return allocated_memory, total_memory

def estimate_memory_split(training_model_size_gb: float, safety_margin: float = 0.1) -> float:
    """
    Estimate optimal vLLM gpu_memory_utilization based on training model size.
    
    Args:
        training_model_size_gb: Estimated training model memory usage in GB
        safety_margin: Safety margin to prevent OOM (default 10%)
    
    Returns:
        Recommended gpu_memory_utilization for vLLM (0.0-1.0)
    """
    _, total_gpu_memory = get_gpu_memory_info()
    
    if total_gpu_memory == 0:
        raise RuntimeError("No GPU available")
    
    # Reserve memory for training + safety margin
    reserved_memory = training_model_size_gb + (total_gpu_memory * safety_margin)
    available_for_vllm = total_gpu_memory - reserved_memory
    
    # Calculate utilization ratio for vLLM
    vllm_utilization = max(0.1, min(0.9, available_for_vllm / total_gpu_memory))
    
    return vllm_utilization

def set_training_memory_limit(fraction: float):
    """Set memory fraction limit for training model."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)

def monitor_gpu_memory() -> dict:
    """Monitor current GPU memory usage."""
    allocated, total = get_gpu_memory_info()
    
    return {
        "allocated_gb": allocated,
        "total_gb": total,
        "utilization_percent": (allocated / total * 100) if total > 0 else 0,
        "free_gb": total - allocated
    }