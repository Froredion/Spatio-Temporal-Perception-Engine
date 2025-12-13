"""
Memory management utilities for GPU and Python object cleanup.

Provides functions to clear CUDA cache, force garbage collection,
and monitor memory usage. Used after processing requests since
R2 is the source of truth and data doesn't need to persist in memory.
"""

import gc
from typing import Dict, Any, Optional

import torch


def clear_gpu_cache() -> None:
    """
    Clear PyTorch CUDA cache.

    Releases all unoccupied cached memory held by the caching allocator.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clear_python_gc() -> None:
    """
    Force Python garbage collection.

    Runs all three generations of garbage collection to ensure
    unreferenced objects are cleaned up promptly.
    """
    gc.collect()
    gc.collect()
    gc.collect()


def full_cleanup() -> None:
    """
    Full memory cleanup - Python GC followed by GPU cache clear.

    Call this after processing requests to return VRAM to baseline.
    Model weights remain loaded; only intermediate results are cleared.
    """
    clear_python_gc()
    clear_gpu_cache()


def get_memory_info() -> Dict[str, Any]:
    """
    Get current GPU memory statistics.

    Returns:
        Dictionary with memory info or error if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA not available"}

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory

    return {
        "available": True,
        "allocated_gb": round(allocated / 1024**3, 2),
        "reserved_gb": round(reserved / 1024**3, 2),
        "max_allocated_gb": round(max_allocated / 1024**3, 2),
        "total_gb": round(total / 1024**3, 2),
        "utilization_percent": round((allocated / total) * 100, 1),
    }


def reset_peak_memory_stats() -> None:
    """Reset peak memory tracking statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def delete_tensors(*tensors) -> None:
    """
    Explicitly delete tensor references and clear cache.

    Args:
        *tensors: Variable number of tensors to delete.
    """
    for t in tensors:
        if t is not None:
            del t
    clear_gpu_cache()


def log_memory_usage(prefix: str = "") -> None:
    """
    Log current memory usage to console.

    Args:
        prefix: Optional prefix for the log message.
    """
    info = get_memory_info()
    if info.get("available"):
        msg = f"[Memory] {prefix}" if prefix else "[Memory]"
        print(
            f"{msg} Allocated: {info['allocated_gb']:.2f} GB, "
            f"Reserved: {info['reserved_gb']:.2f} GB, "
            f"Utilization: {info['utilization_percent']:.1f}%"
        )
