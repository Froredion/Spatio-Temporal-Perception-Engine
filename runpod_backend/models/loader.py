"""
Unified Model Loader for STPE.

Handles loading/unloading of all models with memory management.
"""

import torch
from typing import Optional

from config import STPEConfig, default_config
from models.dinov3 import DINOv3Model
from models.sam3.sam3_wrapper import SAM3Model
from models.qwen3_vision import Qwen3VisionModel


class ModelLoader:
    """
    Unified model loader with memory management.

    Usage:
        loader = ModelLoader(config)
        loader.load_all()

        # Use models
        dinov3_output = loader.dinov3.encode(image)
        sam3_segments = loader.sam3.segment_with_text(image, "person")
        vlm_output = loader.vlm.caption(image)

        # Cleanup
        loader.unload_all()
    """

    def __init__(
        self,
        config: Optional[STPEConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or default_config
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize model wrappers (not loaded yet)
        self.dinov3: Optional[DINOv3Model] = None
        self.sam3: Optional[SAM3Model] = None
        self.vlm: Optional[Qwen3VisionModel] = None

        self._models_loaded = False

    def load_all(self):
        """Load all models."""
        print(f"Loading STPE models on {self.device}...")

        self.dinov3 = DINOv3Model(self.config.dinov3, self.device)
        self.dinov3.load()
        print(f"  [+] DINOv3 loaded ({self._get_gpu_memory_usage():.1f} GB VRAM)")

        self.sam3 = SAM3Model(self.config.sam3, self.device)
        self.sam3.load()
        print(f"  [+] SAM-3 loaded ({self._get_gpu_memory_usage():.1f} GB VRAM)")

        self.vlm = Qwen3VisionModel(self.config.vlm, self.device)
        self.vlm.load()
        print(f"  [+] Qwen3-VL loaded ({self._get_gpu_memory_usage():.1f} GB VRAM)")

        self._models_loaded = True
        print(f"All models loaded. Total VRAM: {self._get_gpu_memory_usage():.1f} GB")

    def load_vision_only(self):
        """Load only DINOv3 and SAM-3 (skip LLM for embedding-only tasks)."""
        print(f"Loading vision models on {self.device}...")

        self.dinov3 = DINOv3Model(self.config.dinov3, self.device)
        self.dinov3.load()
        print(f"  [+] DINOv3 loaded ({self._get_gpu_memory_usage():.1f} GB VRAM)")

        self.sam3 = SAM3Model(self.config.sam3, self.device)
        self.sam3.load()
        print(f"  [+] SAM-3 loaded ({self._get_gpu_memory_usage():.1f} GB VRAM)")

        print(f"Vision models loaded. Total VRAM: {self._get_gpu_memory_usage():.1f} GB")

    def load_dinov3(self):
        """Load only DINOv3."""
        if self.dinov3 is None:
            self.dinov3 = DINOv3Model(self.config.dinov3, self.device)
        self.dinov3.load()

    def load_sam3(self):
        """Load only SAM-3."""
        if self.sam3 is None:
            self.sam3 = SAM3Model(self.config.sam3, self.device)
        self.sam3.load()

    def load_vlm(self):
        """Load only Qwen3-VL."""
        if self.vlm is None:
            self.vlm = Qwen3VisionModel(self.config.vlm, self.device)
        self.vlm.load()

    def unload_all(self):
        """Unload all models and free GPU memory."""
        if self.dinov3 is not None:
            self.dinov3.unload()
            self.dinov3 = None

        if self.sam3 is not None:
            self.sam3.unload()
            self.sam3 = None

        if self.vlm is not None:
            self.vlm.unload()
            self.vlm = None

        self._models_loaded = False
        torch.cuda.empty_cache()
        print("All models unloaded.")

    def unload_vlm(self):
        """Unload only Qwen3-VL to save memory."""
        if self.vlm is not None:
            self.vlm.unload()
            print(f"Qwen3-VL unloaded. VRAM: {self._get_gpu_memory_usage():.1f} GB")

    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "device": torch.cuda.get_device_name(),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }

    @property
    def is_loaded(self) -> bool:
        """Check if all models are loaded."""
        return self._models_loaded

    def __enter__(self):
        """Context manager entry."""
        self.load_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_all()
        return False
