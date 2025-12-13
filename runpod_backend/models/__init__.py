"""
Model loaders for STPE.

Provides unified loading interface for:
- DINOv3-7B (Vision Backbone - feature extraction)
- SAM-3 (Segmentation)
- Qwen3-VL (LLM Reasoning)
"""

from models.loader import ModelLoader
from models.dinov3 import DINOv3Model
from models.sam3.sam3_wrapper import SAM3Model
from models.qwen3_vision import Qwen3VisionModel

__all__ = [
    "ModelLoader",
    "DINOv3Model",
    "SAM3Model",
    "Qwen3VisionModel",
]
