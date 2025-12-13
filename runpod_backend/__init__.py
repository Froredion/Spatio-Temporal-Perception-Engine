"""
Spatio-Temporal Perception Engine (STPE) - RunPod Backend

A unified perception backbone that combines DINOv3 (vision), SAM-3 (segmentation),
Qwen3-VL (reasoning), and custom temporal modules to convert raw video into
hierarchical spatial-temporal embeddings, object tracks, and query-ready representations.
"""

__version__ = "1.0.0"
__author__ = "ClipSearchAI"

from pipeline import STPEPipeline
from models import ModelLoader
from handler import handler

__all__ = [
    "STPEPipeline",
    "ModelLoader",
    "handler",
]
