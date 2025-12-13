"""
Vision Processing Modules for STPE.

Implements the 3-level vision processing pipeline:
- Level 1: Global image embedding
- Level 2: Auto-segmentation + object embeddings
- Level 3: High-resolution dense maps
"""

from processing.level1_global import Level1GlobalProcessor
from processing.level2_objects import Level2ObjectProcessor, DetectedObject
from processing.level3_dense import Level3DenseProcessor
from processing.vision_pipeline import VisionPipeline, FrameResult

__all__ = [
    "Level1GlobalProcessor",
    "Level2ObjectProcessor",
    "Level3DenseProcessor",
    "VisionPipeline",
    "FrameResult",
    "DetectedObject",
]
