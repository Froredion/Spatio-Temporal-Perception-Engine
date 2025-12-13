"""
Dense Captioning Module for STPE

Provides rich, multi-granularity captions for training data generation:
- Dense per-frame captions (every frame or configurable interval)
- Multi-granularity levels (brief, normal, detailed)
- Spatial relationship extraction
"""

from captioning.dense_captioner import DenseCaptioner
from captioning.spatial_relations import SpatialRelationExtractor

__all__ = [
    "DenseCaptioner",
    "SpatialRelationExtractor",
]
