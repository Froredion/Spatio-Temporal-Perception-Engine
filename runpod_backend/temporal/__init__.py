"""
Temporal Modules for STPE.

Custom temporal components for video understanding:
- Temporal Positional Encoding
- Object Tracking (DINOv3 feature-based)
- Motion Feature Extraction
- Cross-frame Temporal Attention
- Temporal Pooling for clip embeddings
"""

from temporal.positional_encoding import TemporalPositionalEncoding
from temporal.tracker import DINOv3Tracker
from temporal.motion import MotionFeatureExtractor
from temporal.attention import TemporalAttention
from temporal.pooling import TemporalPooling

__all__ = [
    "TemporalPositionalEncoding",
    "DINOv3Tracker",
    "MotionFeatureExtractor",
    "TemporalAttention",
    "TemporalPooling",
]
