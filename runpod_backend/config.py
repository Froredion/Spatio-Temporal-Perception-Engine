"""
STPE Configuration - Model specifications and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DINOv3Config:
    """DINOv3-7B configuration."""
    model_id: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    embedding_dim: int = 4096
    patch_size: int = 16
    max_resolution: int = 2048  # Can go up to 4096
    num_registers: int = 4
    use_fp16: bool = True


@dataclass
class SAM3Config:
    """SAM-3 segmentation model configuration."""
    model_id: str = "facebook/sam3"
    use_fp16: bool = True
    default_threshold: float = 0.3  # Lower threshold to detect more objects
    default_mask_threshold: float = 0.5


@dataclass
class VLMConfig:
    """Qwen3-VL configuration (Vision Language Model)."""
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    max_new_tokens: int = 512
    use_bf16: bool = True  # Qwen3-VL works best with bfloat16


@dataclass
class TemporalConfig:
    """Temporal modules configuration."""
    d_model: int = 4096  # Match DINOv3 embedding dim
    max_frames: int = 1000
    num_attention_heads: int = 8
    num_attention_layers: int = 4
    motion_hidden_dim: int = 4096
    pooling_hidden_dim: int = 1024
    similarity_threshold: float = 0.8


@dataclass
class SceneGraphConfig:
    """Scene graph configuration."""
    node_dim: int = 4096  # Match DINOv3 embedding dim
    edge_dim: int = 1024
    num_gnn_layers: int = 3


@dataclass
class ProcessingConfig:
    """Video processing configuration."""
    default_fps: float = 2.0
    batch_size: int = 4
    clustering_threshold: float = 0.7
    min_object_area: int = 100  # Minimum pixels for valid object
    # Default processing resolution (width, height) - None means use original
    # Recommended: (854, 480) for 480p - good balance of speed and quality
    # Options: (1920, 1080) 1080p, (1280, 720) 720p, (960, 540) 540p, (854, 480) 480p, (640, 360) 360p
    default_processing_resolution: tuple = None


@dataclass
class STPEConfig:
    """Master STPE configuration."""
    dinov3: DINOv3Config = field(default_factory=DINOv3Config)
    sam3: SAM3Config = field(default_factory=SAM3Config)
    vlm: VLMConfig = field(default_factory=VLMConfig)  # Vision Language Model (Qwen3-VL)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    device: str = "cuda"


# Default configuration instance
default_config = STPEConfig()
