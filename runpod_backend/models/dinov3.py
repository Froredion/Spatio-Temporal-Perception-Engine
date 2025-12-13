"""
DINOv3-7B Model Wrapper

Primary Vision Backbone for feature extraction:
- Level 1: Global embedding (4096-dim CLS token)
- Level 2: Patch features for object embedding (segmentation via SAM-3)
- Level 3: High-resolution dense feature maps

Key specs:
- 6.7B parameters
- 4096-dim embeddings (vs 1536 in DINOv2)
- Native high-res support up to 4096x4096
- Gram Anchoring for dense task preservation
- Axial RoPE positional embeddings

Note: Segmentation is now handled by SAM-3 (models/sam3.py).
DINOv3 provides embeddings; SAM-3 provides masks.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from PIL import Image


class DINOv3Model:
    """DINOv3-7B inference wrapper with 3-level output."""

    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self):
        """Load DINOv3 model and processor."""
        if self._loaded:
            return

        from transformers import AutoModel, AutoImageProcessor

        print(f"Loading DINOv3 from {self.config.model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )

        self.model.eval()
        self._loaded = True
        print(f"DINOv3 loaded successfully")

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self._loaded = False
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(
        self,
        image: Image.Image,
        return_patch_features: bool = True,
        high_res: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode image through DINOv3 at all 3 levels.

        Args:
            image: PIL Image to encode
            return_patch_features: Whether to return patch-level features
            high_res: Whether to process at high resolution

        Returns:
            Dictionary with:
                - global_embedding: (1, 4096) CLS token embedding
                - patch_features: (1, num_patches, 4096) if return_patch_features
                - feature_map: (1, H, W, 4096) reshaped dense features
        """
        if not self._loaded:
            self.load()

        # Prepare size based on high_res flag
        size_kwargs = {}
        if high_res:
            size_kwargs = {"size": {"height": self.config.max_resolution, "width": self.config.max_resolution}}

        inputs = self.processor(images=image, return_tensors="pt", **size_kwargs)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        # Level 1: Global embedding (CLS token)
        global_embedding = outputs.last_hidden_state[:, 0]

        result = {"global_embedding": global_embedding}

        if return_patch_features:
            # Level 2: Patch features (excluding CLS and registers)
            # DINOv3 uses 4 registers, so patches start at index 5
            num_registers = getattr(self.config, 'num_registers', 4)
            patch_features = outputs.last_hidden_state[:, 1 + num_registers:]
            result["patch_features"] = patch_features

            # Level 3: Reshape to spatial feature map
            num_patches = patch_features.shape[1]
            h = w = int(num_patches ** 0.5)
            if h * w == num_patches:
                feature_map = patch_features.reshape(1, h, w, -1)
                result["feature_map"] = feature_map

        return result

    @torch.no_grad()
    def encode_batch(
        self,
        images: list,
        return_patch_features: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode batch of images.

        Args:
            images: List of PIL Images
            return_patch_features: Whether to return patch-level features

        Returns:
            Dictionary with batched tensors
        """
        if not self._loaded:
            self.load()

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        # Global embeddings for all images
        global_embeddings = outputs.last_hidden_state[:, 0]

        result = {"global_embedding": global_embeddings}

        if return_patch_features:
            num_registers = getattr(self.config, 'num_registers', 4)
            patch_features = outputs.last_hidden_state[:, 1 + num_registers:]
            result["patch_features"] = patch_features

        return result

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.config.embedding_dim
