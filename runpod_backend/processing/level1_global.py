"""
Level 1: Global Image Embedding

Produces scene-level understanding and similarity search embeddings.
Uses DINOv3 CLS token as the global embedding.
"""

import torch
import numpy as np
from typing import Dict, Optional
from PIL import Image

from models import DINOv3Model


class Level1GlobalProcessor:
    """
    Level 1 processor for global image embeddings.

    Extracts:
    - Global embedding (4096-dim CLS token)
    - Optional PCA grid for spatial awareness visualization
    """

    def __init__(self, dinov3_model: DINOv3Model, n_pca_components: int = 3):
        self.dinov3 = dinov3_model
        self.n_pca_components = n_pca_components

    @torch.no_grad()
    def process(
        self,
        image: Image.Image,
        compute_pca_grid: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Process image to extract global embedding.

        Args:
            image: PIL Image to process
            compute_pca_grid: Whether to compute PCA visualization of patches

        Returns:
            Dictionary with:
                - global_embedding: (1, 4096) scene-level embedding
                - patch_features: (1, num_patches, 4096) for Level 2
                - pca_grid: (H, W, 3) RGB visualization if requested
        """
        # Get DINOv3 outputs
        outputs = self.dinov3.encode(
            image,
            return_patch_features=True,
        )

        result = {
            "global_embedding": outputs["global_embedding"],
            "patch_features": outputs["patch_features"],
        }

        if compute_pca_grid and "feature_map" in outputs:
            pca_grid = self._compute_pca_grid(outputs["feature_map"])
            result["pca_grid"] = pca_grid

        return result

    @torch.no_grad()
    def process_batch(
        self,
        images: list,
    ) -> Dict[str, torch.Tensor]:
        """
        Process batch of images.

        Args:
            images: List of PIL Images

        Returns:
            Dictionary with batched tensors
        """
        outputs = self.dinov3.encode_batch(
            images,
            return_patch_features=True,
        )

        return {
            "global_embeddings": outputs["global_embedding"],
            "patch_features": outputs["patch_features"],
        }

    def _compute_pca_grid(self, feature_map: torch.Tensor) -> np.ndarray:
        """
        Compute PCA visualization of spatial features.

        Args:
            feature_map: (1, H, W, D) feature tensor

        Returns:
            (H, W, 3) RGB numpy array for visualization
        """
        # Reshape to (H*W, D)
        h, w = feature_map.shape[1:3]
        features_flat = feature_map[0].reshape(-1, feature_map.shape[-1])

        # Move to CPU for sklearn
        features_np = features_flat.cpu().float().numpy()

        # Apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_pca_components)
        pca_features = pca.fit_transform(features_np)

        # Normalize to 0-255 range for RGB
        pca_min = pca_features.min(axis=0)
        pca_max = pca_features.max(axis=0)
        pca_normalized = (pca_features - pca_min) / (pca_max - pca_min + 1e-8)
        pca_rgb = (pca_normalized * 255).astype(np.uint8)

        # Reshape back to spatial grid
        pca_grid = pca_rgb.reshape(h, w, 3)

        return pca_grid

    def get_embedding_for_search(self, image: Image.Image) -> torch.Tensor:
        """
        Get embedding optimized for similarity search.

        Args:
            image: PIL Image

        Returns:
            Normalized (1, 4096) embedding
        """
        outputs = self.process(image, compute_pca_grid=False)
        embedding = outputs["global_embedding"]

        # L2 normalize for cosine similarity
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)

        return embedding
