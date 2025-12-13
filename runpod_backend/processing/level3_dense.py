"""
Level 3: High-Resolution Dense Map

Generates high-resolution dense feature maps for:
- Object tracking
- Movement detection
- Spatial reasoning
- Depth cues
- Edge detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from PIL import Image

from models import DINOv3Model


class Level3DenseProcessor:
    """
    Level 3 processor for high-resolution dense feature maps.

    Extracts:
    - Dense feature grid at high resolution
    - Implicit depth cues from features
    - Edge/boundary maps from feature gradients
    """

    def __init__(
        self,
        dinov3_model: DINOv3Model,
        high_res_size: int = 2048,
    ):
        self.dinov3 = dinov3_model
        self.high_res_size = high_res_size

    @torch.no_grad()
    def process(
        self,
        image: Image.Image,
        high_res: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate high-resolution dense feature map.

        Args:
            image: PIL Image to process
            high_res: Whether to process at high resolution (2048x2048)

        Returns:
            Dictionary with:
                - features: (1, H, W, D) dense feature grid
                - depth_cues: (H, W) relative depth estimation
                - edge_map: (H, W) object boundary detection
        """
        outputs = self.dinov3.encode(
            image,
            return_patch_features=True,
            high_res=high_res,
        )

        patch_features = outputs["patch_features"]

        # Reshape to spatial grid
        num_patches = patch_features.shape[1]
        h = w = int(np.sqrt(num_patches))

        if h * w != num_patches:
            h = int(np.sqrt(num_patches))
            w = num_patches // h

        features = patch_features.reshape(1, h, w, -1)

        result = {
            "features": features,
        }

        # Extract depth cues from features
        depth_cues = self._extract_depth_cues(features)
        result["depth_cues"] = depth_cues

        # Extract edge map from feature gradients
        edge_map = self._extract_edge_map(features)
        result["edge_map"] = edge_map

        return result

    def _extract_depth_cues(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract relative depth estimation from DINOv3 features.

        DINOv3 learns implicit depth information through self-supervised training.
        We extract this by projecting features to a depth-like representation.

        Args:
            features: (1, H, W, D) feature grid

        Returns:
            (H, W) relative depth map (higher = further)
        """
        h, w, d = features.shape[1:4]
        features_flat = features[0].reshape(-1, d)

        # Compute mean and center
        mean = features_flat.mean(dim=0)
        centered = features_flat - mean

        try:
            # Get first principal direction via power iteration (more robust than eigh)
            # Power iteration: v = A @ v / ||A @ v|| repeated
            cov_approx = torch.mm(centered.T, centered) / (h * w)

            # Add regularization to prevent ill-conditioning
            cov_approx = cov_approx + 1e-5 * torch.eye(d, device=cov_approx.device, dtype=cov_approx.dtype)

            # Power iteration for largest eigenvector (5 iterations is usually enough)
            v = torch.randn(d, device=cov_approx.device, dtype=cov_approx.dtype)
            v = v / torch.norm(v)

            for _ in range(5):
                v = torch.mv(cov_approx.float(), v.float())
                v = v / (torch.norm(v) + 1e-8)

            first_pc = v.to(centered.dtype)

            # Project features onto first PC
            depth = torch.mm(centered, first_pc.unsqueeze(1)).squeeze()
            depth = depth.reshape(h, w)

            # Normalize to 0-1
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        except Exception as e:
            print(f"Depth cue extraction failed: {e}, using feature norm fallback")
            # Fallback: use feature norm as depth proxy
            depth = torch.norm(features[0], dim=-1)
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        return depth

    def _extract_edge_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract object boundaries from feature gradients.

        High gradient magnitude indicates object boundaries.

        Args:
            features: (1, H, W, D) feature grid

        Returns:
            (H, W) edge strength map
        """
        # Compute feature magnitude per patch
        feature_norm = torch.norm(features[0], dim=-1)  # (H, W)

        # Compute gradients in x and y
        # Sobel-like operators
        grad_x = torch.zeros_like(feature_norm)
        grad_y = torch.zeros_like(feature_norm)

        # Horizontal gradient
        grad_x[:, 1:-1] = feature_norm[:, 2:] - feature_norm[:, :-2]

        # Vertical gradient
        grad_y[1:-1, :] = feature_norm[2:, :] - feature_norm[:-2, :]

        # Gradient magnitude
        edge_map = torch.sqrt(grad_x**2 + grad_y**2)

        # Normalize
        edge_map = (edge_map - edge_map.min()) / (edge_map.max() - edge_map.min() + 1e-8)

        return edge_map

    def compute_flow(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute optical flow-like correspondence between two frames.

        Args:
            features1: (1, H, W, D) features from frame 1
            features2: (1, H, W, D) features from frame 2

        Returns:
            (H, W, 2) flow vectors (dx, dy)
        """
        h, w, d = features1.shape[1:4]

        # Normalize features
        f1 = F.normalize(features1[0].reshape(-1, d), dim=-1)
        f2 = F.normalize(features2[0].reshape(-1, d), dim=-1)

        # Compute all-pairs similarity
        similarity = torch.mm(f1, f2.T)  # (H*W, H*W)

        # Find best match for each pixel in frame 1
        best_matches = similarity.argmax(dim=1)  # (H*W,)

        # Convert linear indices to 2D coordinates
        match_y = best_matches // w
        match_x = best_matches % w

        # Original coordinates
        orig_y = torch.arange(h, device=features1.device).unsqueeze(1).expand(h, w).flatten()
        orig_x = torch.arange(w, device=features1.device).unsqueeze(0).expand(h, w).flatten()

        # Compute flow
        flow_y = (match_y.float() - orig_y.float()).reshape(h, w)
        flow_x = (match_x.float() - orig_x.float()).reshape(h, w)

        flow = torch.stack([flow_x, flow_y], dim=-1)

        return flow

    def upsample_features(
        self,
        features: torch.Tensor,
        target_size: tuple,
    ) -> torch.Tensor:
        """
        Upsample feature map to target size.

        Args:
            features: (1, H, W, D) feature grid
            target_size: (target_H, target_W)

        Returns:
            (1, target_H, target_W, D) upsampled features
        """
        # Rearrange to (1, D, H, W) for interpolation
        features_t = features.permute(0, 3, 1, 2)

        # Bilinear interpolation
        upsampled = F.interpolate(
            features_t,
            size=target_size,
            mode='bilinear',
            align_corners=False,
        )

        # Rearrange back to (1, H, W, D)
        return upsampled.permute(0, 2, 3, 1)
