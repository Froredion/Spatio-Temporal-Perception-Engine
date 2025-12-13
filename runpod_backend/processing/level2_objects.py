"""
Level 2: SAM-3 Instance Segmentation + Object Embeddings

Object-level understanding using SAM-3 for high-quality instance segmentation.
SAM-3 provides promptable segmentation with text, boxes, or points.

Key features:
- Automatic foreground segmentation via SAM-3
- High-quality masks at original resolution
- Text-prompted object detection with SAM-3 labels
- DINOv3 features for object embeddings
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from dataclasses import dataclass

from models import DINOv3Model
from models.sam3.sam3_wrapper import SAM3Model


@dataclass
class DetectedObject:
    """Represents a detected object in a frame."""
    mask: np.ndarray  # Binary mask (H, W) at original image resolution
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in IMAGE coordinates
    dinov3_embedding: torch.Tensor  # DINOv3 pooled embedding
    area: int  # Number of pixels in mask
    confidence: float  # SAM-3 segmentation confidence
    label: str = "object"  # Label from SAM-3 text-prompted detection


class Level2ObjectProcessor:
    """
    Level 2 processor using SAM-3 for instance segmentation.

    Uses SAM-3 for high-quality segmentation:
    - Automatic object detection via text prompts
    - High-resolution masks at original image size
    - DINOv3 for semantic embeddings
    - SAM-3 labels for object identification

    Algorithm:
    1. Run SAM-3 automatic segmentation with gaming/common categories
    2. Filter masks by area constraints
    3. Extract DINOv3 embeddings for each object
    4. Use SAM-3's text-prompted labels
    """

    def __init__(
        self,
        dinov3_model: DINOv3Model,
        sam3_model: SAM3Model,
        min_object_ratio: float = 0.01,  # Min 1% of image area
        max_object_ratio: float = 0.85,  # Max 85% of image (probably background)
        detection_threshold: float = 0.5,
    ):
        self.dinov3 = dinov3_model
        self.sam3 = sam3_model
        self.min_object_ratio = min_object_ratio
        self.max_object_ratio = max_object_ratio
        self.detection_threshold = detection_threshold

    @torch.no_grad()
    def process(
        self,
        image: Image.Image,
        patch_features: torch.Tensor,
        text_prompt: Optional[str] = None,
    ) -> List[DetectedObject]:
        """
        Extract and embed individual objects using SAM-3 segmentation.

        Args:
            image: Original PIL Image
            patch_features: (1, num_patches, D) from DINOv3 Level 1
            text_prompt: Optional text to guide segmentation (e.g., "person", "car")

        Returns:
            List of DetectedObject instances
        """
        img_w, img_h = image.size
        total_pixels = img_w * img_h

        # Run SAM-3 segmentation
        if text_prompt:
            segments = self.sam3.segment_with_text(
                image,
                text_prompt,
                threshold=self.detection_threshold,
            )
        else:
            segments = self.sam3.segment_automatic(
                image,
                threshold=self.detection_threshold,
            )

        if not segments:
            return []

        # Process segments and extract DINOv3 embeddings
        objects = []

        for seg in segments:
            mask = seg["mask"]
            bbox = seg["bbox"]
            confidence = seg["score"]
            label = seg.get("label", "object")  # Use SAM-3's label directly

            # Calculate area
            area = int(mask.sum())
            area_ratio = area / total_pixels

            # Filter by area ratio
            if area_ratio < self.min_object_ratio:
                continue
            if area_ratio > self.max_object_ratio:
                continue

            # Get DINOv3 embedding via masked pooling on patch features
            dinov3_emb = self._masked_pooling(patch_features, mask, img_w, img_h)

            obj = DetectedObject(
                mask=mask,
                bbox=bbox,
                dinov3_embedding=dinov3_emb,
                area=area,
                confidence=confidence,
                label=label,
            )
            objects.append(obj)

        # Sort by area descending (largest objects first)
        objects.sort(key=lambda x: x.area, reverse=True)

        return objects

    @torch.no_grad()
    def process_with_boxes(
        self,
        image: Image.Image,
        patch_features: torch.Tensor,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[DetectedObject]:
        """
        Segment objects within specified bounding boxes.

        Args:
            image: Original PIL Image
            patch_features: (1, num_patches, D) from DINOv3
            boxes: List of (x1, y1, x2, y2) bounding boxes

        Returns:
            List of DetectedObject instances
        """
        img_w, img_h = image.size
        total_pixels = img_w * img_h

        objects = []

        for box in boxes:
            segments = self.sam3.segment_with_box(
                image,
                box,
                threshold=self.detection_threshold,
            )

            for seg in segments:
                mask = seg["mask"]
                bbox = seg["bbox"]
                confidence = seg["score"]
                label = seg.get("label", "object")

                area = int(mask.sum())
                area_ratio = area / total_pixels

                if area_ratio < self.min_object_ratio:
                    continue
                if area_ratio > self.max_object_ratio:
                    continue

                dinov3_emb = self._masked_pooling(patch_features, mask, img_w, img_h)

                obj = DetectedObject(
                    mask=mask,
                    bbox=bbox,
                    dinov3_embedding=dinov3_emb,
                    area=area,
                    confidence=confidence,
                    label=label,
                )
                objects.append(obj)

        objects.sort(key=lambda x: x.area, reverse=True)
        return objects

    def _masked_pooling(
        self,
        patch_features: torch.Tensor,
        mask: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> torch.Tensor:
        """
        Pool DINOv3 patch features within the SAM-3 mask region.

        Args:
            patch_features: (1, num_patches, D) from DINOv3
            mask: Binary mask at original image resolution (H, W)
            img_w, img_h: Original image dimensions

        Returns:
            Pooled embedding (D,)
        """
        from scipy.ndimage import zoom

        features = patch_features[0]  # (num_patches, D)
        num_patches = features.shape[0]
        patch_grid_size = int(np.sqrt(num_patches))

        # Downsample mask to patch grid resolution
        mask_h, mask_w = mask.shape
        scale_y = patch_grid_size / mask_h
        scale_x = patch_grid_size / mask_w

        # Use nearest neighbor interpolation for binary mask
        mask_downsampled = zoom(mask.astype(float), (scale_y, scale_x), order=0)
        mask_flat = (mask_downsampled.flatten() > 0.5)

        # Handle potential size mismatch
        if len(mask_flat) != num_patches:
            # Resize to exact patch count
            mask_flat = zoom(mask_flat.astype(float), num_patches / len(mask_flat), order=0) > 0.5

        mask_tensor = torch.from_numpy(mask_flat.astype(bool)).to(features.device)
        masked_features = features[mask_tensor]

        if masked_features.shape[0] == 0:
            return features.mean(dim=0)

        # Mean pooling
        return masked_features.mean(dim=0)
