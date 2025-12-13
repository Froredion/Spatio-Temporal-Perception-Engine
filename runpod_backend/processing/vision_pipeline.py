"""
Vision Pipeline - Unified 3-Level Processing

Combines Level 1 (Global), Level 2 (Objects), and Level 3 (Dense)
into a single unified pipeline for frame processing.
"""

import torch
from typing import Dict, List, Optional
from PIL import Image
from dataclasses import dataclass, field

from models import ModelLoader
from processing.level1_global import Level1GlobalProcessor
from processing.level2_objects import Level2ObjectProcessor, DetectedObject
from processing.level3_dense import Level3DenseProcessor


@dataclass
class FrameResult:
    """Complete processing result for a single frame."""
    frame_idx: int
    global_embedding: torch.Tensor  # (4096,) scene embedding
    patch_features: torch.Tensor  # (num_patches, 4096)
    objects: List[DetectedObject]  # Detected and embedded objects
    frame_width: int = 0  # Original image width
    frame_height: int = 0  # Original image height
    dense_features: Optional[torch.Tensor] = None  # (H, W, 4096)
    depth_cues: Optional[torch.Tensor] = None  # (H, W)
    edge_map: Optional[torch.Tensor] = None  # (H, W)
    pca_grid: Optional[torch.Tensor] = None  # (H, W, 3) visualization
    image: Optional[Image.Image] = None  # Original PIL image for R2 upload


class VisionPipeline:
    """
    Unified 3-level vision processing pipeline.

    Processes images through all three levels:
    1. Global: Scene-level embedding
    2. Objects: Auto-segmentation and object embeddings
    3. Dense: High-resolution feature maps

    Usage:
        models = ModelLoader(config)
        models.load_vision_only()

        pipeline = VisionPipeline(models)
        result = pipeline.process_frame(image, frame_idx=0)
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        enable_level3: bool = True,
        compute_pca: bool = False,
    ):
        self.models = model_loader
        self.enable_level3 = enable_level3
        self.compute_pca = compute_pca

        # Initialize processors
        self.level1 = Level1GlobalProcessor(model_loader.dinov3)
        self.level2 = Level2ObjectProcessor(
            dinov3_model=model_loader.dinov3,
            sam3_model=model_loader.sam3,
        )

        if enable_level3:
            self.level3 = Level3DenseProcessor(model_loader.dinov3)
        else:
            self.level3 = None

    @torch.no_grad()
    def process_frame(
        self,
        image: Image.Image,
        frame_idx: int = 0,
        extract_objects: bool = True,
        compute_dense: bool = True,
    ) -> FrameResult:
        """
        Process a single frame through all levels.

        Args:
            image: PIL Image to process
            frame_idx: Frame index for tracking
            extract_objects: Whether to run Level 2 object extraction
            compute_dense: Whether to run Level 3 dense processing

        Returns:
            FrameResult with all extracted features
        """
        # Level 1: Global embedding
        level1_output = self.level1.process(
            image,
            compute_pca_grid=self.compute_pca,
        )

        global_embedding = level1_output["global_embedding"].squeeze(0)
        patch_features = level1_output["patch_features"].squeeze(0)

        # Level 2: Object extraction (optional)
        objects = []
        if extract_objects:
            objects = self.level2.process(
                image,
                level1_output["patch_features"],
            )

        # Level 3: Dense features (optional)
        dense_features = None
        depth_cues = None
        edge_map = None

        if compute_dense and self.level3 is not None:
            level3_output = self.level3.process(image, high_res=True)
            dense_features = level3_output["features"].squeeze(0)
            depth_cues = level3_output["depth_cues"]
            edge_map = level3_output["edge_map"]

        # Build result
        result = FrameResult(
            frame_idx=frame_idx,
            global_embedding=global_embedding,
            patch_features=patch_features,
            objects=objects,
            frame_width=image.size[0],
            frame_height=image.size[1],
            dense_features=dense_features,
            depth_cues=depth_cues,
            edge_map=edge_map,
            pca_grid=level1_output.get("pca_grid"),
            image=image,  # Store original image for R2 upload
        )

        return result

    @torch.no_grad()
    def process_batch(
        self,
        images: List[Image.Image],
        start_idx: int = 0,
        extract_objects: bool = True,
    ) -> List[FrameResult]:
        """
        Process batch of frames.

        Args:
            images: List of PIL Images
            start_idx: Starting frame index
            extract_objects: Whether to extract objects per frame

        Returns:
            List of FrameResult for each frame
        """
        # Batch process Level 1
        level1_output = self.level1.process_batch(images)
        global_embeddings = level1_output["global_embeddings"]
        all_patch_features = level1_output["patch_features"]

        results = []
        for i, image in enumerate(images):
            frame_idx = start_idx + i

            global_emb = global_embeddings[i]
            patch_features = all_patch_features[i]

            # Level 2: Per-frame object extraction
            objects = []
            if extract_objects:
                objects = self.level2.process(
                    image,
                    patch_features.unsqueeze(0),
                )

            # Skip Level 3 for batch processing (memory intensive)
            result = FrameResult(
                frame_idx=frame_idx,
                global_embedding=global_emb,
                patch_features=patch_features,
                objects=objects,
                frame_width=image.size[0],
                frame_height=image.size[1],
                image=image,  # Store original image for R2 upload
            )
            results.append(result)

        return results

    def get_embedding_for_search(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Get embeddings optimized for search.

        Returns DINOv3 global embedding.

        Args:
            image: PIL Image

        Returns:
            Dictionary with 'dinov3' embedding
        """
        # DINOv3 global embedding
        dinov3_emb = self.level1.get_embedding_for_search(image)

        return {
            "dinov3": dinov3_emb,
        }

    def search_objects_in_image(
        self,
        image: Image.Image,
    ) -> Dict[str, any]:
        """
        Extract objects from a single image.

        Uses SAM-3 for segmentation and labeling.

        Args:
            image: PIL Image

        Returns:
            Dictionary with detected objects
        """
        # Process frame
        result = self.process_frame(image, extract_objects=True)

        # Return objects with their SAM-3 labels
        return {
            "objects": [
                {"label": obj.label, "confidence": obj.confidence, "bbox": obj.bbox}
                for obj in result.objects
            ],
        }
