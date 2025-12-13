"""
STPE Pipeline - Spatio-Temporal Perception Engine

Complete pipeline that processes video through:
1. Model loading (DINOv3, SAM-3, Qwen3-VL)
2. 3-level vision processing (global, object, dense)
3. Temporal processing (tracking, motion, attention, pooling)
4. Scene graph construction
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from PIL import Image
import uuid

from config import STPEConfig, default_config
from models import ModelLoader
from processing import VisionPipeline, FrameResult
from temporal import (
    TemporalPositionalEncoding,
    DINOv3Tracker,
    MotionFeatureExtractor,
    TemporalAttention,
    TemporalPooling,
)
from temporal.scene_graph import SceneGraph
from utils import extract_frames
from captioning import DenseCaptioner, SpatialRelationExtractor
from captioning.dense_captioner import CaptionGranularity, DenseCaptionResult


@dataclass
class VideoResult:
    """Complete processing result for a video."""
    video_id: str
    frames: List[FrameResult]
    temporal_features: torch.Tensor  # (num_frames, d_model)
    motion_features: torch.Tensor  # (num_frames-1, d_model)
    clip_embedding: torch.Tensor  # (d_model,)
    scene_graph: Dict[str, Any]
    tracks: Dict[int, Any]
    captions: Optional[Dict[str, Any]] = None
    dense_captions: Optional[DenseCaptionResult] = None  # Rich per-frame captions


class STPEPipeline:
    """
    Spatio-Temporal Perception Engine - Complete Pipeline.

    Processes videos through the full STPE stack:
    - DINOv3-7B for hierarchical vision
    - SAM-3 for segmentation and object labels
    - Qwen3-VL for reasoning
    - Custom temporal modules

    Usage:
        pipeline = STPEPipeline()
        pipeline.load_models()

        result = pipeline.process_video("path/to/video.mp4")

        # Visual search
        matches = pipeline.search_visual(query_image)

        pipeline.unload_models()
    """

    def __init__(
        self,
        config: Optional[STPEConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or default_config
        self.device = device if torch.cuda.is_available() else "cpu"

        # Model loader
        self.models: Optional[ModelLoader] = None

        # Vision pipeline
        self.vision_pipeline: Optional[VisionPipeline] = None

        # Temporal modules
        self.temporal_pos = TemporalPositionalEncoding(
            d_model=self.config.temporal.d_model,
            max_frames=self.config.temporal.max_frames,
        ).to(self.device)

        self.temporal_attention = TemporalAttention(
            d_model=self.config.temporal.d_model,
            nhead=self.config.temporal.num_attention_heads,
            num_layers=self.config.temporal.num_attention_layers,
        ).to(self.device)

        self.motion_extractor = MotionFeatureExtractor(
            feature_dim=self.config.temporal.d_model,
        ).to(self.device)

        self.temporal_pooling = TemporalPooling(
            d_model=self.config.temporal.d_model,
        ).to(self.device)

        self.tracker = DINOv3Tracker(
            similarity_threshold=self.config.temporal.similarity_threshold,
        )

        self.scene_graph = SceneGraph(
            node_dim=self.config.scene_graph.node_dim,
            edge_dim=self.config.scene_graph.edge_dim,
            num_layers=self.config.scene_graph.num_gnn_layers,
        ).to(self.device)

        self._models_loaded = False
        self._vision_only = False

    def load_models(self, vision_only: bool = False):
        """
        Load all models (optional - models also lazy-load on first use).

        Args:
            vision_only: If True, skip loading Qwen3-VL (for embedding-only tasks)
        """
        self._vision_only = vision_only
        self._ensure_models_initialized()

        if vision_only:
            self.models.load_vision_only()
        else:
            self.models.load_all()

        self._models_loaded = True

    def _ensure_models_initialized(self):
        """Ensure ModelLoader is created (but not necessarily loaded)."""
        if self.models is None:
            self.models = ModelLoader(self.config, self.device)

    def _ensure_vision_pipeline(self):
        """Lazy-init vision pipeline."""
        if self.vision_pipeline is None:
            print("[Pipeline] Creating VisionPipeline...")
            self._ensure_models_initialized()
            # Ensure vision models are loaded
            if self.models.dinov3 is None:
                print("[Pipeline] DINOv3 not loaded, loading now...")
                self.models.load_dinov3()
            else:
                print("[Pipeline] DINOv3 already loaded")
            self.vision_pipeline = VisionPipeline(self.models)
            print("[Pipeline] VisionPipeline created")
        else:
            print("[Pipeline] VisionPipeline already exists")

    def unload_models(self):
        """Unload all models."""
        if self.models:
            self.models.unload_all()
        self._models_loaded = False

    def clear_caches(self):
        """
        Clear all cached data to free memory.

        Called after each processing request since R2 is the source of truth.
        Model weights are preserved - only intermediate results are cleared.
        """
        from utils.memory import full_cleanup, log_memory_usage

        log_memory_usage("Before clear_caches")

        # Clear tracker (releases all track tensors)
        self.tracker.clear_tracks()

        # Full cleanup (GC + GPU cache)
        full_cleanup()

        log_memory_usage("After clear_caches")

    def cleanup_after_request(self):
        """
        Full cleanup after each processing request.

        Since data is saved to R2, we can aggressively clear everything.
        Only model weights are preserved.
        """
        self.clear_caches()

    @torch.no_grad()
    def process_video(
        self,
        video_path: str,
        fps: float = 2.0,
        generate_captions: bool = True,
        dense_captions: bool = True,
        caption_interval: int = 1,
        caption_granularities: Optional[List[str]] = None,
        processing_resolution: Optional[Tuple[int, int]] = None,
    ) -> VideoResult:
        """
        Process entire video through STPE pipeline.

        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            generate_captions: Whether to generate captions with Qwen3-VL
            dense_captions: Whether to generate dense per-frame captions
            caption_interval: Generate captions every N frames (1 = every frame)
            caption_granularities: List of granularity levels ['brief', 'normal', 'detailed']
            processing_resolution: Optional (width, height) to resize frames before processing.
                                   E.g., (1280, 720) for 720p, (854, 480) for 480p.
                                   Lower resolution = faster processing but less detail.

        Returns:
            VideoResult with all extracted features
        """
        start_time = time.time()
        print(f"[Pipeline] process_video started: {video_path}")

        # Lazy-load required components
        print("[Pipeline] Ensuring vision pipeline...")
        self._ensure_vision_pipeline()
        print("[Pipeline] Vision pipeline ready")

        video_id = str(uuid.uuid4())[:8]
        print(f"[Pipeline] Video ID: {video_id}")

        # Extract frames (with optional resize for faster processing)
        if processing_resolution:
            print(f"[Pipeline] Extracting frames at {fps} fps, resizing to {processing_resolution[0]}x{processing_resolution[1]}...")
        else:
            print(f"[Pipeline] Extracting frames at {fps} fps...")
        frames = extract_frames(video_path, fps, output_size=processing_resolution)
        print(f"[Pipeline] Extracted {len(frames)} frames")

        # Process frames through vision pipeline (batched for GPU efficiency)
        print("[Pipeline] Processing frames through vision pipeline...")
        frame_results = []
        frame_embeddings = []

        self.tracker.reset()

        # Batch size for GPU processing (adjust based on VRAM)
        batch_size = self.config.processing.batch_size if hasattr(self.config.processing, 'batch_size') else 4

        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            print(f"[Pipeline] Processing batch {batch_start//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size} (frames {batch_start+1}-{batch_end}/{len(frames)})...")

            # Process batch through Level 1 + Level 2 (batched)
            batch_results = self.vision_pipeline.process_batch(
                batch_frames,
                start_idx=batch_start,
                extract_objects=True,
            )

            # Add Level 3 dense features for selected frames
            for i, result in enumerate(batch_results):
                frame_idx = batch_start + i
                if frame_idx % 5 == 0 and self.vision_pipeline.level3 is not None:
                    # Compute dense features for every 5th frame
                    level3_output = self.vision_pipeline.level3.process(batch_frames[i], high_res=True)
                    result.dense_features = level3_output["features"].squeeze(0)
                    result.depth_cues = level3_output["depth_cues"]
                    result.edge_map = level3_output["edge_map"]

                frame_results.append(result)
                frame_embeddings.append(result.global_embedding)

                # Track objects (include SAM-3 label for identification)
                objects_for_tracking = [
                    {
                        'dinov3_embedding': obj.dinov3_embedding,
                        'label': obj.label,  # SAM-3 label
                        'bbox': obj.bbox,
                        'confidence': obj.confidence,
                    }
                    for obj in result.objects
                ]
                self.tracker.update(frame_idx, objects_for_tracking)

        # Stack frame embeddings
        print("[Pipeline] Stacking frame embeddings...")
        frame_embeddings = torch.stack(frame_embeddings)  # (num_frames, d_model)
        # Convert to float32 for temporal modules (they don't support fp16)
        frame_embeddings = frame_embeddings.float()
        frame_indices = torch.arange(len(frames), device=self.device)

        # Temporal processing
        print("[Pipeline] Running temporal attention...")
        temporal_features = self.temporal_attention(
            frame_embeddings.unsqueeze(0),
            frame_indices.unsqueeze(0),
        ).squeeze(0)

        # Motion features
        print("[Pipeline] Extracting motion features...")
        motion_features = self.motion_extractor(
            frame_embeddings.unsqueeze(0)
        ).squeeze(0)

        # Clip-level embedding
        print("[Pipeline] Computing clip embedding...")
        clip_embedding = self.temporal_pooling(
            temporal_features.unsqueeze(0)
        ).squeeze(0)

        # Scene graph
        print("[Pipeline] Building scene graph...")
        track_features = {}
        for track_id, track in self.tracker.tracks.items():
            # Convert embeddings to float32 for scene graph module
            embeddings = [entry.embedding.float() for entry in track.entries]
            track_features[track_id] = embeddings

        scene_graph_output = self.scene_graph(track_features)

        # Generate captions if requested
        captions = None
        dense_caption_result = None

        if generate_captions or dense_captions:
            print("[Pipeline] Generating captions...")
            # Lazy-load Qwen3-VL for captioning
            if self.models.vlm is None and not self._vision_only:
                print("[Pipeline] Loading Qwen3-VL for captioning...")
                self.models.load_vlm()

            if self.models.vlm is not None:
                # Dense per-frame captioning (new system)
                if dense_captions:
                    print(f"[Pipeline] Generating dense captions (interval={caption_interval})...")
                    dense_caption_result = self._generate_dense_captions(
                        frames=frames,
                        frame_results=frame_results,
                        tracks=self.tracker.tracks,
                        fps=fps,
                        video_id=video_id,
                        caption_interval=caption_interval,
                        granularities=caption_granularities,
                    )
                    print(f"[Pipeline] Generated {len(dense_caption_result.frame_captions)} dense frame captions")

                    # Also populate legacy captions from dense captions for backwards compatibility
                    captions = self._dense_to_legacy_captions(dense_caption_result)

                # Legacy captioning (if dense captions not requested but generate_captions is)
                elif generate_captions:
                    captions = self._generate_captions(
                        frames=frames,
                        frame_results=frame_results,
                        motion_features=motion_features,
                        tracks=self.tracker.tracks,
                        fps=fps,
                    )
                    print(f"[Pipeline] Generated captions with {len(captions.get('events', []))} events")
            else:
                print("[Pipeline] VLM not available, skipping captions")

        # Build result
        print("[Pipeline] Building final result...")
        result = VideoResult(
            video_id=video_id,
            frames=frame_results,
            temporal_features=temporal_features,
            motion_features=motion_features,
            clip_embedding=clip_embedding,
            scene_graph=scene_graph_output,
            tracks=self.tracker.tracks,
            captions=captions,
            dense_captions=dense_caption_result,
        )

        elapsed = time.time() - start_time
        print(f"[Pipeline] process_video completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return result

    @torch.no_grad()
    def process_frames(
        self,
        frames: List[Image.Image],
        fps: float = 10.0,
        generate_captions: bool = True,
        dense_captions: bool = True,
        caption_interval: int = 1,
        caption_granularities: Optional[List[str]] = None,
        frame_metadata: Optional[Dict[int, Dict]] = None,
        processing_resolution: Optional[Tuple[int, int]] = None,
    ) -> VideoResult:
        """
        Process pre-extracted frames (no video file needed).

        Same processing as process_video but accepts frames directly.
        Optionally includes per-frame metadata for enhanced captioning.

        Args:
            frames: List of PIL Images
            fps: Assumed FPS for temporal calculations
            generate_captions: Whether to generate captions with Qwen3-VL
            dense_captions: Whether to generate dense per-frame captions
            caption_interval: Generate captions every N frames
            caption_granularities: List of ['brief', 'normal', 'detailed']
            frame_metadata: Optional dict mapping frame_number (1-indexed) -> metadata dict.
                           When provided, metadata is passed to VLM prompts for
                           ground-truth context (e.g., Roblox game state).
            processing_resolution: Optional (width, height) to resize frames before processing.
                                   E.g., (1280, 720) for 720p, (854, 480) for 480p.
                                   Lower resolution = faster processing but less detail.

        Returns:
            VideoResult with all extracted features
        """
        start_time = time.time()

        # Resize frames if processing_resolution specified
        if processing_resolution:
            print(f"[Pipeline] Resizing {len(frames)} frames to {processing_resolution[0]}x{processing_resolution[1]}...")
            frames = [frame.resize(processing_resolution, Image.LANCZOS) for frame in frames]

        print(f"[Pipeline] process_frames started: {len(frames)} frames at {fps} fps")

        # Lazy-load required components
        print("[Pipeline] Ensuring vision pipeline...")
        self._ensure_vision_pipeline()
        print("[Pipeline] Vision pipeline ready")

        video_id = str(uuid.uuid4())[:8]
        print(f"[Pipeline] Video ID: {video_id}")

        # Process frames through vision pipeline (batched for GPU efficiency)
        print("[Pipeline] Processing frames through vision pipeline...")
        frame_results = []
        frame_embeddings = []

        self.tracker.reset()

        # Batch size for GPU processing
        batch_size = self.config.processing.batch_size if hasattr(self.config.processing, 'batch_size') else 4

        # Process frames in batches
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            print(f"[Pipeline] Processing batch {batch_start//batch_size + 1}/{(len(frames) + batch_size - 1)//batch_size} (frames {batch_start+1}-{batch_end}/{len(frames)})...")

            # Process batch through Level 1 + Level 2 (batched)
            batch_results = self.vision_pipeline.process_batch(
                batch_frames,
                start_idx=batch_start,
                extract_objects=True,
            )

            # Add Level 3 dense features for selected frames
            for i, result in enumerate(batch_results):
                frame_idx = batch_start + i
                if frame_idx % 5 == 0 and self.vision_pipeline.level3 is not None:
                    level3_output = self.vision_pipeline.level3.process(batch_frames[i], high_res=True)
                    result.dense_features = level3_output["features"].squeeze(0)
                    result.depth_cues = level3_output["depth_cues"]
                    result.edge_map = level3_output["edge_map"]

                frame_results.append(result)
                frame_embeddings.append(result.global_embedding)

                # Track objects
                objects_for_tracking = [
                    {
                        'dinov3_embedding': obj.dinov3_embedding,
                        'label': obj.label,
                        'bbox': obj.bbox,
                        'confidence': obj.confidence,
                    }
                    for obj in result.objects
                ]
                self.tracker.update(frame_idx, objects_for_tracking)

        # Stack frame embeddings
        print("[Pipeline] Stacking frame embeddings...")
        frame_embeddings = torch.stack(frame_embeddings)
        frame_embeddings = frame_embeddings.float()
        frame_indices = torch.arange(len(frames), device=self.device)

        # Temporal processing
        print("[Pipeline] Running temporal attention...")
        temporal_features = self.temporal_attention(
            frame_embeddings.unsqueeze(0),
            frame_indices.unsqueeze(0),
        ).squeeze(0)

        # Motion features
        print("[Pipeline] Extracting motion features...")
        motion_features = self.motion_extractor(
            frame_embeddings.unsqueeze(0)
        ).squeeze(0)

        # Clip-level embedding
        print("[Pipeline] Computing clip embedding...")
        clip_embedding = self.temporal_pooling(
            temporal_features.unsqueeze(0)
        ).squeeze(0)

        # Scene graph
        print("[Pipeline] Building scene graph...")
        track_features = {}
        for track_id, track in self.tracker.tracks.items():
            embeddings = [entry.embedding.float() for entry in track.entries]
            track_features[track_id] = embeddings

        scene_graph_output = self.scene_graph(track_features)

        # Generate captions if requested
        captions = None
        dense_caption_result = None

        if generate_captions or dense_captions:
            print("[Pipeline] Generating captions...")
            if self.models.vlm is None and not self._vision_only:
                print("[Pipeline] Loading Qwen3-VL for captioning...")
                self.models.load_vlm()

            if self.models.vlm is not None:
                if dense_captions:
                    print(f"[Pipeline] Generating dense captions (interval={caption_interval})...")
                    dense_caption_result = self._generate_dense_captions(
                        frames=frames,
                        frame_results=frame_results,
                        tracks=self.tracker.tracks,
                        fps=fps,
                        video_id=video_id,
                        caption_interval=caption_interval,
                        granularities=caption_granularities,
                        frame_metadata=frame_metadata,
                    )
                    print(f"[Pipeline] Generated {len(dense_caption_result.frame_captions)} dense frame captions")
                    captions = self._dense_to_legacy_captions(dense_caption_result)

                elif generate_captions:
                    captions = self._generate_captions(
                        frames=frames,
                        frame_results=frame_results,
                        motion_features=motion_features,
                        tracks=self.tracker.tracks,
                        fps=fps,
                    )
                    print(f"[Pipeline] Generated captions with {len(captions.get('events', []))} events")
            else:
                print("[Pipeline] VLM not available, skipping captions")

        # Build result
        print("[Pipeline] Building final result...")
        result = VideoResult(
            video_id=video_id,
            frames=frame_results,
            temporal_features=temporal_features,
            motion_features=motion_features,
            clip_embedding=clip_embedding,
            scene_graph=scene_graph_output,
            tracks=self.tracker.tracks,
            captions=captions,
            dense_captions=dense_caption_result,
        )

        elapsed = time.time() - start_time
        print(f"[Pipeline] process_frames completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return result

    def _generate_dense_captions(
        self,
        frames: List[Image.Image],
        frame_results: List[FrameResult],
        tracks: Dict[int, Any],
        fps: float,
        video_id: str,
        caption_interval: int = 1,
        granularities: Optional[List[str]] = None,
        frame_metadata: Optional[Dict[int, Dict]] = None,
    ) -> DenseCaptionResult:
        """
        Generate dense per-frame captions with multi-granularity support.

        Args:
            frames: List of PIL Images
            frame_results: List of FrameResult from vision pipeline
            tracks: Dict of Track objects from tracker
            fps: Frames per second
            video_id: Video identifier
            caption_interval: Caption every N frames (1 = every frame)
            granularities: List of ['brief', 'normal', 'detailed']
            frame_metadata: Optional dict mapping frame_number (1-indexed) -> metadata dict.
                           When provided, metadata is included in VLM prompts for
                           ground-truth context (e.g., Roblox game state).

        Returns:
            DenseCaptionResult with all frame captions
        """
        # Parse granularity strings to enums
        granularity_map = {
            'brief': CaptionGranularity.BRIEF,
            'normal': CaptionGranularity.NORMAL,
            'detailed': CaptionGranularity.DETAILED,
        }

        if granularities:
            parsed_granularities = [
                granularity_map[g.lower()]
                for g in granularities
                if g.lower() in granularity_map
            ]
        else:
            # Default: detailed only (richest description for training data)
            parsed_granularities = [CaptionGranularity.DETAILED]

        # Create dense captioner
        captioner = DenseCaptioner(
            vlm_model=self.models.vlm,
            caption_interval=caption_interval,
            batch_size=12,  # 12 frames * 3 granularities = 36 items per mega-batch
        )

        # Generate dense captions
        result = captioner.generate_dense_captions(
            frames=frames,
            frame_results=frame_results,
            tracks=tracks,
            fps=fps,
            video_id=video_id,
            granularities=parsed_granularities,
            frame_metadata=frame_metadata,
        )

        return result

    def _dense_to_legacy_captions(
        self,
        dense_result: DenseCaptionResult,
    ) -> Dict[str, Any]:
        """
        Convert dense captions to legacy format for backwards compatibility.

        Args:
            dense_result: DenseCaptionResult from dense captioner

        Returns:
            Legacy captions dict with events, tracked_objects, summary
        """
        events = []
        for fc in dense_result.frame_captions:
            events.append({
                "timestamp": fc.timestamp,
                "frame_idx": fc.frame_idx,
                "action": fc.normal or fc.brief,  # Use normal caption, fallback to brief
                "active_tracks": len(fc.active_tracks),
            })

        # Extract tracked objects from frame captions
        tracked_objects = []
        seen_tracks = set()
        for fc in dense_result.frame_captions:
            for track in fc.active_tracks:
                track_id = track.get("track_id")
                if track_id not in seen_tracks:
                    seen_tracks.add(track_id)
                    tracked_objects.append({
                        "track_id": track_id,
                        "label": track.get("label", "object"),
                        "movement_type": track.get("movement_type", "unknown"),
                        "primary_direction": track.get("direction", "unknown"),
                    })

        return {
            "events": events,
            "tracked_objects": tracked_objects,
            "summary": dense_result.summary,
            "dense_captions": dense_result.to_dict(),  # Include full dense data
        }

    def _generate_captions(
        self,
        frames: List[Image.Image],
        frame_results: List[FrameResult],
        motion_features: torch.Tensor,
        tracks: Dict[int, Any],
        fps: float,
    ) -> Dict[str, Any]:
        """
        Generate captions for every 5 frames throughout the video.

        Processes frames in groups of 5, using each group's context to describe
        what's happening. Enables parallel batch processing for efficiency.
        """
        result = {
            "events": [],
            "tracked_objects": [],
            "summary": "",
        }

        # 1. Analyze tracked objects with motion data
        print("[Pipeline] Analyzing tracked objects with motion...")
        tracked_objects = self._analyze_tracks(tracks, frames, fps)
        result["tracked_objects"] = tracked_objects
        print(f"[Pipeline] Tracked {len(tracked_objects)} objects with motion analysis")

        # 2. Generate captions for every 5 frames (batched for parallel processing)
        caption_interval = 5
        caption_frames = list(range(caption_interval - 1, len(frames), caption_interval))

        # If no frames meet the interval, at least caption the last frame
        if not caption_frames and len(frames) > 0:
            caption_frames = [len(frames) - 1]

        print(f"[Pipeline] Generating captions for {len(caption_frames)} frame groups (every {caption_interval} frames)...")

        if caption_frames and self.models.vlm is not None:
            # Prepare batch items for all caption points
            batch_items = []
            caption_metadata = []

            for target_frame_idx in caption_frames:
                timestamp = target_frame_idx / fps

                # Get the previous 5 frames (or fewer if at start of video)
                start_idx = max(0, target_frame_idx - caption_interval + 1)
                context_indices = list(range(start_idx, target_frame_idx + 1))

                # Limit to 4 frames for VLM (select evenly spaced if more)
                if len(context_indices) > 4:
                    step = len(context_indices) / 4
                    context_indices = [context_indices[int(i * step)] for i in range(4)]

                context_frames = [frames[i] for i in context_indices]

                # Find active tracks at this moment
                active_tracks = self._get_active_tracks_at_frame(
                    tracked_objects, target_frame_idx, fps
                )

                # Get per-frame object detections for context frames
                frame_objects = self._get_frame_objects(frame_results, context_indices)

                # Build prompt with context from the frame group
                action_prompt = self._build_frame_group_prompt(
                    active_tracks, timestamp, frame_objects, context_indices, fps
                )

                batch_items.append({
                    'images': context_frames,
                    'prompt': action_prompt,
                })
                caption_metadata.append({
                    'timestamp': timestamp,
                    'frame_idx': target_frame_idx,
                    'context_start': start_idx,
                    'context_end': target_frame_idx,
                    'active_tracks': len(active_tracks),
                })

            # Process in optimal batch sizes for parallel GPU utilization
            max_batch_size = 8  # Adjust based on VRAM
            all_responses = []

            for batch_start in range(0, len(batch_items), max_batch_size):
                batch_end = min(batch_start + max_batch_size, len(batch_items))
                current_batch = batch_items[batch_start:batch_end]

                print(f"[Pipeline] Processing caption batch {batch_start//max_batch_size + 1}/{(len(batch_items) + max_batch_size - 1)//max_batch_size}...")

                batch_responses = self.models.vlm.generate_batch(
                    current_batch,
                    temperature=0.3,
                    max_new_tokens=150,
                )
                all_responses.extend(batch_responses)

            # Build events from responses
            for i, action in enumerate(all_responses):
                meta = caption_metadata[i]
                result["events"].append({
                    "timestamp": round(meta['timestamp'], 1),
                    "frame_idx": meta['frame_idx'],
                    "context_range": f"{meta['context_start']}-{meta['context_end']}",
                    "action": action.strip() if action else "Scene captured",
                    "active_tracks": meta['active_tracks'],
                })

        # 3. Generate overall summary using frames + generated captions as context
        print("[Pipeline] Generating video summary with caption context...")
        if len(frames) >= 4 and self.models.vlm is not None:
            # Spread 4 frames evenly across the video
            summary_indices = [
                0,
                len(frames) // 3,
                2 * len(frames) // 3,
                len(frames) - 1,
            ]

            summary_frames = [frames[i] for i in summary_indices]

            # Build summary prompt using the generated segment captions
            summary_prompt = self._build_summary_prompt_with_captions(
                result["events"], tracked_objects, len(frames), fps
            )

            result["summary"] = self.models.vlm.generate(
                summary_frames,
                summary_prompt,
                temperature=0.3,
                max_new_tokens=300,
            ).strip()

        return result

    def _build_frame_group_prompt(
        self,
        active_tracks: List[Dict[str, Any]],
        timestamp: float,
        frame_objects: Optional[List[Dict[str, Any]]] = None,
        context_indices: Optional[List[int]] = None,
        fps: float = 2.0,
    ) -> str:
        """
        Build a prompt for describing what happens across a group of frames.

        Focuses on detailed but reliable observations - what can actually be seen.

        Args:
            active_tracks: Tracked objects with motion analysis
            timestamp: End timestamp of this frame group
            frame_objects: Per-frame detected objects with bbox and label
            context_indices: Frame indices in this group
            fps: Frames per second
        """
        # Calculate time range
        if context_indices and len(context_indices) > 1:
            start_time = context_indices[0] / fps
            time_range = f"{start_time:.1f}s - {timestamp:.1f}s"
        else:
            time_range = f"{timestamp:.1f}s"

        context_parts = []

        # Add per-frame object detections with spatial info
        if frame_objects:
            obj_descriptions = []
            seen_labels = set()
            for obj in frame_objects[:8]:  # Top 8 objects
                label = obj.get("label", "object")
                bbox = obj.get("bbox")
                confidence = obj.get("confidence", 0)

                if bbox and confidence > 0.3 and label not in seen_labels:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    width = obj.get("frame_width", 1920)
                    height = obj.get("frame_height", 1080)

                    h_pos = "left" if cx < width * 0.33 else "right" if cx > width * 0.67 else "center"
                    v_pos = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.67 else "middle"

                    obj_descriptions.append(f"{label} ({h_pos}-{v_pos})")
                    seen_labels.add(label)

            if obj_descriptions:
                context_parts.append(f"Detected objects: {', '.join(obj_descriptions)}.")

        # Add motion context with object identities
        if active_tracks:
            motion_hints = []
            for track in active_tracks[:4]:  # Top 4 active tracks
                label = track.get("label", "object")
                direction = track.get("primary_direction", "")
                movement = track.get("movement_type", "")

                if movement and movement != "stationary":
                    motion_hints.append(f"{label} moving {direction} ({movement})")

            if motion_hints:
                context_parts.append(f"Motion tracked: {', '.join(motion_hints)}.")

        context = " ".join(context_parts) + " " if context_parts else ""

        prompt = f"""Analyze this video segment ({time_range}).
{context}
Describe in detail:
1. What objects/people are visible and where they are positioned
2. What actions or movements are occurring (be specific about direction and manner)
3. Any interactions between objects/people
4. Changes from the first frame to the last frame

Only describe what you can directly observe. Do not guess or assume. Be detailed but factual."""

        return prompt

    def _build_motion_context(
        self,
        tracked_objects: List[Dict[str, Any]],
        key_moments: List[Dict[str, Any]],
    ) -> str:
        """Build a motion context string for VLM prompts."""
        if not tracked_objects:
            return "No significant object motion detected."

        lines = []

        # Summarize moving objects with labels
        moving = [t for t in tracked_objects if t["movement_type"] != "stationary"]
        stationary = [t for t in tracked_objects if t["movement_type"] == "stationary"]

        if moving:
            lines.append(f"Detected {len(moving)} moving object(s):")
            for obj in moving[:5]:  # Top 5
                label = obj.get("label", "object")
                direction = obj["primary_direction"]
                speed_type = obj["movement_type"]
                lines.append(f"  - {label} moving {direction} ({speed_type})")

        if stationary and len(stationary) <= 3:
            labels = [t.get("label", "object") for t in stationary]
            lines.append(f"Stationary: {', '.join(labels)}")

        if key_moments:
            lines.append(f"Key action moments at: {', '.join(f'{m['timestamp']:.1f}s' for m in key_moments)}")

        return "\n".join(lines)

    def _get_active_tracks_at_frame(
        self,
        tracked_objects: List[Dict[str, Any]],
        frame_idx: int,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """Get tracks that are active at a specific frame."""
        timestamp = frame_idx / fps
        active = []

        for obj in tracked_objects:
            if obj["start_time"] <= timestamp <= obj["end_time"]:
                active.append(obj)

        return active

    def _get_frame_objects(
        self,
        frame_results: List["FrameResult"],
        frame_indices: List[int],
    ) -> List[Dict[str, Any]]:
        """
        Extract object detections (bbox, label, confidence) from specific frames.

        Args:
            frame_results: List of FrameResult from vision pipeline
            frame_indices: Which frame indices to extract objects from

        Returns:
            List of dicts with bbox, label, confidence, frame_width, frame_height
        """
        objects = []

        for idx in frame_indices:
            if idx < 0 or idx >= len(frame_results):
                continue

            frame_result = frame_results[idx]
            frame_width = frame_result.frame_width
            frame_height = frame_result.frame_height

            for obj in frame_result.objects:
                objects.append({
                    "label": obj.label,
                    "bbox": obj.bbox,
                    "confidence": obj.confidence,
                    "area": obj.area,
                    "frame_idx": idx,
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                })

        # Sort by confidence descending
        objects.sort(key=lambda x: x["confidence"], reverse=True)

        return objects

    def _build_action_prompt(
        self,
        active_tracks: List[Dict[str, Any]],
        timestamp: float,
        frame_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a motion-aware prompt for action captioning.

        Args:
            active_tracks: Tracked objects with motion analysis
            timestamp: Current timestamp in seconds
            frame_objects: Per-frame detected objects with bbox and label
        """
        base_prompt = "What specific action or event is happening at this moment?"

        context_parts = []

        # Add per-frame object detections with spatial info
        if frame_objects:
            obj_descriptions = []
            for obj in frame_objects[:5]:  # Top 5 objects
                label = obj.get("label", "object")
                bbox = obj.get("bbox")
                confidence = obj.get("confidence", 0)

                if bbox and confidence > 0.3:
                    x1, y1, x2, y2 = bbox
                    # Describe position in image (left/right/center, top/bottom)
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    width = obj.get("frame_width", 1920)
                    height = obj.get("frame_height", 1080)

                    h_pos = "left" if cx < width * 0.33 else "right" if cx > width * 0.67 else "center"
                    v_pos = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.67 else "middle"

                    obj_descriptions.append(f"{label} at {h_pos}-{v_pos}")
                else:
                    obj_descriptions.append(label)

            if obj_descriptions:
                context_parts.append(f"Detected objects: {', '.join(obj_descriptions)}.")

        # Add motion context with object identities
        if active_tracks:
            motion_hints = []
            for track in active_tracks[:3]:  # Top 3 active tracks
                label = track.get("label", "object")
                direction = track.get("primary_direction", "")
                movement = track.get("movement_type", "")
                bbox = track.get("bbox")

                motion_desc = f"{label}"
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    # Add rough position
                    width = 1920  # Default, will be overridden if available
                    h_pos = "left" if cx < width * 0.33 else "right" if cx > width * 0.67 else "center"
                    motion_desc = f"{label} ({h_pos})"

                if movement and movement != "stationary":
                    motion_hints.append(f"{motion_desc} moving {direction} ({movement})")
                else:
                    motion_hints.append(f"{motion_desc} (stationary)")

            if motion_hints:
                context_parts.append(f"Motion tracking: {', '.join(motion_hints)}.")

        if context_parts:
            context = " ".join(context_parts) + " "
            return f"{context}{base_prompt} Describe what these objects are doing. Be specific and concise."
        else:
            return f"{base_prompt} Focus on movement, interactions, and changes. Be concise."

    def _build_summary_prompt(
        self,
        motion_context: str,
        tracked_objects: List[Dict[str, Any]],
        frame_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build a motion-aware prompt for video summary.

        Args:
            motion_context: Text summary of motion data
            tracked_objects: Tracked objects with motion analysis
            frame_objects: Per-frame detected objects with bbox and label
        """
        # Count movement types
        fast = sum(1 for t in tracked_objects if t["movement_type"] == "fast")
        moderate = sum(1 for t in tracked_objects if t["movement_type"] == "moderate")
        slow = sum(1 for t in tracked_objects if t["movement_type"] == "slow")

        activity_level = "high-action" if fast > 0 else "moderate" if moderate > 0 else "calm"

        # Build object detection context
        object_context = ""
        if frame_objects:
            # Get unique labels with their positions
            unique_objects = {}
            for obj in frame_objects:
                label = obj.get("label", "object")
                if label not in unique_objects:
                    unique_objects[label] = obj

            if unique_objects:
                obj_list = []
                for label, obj in list(unique_objects.items())[:6]:
                    bbox = obj.get("bbox")
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2
                        width = obj.get("frame_width", 1920)
                        h_pos = "left" if cx < width * 0.33 else "right" if cx > width * 0.67 else "center"
                        obj_list.append(f"{label} ({h_pos})")
                    else:
                        obj_list.append(label)
                object_context = f"Detected objects in frames: {', '.join(obj_list)}. "

        prompt = (
            f"This appears to be a {activity_level} video sequence. "
            f"{object_context}"
            f"Motion tracking detected: {fast} fast-moving, {moderate} moderate, {slow} slow objects. "
            "Analyze these frames and provide a brief summary of: "
            "1) What is happening (the main activity/event) "
            "2) How objects/subjects are moving (directions, speeds) "
            "3) The outcome or conclusion. "
            "Be specific about movement and actions."
        )

        return prompt

    def _build_summary_prompt_with_captions(
        self,
        events: List[Dict[str, Any]],
        tracked_objects: List[Dict[str, Any]],
        total_frames: int,
        fps: float,
    ) -> str:
        """
        Build a summary prompt that uses generated segment captions as context.

        Args:
            events: List of generated segment captions with timestamps
            tracked_objects: Tracked objects with motion analysis
            total_frames: Total number of frames in video
            fps: Frames per second
        """
        video_duration = total_frames / fps

        # Build timeline from segment captions
        caption_timeline = []
        for event in events:
            timestamp = event.get("timestamp", 0)
            action = event.get("action", "")
            if action:
                caption_timeline.append(f"[{timestamp:.1f}s] {action}")

        # Build tracked objects summary
        object_summary = []
        for obj in tracked_objects[:5]:  # Top 5 tracked objects
            label = obj.get("label", "object")
            movement = obj.get("movement_type", "")
            direction = obj.get("primary_direction", "")
            duration = obj.get("duration", 0)

            if movement != "stationary":
                object_summary.append(f"{label}: {movement} movement {direction} (visible {duration:.1f}s)")
            else:
                object_summary.append(f"{label}: stationary (visible {duration:.1f}s)")

        # Construct prompt
        prompt_parts = [
            f"Video duration: {video_duration:.1f} seconds.",
            "",
            "Segment-by-segment analysis:",
        ]

        if caption_timeline:
            prompt_parts.extend(caption_timeline)
        else:
            prompt_parts.append("(No segment captions available)")

        if object_summary:
            prompt_parts.append("")
            prompt_parts.append("Tracked objects:")
            prompt_parts.extend([f"- {s}" for s in object_summary])

        prompt_parts.extend([
            "",
            "Based on the segment analysis above and these sample frames, provide a comprehensive summary of the entire video.",
            "Synthesize the sequence of events into a coherent narrative.",
            "Focus on: what happens, who/what is involved, how it progresses, and the outcome.",
            "Be concise but complete."
        ])

        return "\n".join(prompt_parts)

    def _detect_key_moments(
        self,
        motion_features: torch.Tensor,
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Detect key moments from motion features.

        High motion magnitude = something interesting happening.
        """
        if motion_features.shape[0] == 0:
            return []

        # Compute motion magnitude for each frame transition
        motion_magnitude = torch.norm(motion_features, dim=-1).cpu().numpy()

        # Normalize
        if motion_magnitude.max() > 0:
            motion_magnitude = motion_magnitude / motion_magnitude.max()

        # Find peaks (moments with high motion)
        # Use adaptive threshold based on mean + std
        threshold = motion_magnitude.mean() + 0.5 * motion_magnitude.std()
        threshold = max(threshold, 0.3)  # Minimum threshold

        key_moments = []
        min_gap = int(fps * 2)  # Minimum 2 seconds between key moments
        last_key_frame = -min_gap

        for i, magnitude in enumerate(motion_magnitude):
            if magnitude > threshold and (i - last_key_frame) >= min_gap:
                key_moments.append({
                    "frame_idx": i + 1,  # +1 because motion is between frames
                    "timestamp": (i + 1) / fps,
                    "intensity": float(magnitude),
                })
                last_key_frame = i

        # Limit to top 5 key moments
        key_moments = sorted(key_moments, key=lambda x: x["intensity"], reverse=True)[:5]
        key_moments = sorted(key_moments, key=lambda x: x["timestamp"])

        return key_moments

    def _analyze_tracks(
        self,
        tracks: Dict[int, Any],
        frames: List[Image.Image],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Analyze tracked objects with motion data.

        Uses SAM-3 labels for object identification.

        Returns track info including:
        - Object identity (SAM-3 label)
        - Timing (start/end/duration)
        - Motion analysis (speed, direction, movement type)
        - Position info (start/end positions)
        """
        tracked_objects = []

        for track_id, track in tracks.items():
            if not hasattr(track, 'entries') or len(track.entries) < 3:
                continue  # Skip very short tracks

            # Get track info
            start_frame = track.entries[0].frame_idx
            end_frame = track.entries[-1].frame_idx
            duration = (end_frame - start_frame) / fps

            # Skip very short duration tracks
            if duration < 0.5:
                continue

            # Get object label from SAM-3 (majority vote across track entries)
            label = track.get_majority_label()

            # Get motion analysis from DINOv3 tracking
            motion = track.get_motion_analysis(fps)

            # Get start/end positions
            trajectory = track.trajectory
            start_pos = trajectory[0] if trajectory else (0, 0)
            end_pos = trajectory[-1] if trajectory else (0, 0)

            # Get a representative bbox (middle of track)
            mid_entry = track.entries[len(track.entries) // 2]
            bbox = getattr(mid_entry, 'bbox', None)

            tracked_objects.append({
                "track_id": track_id,
                # Object identity (SAM-3 label from text-prompted segmentation)
                "label": label,
                # Timing
                "start_time": round(start_frame / fps, 1),
                "end_time": round(end_frame / fps, 1),
                "duration": round(duration, 1),
                "num_detections": len(track.entries),
                # Motion data (DINOv3 tracking)
                "movement_type": motion["movement_type"],
                "primary_direction": motion["primary_direction"],
                "avg_speed": round(motion["avg_speed"], 1),
                "total_distance": round(motion["total_distance"], 1),
                # Position data
                "start_position": start_pos,
                "end_position": end_pos,
                "bbox": bbox,
            })

        # Sort by activity (faster moving objects first, then by duration)
        tracked_objects = sorted(
            tracked_objects,
            key=lambda x: (x["avg_speed"], x["duration"]),
            reverse=True,
        )

        return tracked_objects[:10]  # Top 10 tracked objects

    def _get_context_frames(self, frame_idx: int, total_frames: int, context_size: int = 3) -> List[int]:
        """Get indices for context frames around a key moment."""
        indices = []

        # Get frames before and after
        step = max(1, total_frames // 20)  # Adaptive step based on video length

        for offset in range(-context_size + 1, context_size):
            idx = frame_idx + (offset * step)
            if 0 <= idx < total_frames:
                indices.append(idx)

        # Ensure we have at least the key frame
        if frame_idx not in indices and 0 <= frame_idx < total_frames:
            indices.append(frame_idx)

        return sorted(set(indices))[:4]  # Max 4 frames for VLM

    def process_frame_single(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Process a single frame (no temporal context).

        Args:
            image: PIL Image

        Returns:
            Dictionary with embeddings
        """
        # Lazy-load vision pipeline
        self._ensure_vision_pipeline()

        return self.vision_pipeline.get_embedding_for_search(image)

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if self.models:
            return self.models.get_memory_stats()
        return {}
