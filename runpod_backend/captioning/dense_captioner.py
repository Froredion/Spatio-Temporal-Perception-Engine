"""
Dense Captioning System

Generates rich, multi-granularity captions for every frame:
- Brief: 5-10 words, action-focused
- Normal: 1-2 sentences, balanced detail
- Detailed: Full paragraph with spatial relationships

Designed for self-supervised training dataset generation.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from PIL import Image
from enum import Enum

from captioning.spatial_relations import SpatialRelationExtractor, SpatialRelationship


class CaptionGranularity(Enum):
    """Caption detail levels."""
    BRIEF = "brief"  # 5-10 words
    NORMAL = "normal"  # 1-2 sentences
    DETAILED = "detailed"  # Full paragraph


@dataclass
class FrameCaption:
    """Complete caption data for a single frame."""
    frame_idx: int
    timestamp: float

    # Multi-granularity captions
    brief: str = ""
    normal: str = ""
    detailed: str = ""

    # Structured data
    objects: List[str] = field(default_factory=list)
    object_positions: List[Dict[str, Any]] = field(default_factory=list)
    spatial_relations: List[Dict[str, Any]] = field(default_factory=list)
    scene_layout: str = ""

    # Motion context (if available from tracking)
    motion_description: str = ""
    active_tracks: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    caption_confidence: float = 1.0
    num_objects_detected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 2),
            "captions": {
                "brief": self.brief,
                "normal": self.normal,
                "detailed": self.detailed,
            },
            "objects": self.objects,
            "object_positions": self.object_positions,
            "spatial_relations": self.spatial_relations,
            "scene_layout": self.scene_layout,
            "motion": {
                "description": self.motion_description,
                "active_tracks": self.active_tracks,
            },
            "quality": {
                "confidence": round(self.caption_confidence, 2),
                "num_objects": self.num_objects_detected,
            },
        }


@dataclass
class DenseCaptionResult:
    """Complete dense captioning result for a video."""
    video_id: str
    total_frames: int
    fps: float

    # Per-frame captions
    frame_captions: List[FrameCaption] = field(default_factory=list)

    # Aggregated data
    summary: str = ""
    all_objects: List[str] = field(default_factory=list)  # Unique objects across video
    key_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "frame_captions": [fc.to_dict() for fc in self.frame_captions],
            "summary": self.summary,
            "all_objects": self.all_objects,
            "key_events": self.key_events,
        }


class DenseCaptioner:
    """
    Generate dense, multi-granularity captions for video frames.

    Features:
    - Per-frame captions at multiple detail levels
    - Spatial relationship extraction
    - Motion-aware descriptions
    - Batch processing for efficiency

    Args:
        vlm_model: Qwen3VisionModel instance
        spatial_extractor: Optional SpatialRelationExtractor (created if not provided)
        caption_interval: Generate captions every N frames (1 = every frame)
        batch_size: Frames per VLM batch
    """

    def __init__(
        self,
        vlm_model,
        spatial_extractor: Optional[SpatialRelationExtractor] = None,
        caption_interval: int = 1,
        batch_size: int = 4,
    ):
        self.vlm = vlm_model
        self.spatial_extractor = spatial_extractor
        self.caption_interval = caption_interval
        self.batch_size = batch_size

    def generate_dense_captions(
        self,
        frames: List[Image.Image],
        frame_results: List[Any],  # List of FrameResult
        tracks: Dict[int, Any],
        fps: float,
        video_id: str = "",
        granularities: List[CaptionGranularity] = None,
        frame_metadata: Optional[Dict[int, Dict]] = None,
    ) -> DenseCaptionResult:
        """
        Generate dense captions for all frames.

        Uses mega-batch processing: ALL frames x ALL granularities in ONE VLM call.
        This is ~3x faster than sequential granularity processing.

        Args:
            frames: List of PIL Images
            frame_results: List of FrameResult from vision pipeline
            tracks: Dict of Track objects from tracker
            fps: Frames per second
            video_id: Video identifier
            granularities: Which detail levels to generate (default: all)
            frame_metadata: Optional dict mapping frame_number (1-indexed) -> metadata dict.
                           When provided, metadata is included in prompts for
                           ground-truth context (e.g., Roblox game state).

        Returns:
            DenseCaptionResult with all frame captions
        """
        if granularities is None:
            granularities = list(CaptionGranularity)

        result = DenseCaptionResult(
            video_id=video_id,
            total_frames=len(frames),
            fps=fps,
        )

        # Determine which frames to caption
        caption_frame_indices = list(range(0, len(frames), self.caption_interval))
        num_frames_to_caption = len(caption_frame_indices)
        num_granularities = len(granularities)
        total_captions = num_frames_to_caption * num_granularities

        print(f"[DenseCaptioner] Generating {total_captions} captions ({num_frames_to_caption} frames x {num_granularities} granularities)")

        # Pre-compute motion data for all frames
        motion_data = self._precompute_motion_data(tracks, len(frames), fps)

        # MEGA-BATCH: Prepare ALL frame contexts upfront
        all_frame_contexts = self._prepare_all_frame_contexts(
            frames=frames,
            frame_results=frame_results,
            frame_indices=caption_frame_indices,
            motion_data=motion_data,
            fps=fps,
            frame_metadata=frame_metadata,
        )

        # Build ONE mega-batch with all frames x all granularities
        mega_batch_items = []
        mega_batch_mapping = []  # Track (frame_context_idx, granularity) for each item

        for ctx_idx, ctx in enumerate(all_frame_contexts):
            for granularity in granularities:
                prompt = self._build_prompt(
                    granularity=granularity,
                    objects_data=ctx["objects_data"],
                    spatial_text=ctx["spatial_text"],
                    motion_desc=ctx["motion_desc"],
                    timestamp=ctx["timestamp"],
                    game_state=ctx.get("game_state"),
                )
                mega_batch_items.append({
                    "images": [ctx["frame"]],
                    "prompt": prompt,
                })
                mega_batch_mapping.append((ctx_idx, granularity))

        # Process in larger batches (memory permitting)
        # Larger batch = better GPU utilization
        mega_batch_size = self.batch_size * num_granularities  # e.g., 4 frames * 3 granularities = 12
        all_responses = []

        num_mega_batches = (len(mega_batch_items) + mega_batch_size - 1) // mega_batch_size
        print(f"[DenseCaptioner] Processing {num_mega_batches} mega-batches (size={mega_batch_size})")

        for batch_idx in range(0, len(mega_batch_items), mega_batch_size):
            batch_end = min(batch_idx + mega_batch_size, len(mega_batch_items))
            batch_items = mega_batch_items[batch_idx:batch_end]

            current_batch_num = batch_idx // mega_batch_size + 1
            print(f"[DenseCaptioner] Mega-batch {current_batch_num}/{num_mega_batches} ({len(batch_items)} items)")

            if self.vlm is not None:
                # Determine max tokens for this batch (use max of all granularities in batch)
                batch_granularities = [mega_batch_mapping[batch_idx + i][1] for i in range(len(batch_items))]
                max_tokens = max(self._get_max_tokens(g) for g in batch_granularities)

                responses = self.vlm.generate_batch(
                    batch_items,
                    temperature=0.3,
                    max_new_tokens=max_tokens,
                )
                all_responses.extend(responses)
            else:
                all_responses.extend([""] * len(batch_items))

        # Reconstruct captions by frame
        captions_by_frame = {i: {} for i in range(len(all_frame_contexts))}
        for resp_idx, (ctx_idx, granularity) in enumerate(mega_batch_mapping):
            captions_by_frame[ctx_idx][granularity] = all_responses[resp_idx].strip() if resp_idx < len(all_responses) else ""

        # Build FrameCaption objects
        all_frame_captions = []
        for ctx_idx, ctx in enumerate(all_frame_contexts):
            fc = FrameCaption(
                frame_idx=ctx["frame_idx"],
                timestamp=ctx["timestamp"],
                objects=[obj["label"] for obj in ctx["objects_data"]],
                object_positions=[
                    {
                        "label": obj["label"],
                        "position": self._get_position_label(obj["bbox"], ctx["frame_width"], ctx["frame_height"]),
                        "bbox": obj["bbox"],
                    }
                    for obj in ctx["objects_data"]
                ],
                spatial_relations=[rel.to_dict() for rel in ctx["relationships"]],
                scene_layout=ctx["scene_layout"],
                motion_description=ctx["motion_desc"],
                active_tracks=ctx["active_tracks"],
                num_objects_detected=len(ctx["objects_data"]),
            )

            # Add captions for each granularity
            frame_captions_dict = captions_by_frame.get(ctx_idx, {})
            if CaptionGranularity.BRIEF in granularities:
                fc.brief = frame_captions_dict.get(CaptionGranularity.BRIEF, "")
            if CaptionGranularity.NORMAL in granularities:
                fc.normal = frame_captions_dict.get(CaptionGranularity.NORMAL, "")
            if CaptionGranularity.DETAILED in granularities:
                fc.detailed = frame_captions_dict.get(CaptionGranularity.DETAILED, "")

            # Calculate confidence
            if ctx["objects_data"]:
                avg_conf = sum(o["confidence"] for o in ctx["objects_data"]) / len(ctx["objects_data"])
                fc.caption_confidence = avg_conf
            else:
                fc.caption_confidence = 0.5

            all_frame_captions.append(fc)

        result.frame_captions = all_frame_captions

        # Collect all unique objects
        all_objects = set()
        for fc in all_frame_captions:
            all_objects.update(fc.objects)
        result.all_objects = sorted(list(all_objects))

        # Generate overall summary using frame captions
        if len(all_frame_captions) >= 4 and self.vlm is not None:
            result.summary = self._generate_summary(frames, all_frame_captions, fps)

        # Detect key events (frames with significant changes)
        result.key_events = self._detect_key_events(all_frame_captions, fps)

        return result

    def _prepare_all_frame_contexts(
        self,
        frames: List[Image.Image],
        frame_results: List[Any],
        frame_indices: List[int],
        motion_data: Dict[int, Dict],
        fps: float,
        frame_metadata: Optional[Dict[int, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Prepare context data for all frames upfront (CPU work, no GPU)."""
        frame_contexts = []

        for frame_idx in frame_indices:
            if frame_idx >= len(frames):
                continue

            frame = frames[frame_idx]
            frame_result = frame_results[frame_idx] if frame_idx < len(frame_results) else None
            timestamp = frame_idx / fps

            # Extract object info
            objects_data = []
            if frame_result and hasattr(frame_result, 'objects'):
                for obj in frame_result.objects:
                    objects_data.append({
                        "label": obj.label,
                        "bbox": obj.bbox,
                        "area": obj.area,
                        "confidence": obj.confidence,
                    })

            # Get frame dimensions
            frame_width = frame_result.frame_width if frame_result else frame.size[0]
            frame_height = frame_result.frame_height if frame_result else frame.size[1]

            # Extract spatial relationships
            spatial_extractor = self.spatial_extractor or SpatialRelationExtractor(
                frame_width=frame_width,
                frame_height=frame_height,
            )
            relationships = spatial_extractor.extract_relationships(objects_data, max_relationships=10)
            scene_layout = spatial_extractor.get_scene_layout_description(objects_data)
            spatial_text = spatial_extractor.format_relationships_for_caption(relationships, max_relations=5)

            # Get motion context
            motion_context = motion_data.get(frame_idx, {})
            motion_desc = motion_context.get("description", "")
            active_tracks = motion_context.get("active_tracks", [])

            # Get Roblox game state metadata if available (1-indexed)
            frame_num = frame_idx + 1
            game_state = frame_metadata.get(frame_num) if frame_metadata else None

            frame_contexts.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "frame": frame,
                "objects_data": objects_data,
                "relationships": relationships,
                "scene_layout": scene_layout,
                "spatial_text": spatial_text,
                "motion_desc": motion_desc,
                "active_tracks": active_tracks,
                "frame_width": frame_width,
                "frame_height": frame_height,
                "game_state": game_state,
            })

        return frame_contexts

    def _build_prompt(
        self,
        granularity: CaptionGranularity,
        objects_data: List[Dict],
        spatial_text: str,
        motion_desc: str,
        timestamp: float,
        game_state: Optional[Dict] = None,
    ) -> str:
        """Build prompt for specific granularity level."""
        # If we have Roblox game state metadata, use special prompt
        if game_state:
            return self._build_roblox_prompt(game_state, granularity)

        # Build context from detected objects
        object_list = ", ".join([obj["label"] for obj in objects_data[:8]]) if objects_data else "no specific objects detected"

        context_parts = []
        if objects_data:
            context_parts.append(f"Detected objects: {object_list}.")
        if spatial_text:
            context_parts.append(f"Spatial layout: {spatial_text}")
        if motion_desc:
            context_parts.append(f"Motion: {motion_desc}")

        context = " ".join(context_parts)

        if granularity == CaptionGranularity.BRIEF:
            return f"""{context}

Describe this frame in 5-10 words. Focus on the main action or subject. Be extremely concise."""

        elif granularity == CaptionGranularity.NORMAL:
            return f"""{context}

Describe this frame in 1-2 sentences. Include what objects are present and what is happening. Be specific but concise."""

        else:  # DETAILED
            # Treat pre-detected objects as hints, not ground truth
            hint_context = ""
            if context:
                hint_context = f"Pre-detected (may be incomplete): {context}\n\n"

            return f"""{hint_context}Provide a comprehensive description of this frame:
1. ALL visible objects (including any not listed above) and their positions (left, center, right, foreground, background)
2. What action or event is occurring
3. Spatial relationships between ALL elements in the scene
4. Any motion, changes, or temporal cues visible (blur, pose, trajectory)
5. Scene context (indoor/outdoor, lighting, setting, environment)
6. Fine details: text, symbols, subtle objects, partially occluded items

Look carefully - the pre-detected list may miss objects. Describe everything you observe."""

    def _build_roblox_prompt(
        self,
        game_state: Dict,
        granularity: CaptionGranularity,
    ) -> str:
        """
        Build VLM prompt with Roblox game state context.

        When game state metadata is available, we include it as ground truth
        to help the VLM generate accurate, aligned captions.

        Args:
            game_state: Roblox metadata dict with humanoid_state, hrp, inputs, etc.
            granularity: Caption detail level

        Returns:
            Prompt string with game state context
        """
        # Extract key info from metadata
        state = game_state.get('humanoid_state', 'Unknown')
        hrp = game_state.get('hrp', {})
        speed = hrp.get('speed', 0)
        velocity = hrp.get('velocity', [0, 0, 0])
        inputs = game_state.get('inputs', [])
        ground = game_state.get('ground', {})
        dist_to_ground = ground.get('distance_to_landing', 0)
        surface = ground.get('landing_instance', 'Unknown')

        # Format inputs
        input_str = ', '.join(inputs) if inputs else 'None'

        # Determine movement direction from velocity
        vx, vy, vz = velocity if len(velocity) >= 3 else (0, 0, 0)
        if abs(vx) > abs(vz):
            h_direction = 'right' if vx > 0 else 'left'
        else:
            h_direction = 'forward' if vz < 0 else 'backward'
        v_direction = 'upward' if vy > 1 else 'downward' if vy < -1 else 'level'

        # Build game state context
        game_context = f"""GAME STATE (ground truth):
- Humanoid State: {state}
- Speed: {speed:.1f} studs/sec
- Movement: {h_direction}, {v_direction}
- Player Inputs: [{input_str}]
- Ground Distance: {dist_to_ground:.2f} studs
- Surface: {surface}"""

        if granularity == CaptionGranularity.BRIEF:
            return f"""Analyze this Roblox game frame.

{game_context}

Describe in 5-10 words what the character is doing. Use the humanoid state as ground truth."""

        elif granularity == CaptionGranularity.NORMAL:
            return f"""Analyze this Roblox game frame.

{game_context}

Describe in 1-2 sentences:
1. What the character is doing (use humanoid state as ground truth)
2. The environment visible in the frame

Be specific and accurate."""

        else:  # DETAILED
            return f"""Analyze this Roblox game frame with the following game state:

{game_context}

Based on both the visual content and the game state above, provide a detailed description:
1. What the character is doing (use the humanoid state as ground truth)
2. The environment and any visible objects
3. How the visual matches the reported movement/state
4. Any other players, NPCs, or interactive elements visible
5. Scene context (indoor/outdoor, lighting, setting)

The game state is ground truth - use it to inform your description. Be specific and accurate."""

    def _get_max_tokens(self, granularity: CaptionGranularity) -> int:
        """Get max tokens for each granularity level."""
        return {
            CaptionGranularity.BRIEF: 30,
            CaptionGranularity.NORMAL: 80,
            CaptionGranularity.DETAILED: 400,
        }[granularity]

    def _get_position_label(
        self,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> str:
        """Get human-readable position label for bbox."""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        h_pos = "left" if cx < frame_width * 0.33 else "right" if cx > frame_width * 0.67 else "center"
        v_pos = "top" if cy < frame_height * 0.33 else "bottom" if cy > frame_height * 0.67 else "middle"

        if h_pos == "center" and v_pos == "middle":
            return "center"
        elif v_pos == "middle":
            return h_pos
        elif h_pos == "center":
            return v_pos
        else:
            return f"{v_pos}-{h_pos}"

    def _precompute_motion_data(
        self,
        tracks: Dict[int, Any],
        num_frames: int,
        fps: float,
    ) -> Dict[int, Dict]:
        """Pre-compute motion descriptions for each frame from tracks."""
        motion_data = {}

        for frame_idx in range(num_frames):
            timestamp = frame_idx / fps
            active_tracks = []
            motion_parts = []

            for track_id, track in tracks.items():
                if not hasattr(track, 'entries'):
                    continue

                # Check if track is active at this frame
                start_frame = track.entries[0].frame_idx if track.entries else -1
                end_frame = track.entries[-1].frame_idx if track.entries else -1

                if start_frame <= frame_idx <= end_frame:
                    # Track is active
                    label = track.get_majority_label() if hasattr(track, 'get_majority_label') else "object"
                    motion = track.get_motion_analysis(fps) if hasattr(track, 'get_motion_analysis') else {}

                    track_info = {
                        "track_id": track_id,
                        "label": label,
                        "movement_type": motion.get("movement_type", "unknown"),
                        "direction": motion.get("primary_direction", "unknown"),
                    }
                    active_tracks.append(track_info)

                    # Build motion description
                    if motion.get("movement_type") and motion["movement_type"] != "stationary":
                        motion_parts.append(
                            f"{label} moving {motion.get('primary_direction', '')} ({motion['movement_type']})"
                        )

            motion_data[frame_idx] = {
                "active_tracks": active_tracks,
                "description": ", ".join(motion_parts) if motion_parts else "",
            }

        return motion_data

    def _generate_summary(
        self,
        frames: List[Image.Image],
        frame_captions: List[FrameCaption],
        fps: float,
    ) -> str:
        """Generate overall video summary from frame captions."""
        # Sample 4 frames evenly
        indices = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
        sample_frames = [frames[min(i, len(frames) - 1)] for i in indices]

        # Build context from frame captions
        caption_samples = []
        for i, fc in enumerate(frame_captions):
            if i % max(1, len(frame_captions) // 5) == 0:  # Sample ~5 captions
                caption_samples.append(f"[{fc.timestamp:.1f}s] {fc.normal or fc.brief}")

        caption_context = "\n".join(caption_samples[:6])

        # Collect all unique objects
        all_objects = set()
        for fc in frame_captions:
            all_objects.update(fc.objects)

        prompt = f"""Video duration: {len(frames) / fps:.1f} seconds

Key frames described:
{caption_context}

Objects seen throughout: {', '.join(sorted(all_objects)[:15])}

Provide a comprehensive 2-3 sentence summary of the entire video. Describe:
1. The main subject/activity
2. How it progresses over time
3. The outcome or conclusion

Be specific and accurate based on the frame descriptions."""

        return self.vlm.generate(sample_frames, prompt, temperature=0.3, max_new_tokens=150).strip()

    def _detect_key_events(
        self,
        frame_captions: List[FrameCaption],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """Detect key events based on changes in frame captions."""
        key_events = []

        prev_objects = set()
        for i, fc in enumerate(frame_captions):
            current_objects = set(fc.objects)

            # Detect new objects appearing
            new_objects = current_objects - prev_objects
            if new_objects and i > 0:
                key_events.append({
                    "type": "object_appears",
                    "frame_idx": fc.frame_idx,
                    "timestamp": fc.timestamp,
                    "objects": list(new_objects),
                    "description": f"{', '.join(new_objects)} appears",
                })

            # Detect objects disappearing
            disappeared = prev_objects - current_objects
            if disappeared and i > 0:
                key_events.append({
                    "type": "object_disappears",
                    "frame_idx": fc.frame_idx,
                    "timestamp": fc.timestamp,
                    "objects": list(disappeared),
                    "description": f"{', '.join(disappeared)} exits",
                })

            # Detect significant motion
            for track in fc.active_tracks:
                if track.get("movement_type") == "fast":
                    key_events.append({
                        "type": "fast_motion",
                        "frame_idx": fc.frame_idx,
                        "timestamp": fc.timestamp,
                        "object": track.get("label", "object"),
                        "direction": track.get("direction", ""),
                        "description": f"{track.get('label', 'object')} moving fast {track.get('direction', '')}",
                    })

            prev_objects = current_objects

        # Deduplicate and limit
        seen = set()
        unique_events = []
        for event in key_events:
            key = (event["type"], event.get("objects", [event.get("object", "")])[0] if isinstance(event.get("objects"), list) else event.get("object", ""))
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        return unique_events[:20]  # Limit to 20 key events


def generate_frame_caption_standalone(
    vlm_model,
    frame: Image.Image,
    objects: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    granularity: CaptionGranularity = CaptionGranularity.NORMAL,
    motion_context: str = "",
) -> str:
    """
    Generate a single frame caption (standalone utility function).

    Args:
        vlm_model: Qwen3VisionModel instance
        frame: PIL Image
        objects: List of detected objects with label, bbox, confidence
        frame_width: Frame width
        frame_height: Frame height
        granularity: Caption detail level
        motion_context: Optional motion description

    Returns:
        Caption string
    """
    # Extract spatial relationships
    extractor = SpatialRelationExtractor(frame_width, frame_height)
    relationships = extractor.extract_relationships(objects, max_relationships=5)
    spatial_text = extractor.format_relationships_for_caption(relationships, max_relations=3)

    # Build object list
    object_list = ", ".join([obj["label"] for obj in objects[:8]]) if objects else "no objects"

    # Build prompt based on granularity
    context = f"Objects: {object_list}."
    if spatial_text:
        context += f" {spatial_text}"
    if motion_context:
        context += f" Motion: {motion_context}"

    prompts = {
        CaptionGranularity.BRIEF: f"{context}\n\nDescribe in 5-10 words. Main action only.",
        CaptionGranularity.NORMAL: f"{context}\n\nDescribe in 1-2 sentences. What is happening?",
        CaptionGranularity.DETAILED: f"{context}\n\nProvide a detailed description including all objects, positions, actions, and spatial relationships.",
    }

    max_tokens = {
        CaptionGranularity.BRIEF: 30,
        CaptionGranularity.NORMAL: 80,
        CaptionGranularity.DETAILED: 200,
    }

    return vlm_model.generate(
        frame,
        prompts[granularity],
        max_new_tokens=max_tokens[granularity],
        temperature=0.3,
    ).strip()
