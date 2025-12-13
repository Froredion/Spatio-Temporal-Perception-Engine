"""
Extracted frames processing handler.

Handles pre-extracted frame datasets (ZIP files with PNG + JSON metadata).
Supports Roblox game state metadata for enhanced captioning.
"""

import os
import tempfile
from typing import Dict, Any, List, Optional

from PIL import Image

from .utils import get_zip_file, extract_frames_from_zip
from .storage import save_analysis_to_r2
from utils.r2_storage import get_r2_client


def handle_process_extracted_frames(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Process pre-extracted frames from a ZIP file.

    Input:
        zip_url: URL to download ZIP from (optional)
        zip_base64: Base64 encoded ZIP (optional)
        zip_path: Local path to ZIP (for testing)
        fps: Target FPS for analysis (default 10.0) - frames will be subsampled to this rate
        source_fps: Original recording FPS (default 60.0) - used for subsampling calculation
        include_roblox_metadata: Whether to parse frame JSON files (default True)
        generate_captions: Whether to generate captions (default True)
        dense_captions: Whether to generate dense per-frame captions (default True)
        caption_interval: Generate captions every N frames (default 1)
        caption_granularities: List of ['brief', 'normal', 'detailed'] (default all)
        include_frame_embeddings: Whether to include frame embeddings in output (default False)
        include_frame_details: Whether to include per-frame details (default True)
        processing_resolution: Optional [width, height] to resize frames before processing.
                               E.g., [1280, 720] for 720p, [854, 480] for 480p.
                               Lower resolution = faster processing but less detail.

    Returns:
        Processing results with embeddings and optional Roblox metadata
    """
    print("[Handler] handle_process_extracted_frames started")

    # Parse parameters
    fps = data.get('fps', 10.0)  # Target FPS for analysis
    source_fps = data.get('source_fps', 60.0)  # Original recording FPS
    include_roblox_metadata = data.get('include_roblox_metadata', True)
    generate_captions = data.get('generate_captions', True)
    dense_captions = data.get('dense_captions', True)
    caption_interval = data.get('caption_interval', 1)
    caption_granularities = data.get('caption_granularities', None)

    # Parse processing_resolution (can be list or tuple)
    processing_resolution = data.get('processing_resolution', None)
    if processing_resolution is not None:
        processing_resolution = tuple(processing_resolution)

    print(f"[Handler] Parameters: target_fps={fps}, source_fps={source_fps}, include_metadata={include_roblox_metadata}")
    if processing_resolution:
        print(f"[Handler] Processing resolution: {processing_resolution[0]}x{processing_resolution[1]}")
    print(f"[Handler] Caption settings: dense={dense_captions}, interval={caption_interval}")

    # Download and extract ZIP
    print("[Handler] Getting ZIP file...")
    zip_path = get_zip_file(data)
    if zip_path is None:
        return {'error': 'No ZIP file provided. Use zip_url, zip_base64, or zip_path'}

    print(f"[Handler] ZIP path: {zip_path}")

    try:
        # Extract frames from ZIP
        print("[Handler] Extracting frames from ZIP...")
        frames, roblox_metadata = extract_frames_from_zip(
            zip_path,
            include_metadata=include_roblox_metadata,
        )

        if not frames:
            return {'error': 'No frames found in ZIP file'}

        print(f"[Handler] Extracted {len(frames)} frames from ZIP")
        if roblox_metadata:
            print(f"[Handler] Found metadata for {len(roblox_metadata)} frames")

        # Subsample frames if target fps < source fps
        original_frame_count = len(frames)
        if source_fps > fps:
            step = source_fps / fps  # e.g., 60/10 = 6 (take every 6th frame)
            sampled_indices = []
            i = 0.0
            while int(i) < len(frames):
                sampled_indices.append(int(i))
                i += step

            # Subsample frames
            frames = [frames[idx] for idx in sampled_indices]

            # Subsample metadata (need to remap frame numbers)
            if roblox_metadata:
                # Original frame numbers are 1-indexed
                # Map old frame numbers to new positions
                new_metadata = {}
                for new_idx, old_idx in enumerate(sampled_indices):
                    old_frame_num = old_idx + 1  # Convert to 1-indexed
                    if old_frame_num in roblox_metadata:
                        new_metadata[new_idx + 1] = roblox_metadata[old_frame_num]
                roblox_metadata = new_metadata

            print(f"[Handler] Subsampled {original_frame_count} frames at {source_fps} FPS -> {len(frames)} frames at {fps} FPS (step={step:.1f})")
        else:
            print(f"[Handler] No subsampling needed (source_fps={source_fps} <= target_fps={fps})")

        # Process through pipeline
        print("[Handler] Calling pipeline.process_frames...")
        result = pipeline.process_frames(
            frames=frames,
            fps=fps,
            generate_captions=generate_captions,
            dense_captions=dense_captions,
            caption_interval=caption_interval,
            caption_granularities=caption_granularities,
            frame_metadata=roblox_metadata,
            processing_resolution=processing_resolution,
        )

        # Build output (same structure as process_video)
        output = {
            'video_id': result.video_id,
            'num_frames': len(result.frames),
            'clip_embedding': result.clip_embedding.cpu().tolist(),
            'temporal_features_shape': list(result.temporal_features.shape),
            'num_tracks': len(result.tracks),
            'captions': result.captions,
            'source_type': 'extracted_frames',  # Distinguish from video
            'processing_resolution': list(processing_resolution) if processing_resolution else None,
        }

        # Include dense captions if generated
        if result.dense_captions is not None:
            output['dense_captions'] = result.dense_captions.to_dict()
            output['caption_stats'] = {
                'total_frame_captions': len(result.dense_captions.frame_captions),
                'caption_interval': caption_interval,
                'granularities': caption_granularities or ['brief', 'normal', 'detailed'],
                'unique_objects': len(result.dense_captions.all_objects),
                'key_events': len(result.dense_captions.key_events),
            }

        # Add level stats
        output['level_stats'] = {
            'level1_global': {
                'frames_processed': len(result.frames),
                'embedding_dim': result.clip_embedding.shape[0] if result.clip_embedding is not None else 0,
            },
            'level2_objects': {
                'total_objects_detected': sum(len(f.objects) for f in result.frames),
                'tracks_created': len(result.tracks),
                'avg_objects_per_frame': sum(len(f.objects) for f in result.frames) / max(1, len(result.frames)),
            },
            'level3_dense': {
                'frames_with_dense': sum(1 for f in result.frames if f.dense_features is not None),
                'scene_graph_nodes': len(result.scene_graph.get('node_embeddings', [])) if isinstance(result.scene_graph, dict) else 0,
            },
        }

        # Add Roblox metadata to output
        if roblox_metadata:
            output['roblox_metadata'] = roblox_metadata
            output['metadata_stats'] = {
                'frames_with_metadata': len(roblox_metadata),
                'humanoid_states': list(set(
                    m.get('humanoid_state') for m in roblox_metadata.values()
                    if m.get('humanoid_state')
                )),
                'has_inputs': any(m.get('inputs') for m in roblox_metadata.values()),
            }

        # Include frame embeddings if requested
        if data.get('include_frame_embeddings', False):
            output['frame_embeddings'] = [
                f.global_embedding.cpu().tolist() for f in result.frames
            ]

        # Include frame details
        if data.get('include_frame_details', True):
            frame_details = _build_frame_details(result, fps, roblox_metadata)
            output['frame_details'] = frame_details

        # Upload frames and thumbnail to R2
        frame_urls = []
        thumbnail_url = None
        try:
            r2 = get_r2_client()
            if r2 is not None:
                from io import BytesIO

                for i, frame in enumerate(result.frames):
                    if frame.image is not None:
                        img_buffer = BytesIO()
                        frame.image.save(img_buffer, format='JPEG', quality=85)
                        img_bytes = img_buffer.getvalue()

                        # Upload frame
                        padded_idx = str(i).zfill(4)
                        frame_key = f"videos/processed/{result.video_id}/frames/frame_{padded_idx}.jpg"
                        frame_url = r2.upload_bytes(img_bytes, frame_key, content_type='image/jpeg')
                        frame_urls.append(frame_url)

                        # Use first frame as thumbnail
                        if i == 0:
                            thumb_key = f"thumbnails/{result.video_id}_thumb.jpg"
                            thumbnail_url = r2.upload_bytes(img_bytes, thumb_key, content_type='image/jpeg')

                print(f"[Handler] Uploaded {len(frame_urls)} frames to R2")
        except Exception as e:
            print(f"[Handler] Failed to upload frames to R2: {e}")

        if frame_urls:
            output['frame_urls'] = frame_urls
        if thumbnail_url:
            output['thumbnail_url'] = thumbnail_url

        # Save analysis to R2
        analysis_url = save_analysis_to_r2(result.video_id, output)
        if analysis_url:
            output['analysis_url'] = analysis_url

        # Cleanup VideoResult GPU tensors
        _cleanup_video_result(result)

        print("[Handler] handle_process_extracted_frames completed")
        return {'output': output}

    finally:
        # Cleanup temp ZIP file
        if zip_path and zip_path.startswith(tempfile.gettempdir()):
            try:
                os.unlink(zip_path)
            except Exception:
                pass


def _build_frame_details(
    result,
    fps: float,
    roblox_metadata: Optional[Dict[int, Dict]] = None,
) -> list:
    """Build per-frame details with objects, tracks, and Roblox metadata."""
    frame_details = []

    for i, f in enumerate(result.frames):
        frame_num = i + 1  # Roblox metadata uses 1-indexed frame numbers
        detail = {
            'frame_idx': i,
            'timestamp': round(i / fps, 2),
            'num_objects': len(f.objects),
            'has_dense': f.dense_features is not None,
            'frame_width': f.frame_width,
            'frame_height': f.frame_height,
        }

        # Add detected objects
        if f.objects:
            detail['objects'] = [
                {
                    'bbox': obj.bbox,
                    'area': obj.area,
                    'confidence': round(obj.confidence, 2),
                    'label': obj.label,
                }
                for obj in f.objects
            ]

        # Add Roblox metadata for this frame
        if roblox_metadata and frame_num in roblox_metadata:
            meta = roblox_metadata[frame_num]
            detail['roblox_metadata'] = {
                'humanoid_state': meta.get('humanoid_state'),
                'speed': meta.get('hrp', {}).get('speed', 0),
                'velocity': meta.get('hrp', {}).get('velocity'),
                'inputs': meta.get('inputs', []),
                'ground_distance': meta.get('ground', {}).get('distance_to_landing', 0),
                'surface': meta.get('ground', {}).get('landing_instance'),
            }

        frame_details.append(detail)

    # Add motion data from tracks
    for track_id, track in result.tracks.items():
        motion = track.get_motion_analysis(fps)
        for entry in track.entries:
            frame_idx = entry.frame_idx
            if frame_idx < len(frame_details):
                if 'tracks' not in frame_details[frame_idx]:
                    frame_details[frame_idx]['tracks'] = []

                entry_idx = track.entries.index(entry)
                velocity = None
                direction = None

                if entry_idx > 0 and entry_idx - 1 < len(motion['velocities']):
                    vel = motion['velocities'][entry_idx - 1]
                    velocity = round((vel[0]**2 + vel[1]**2)**0.5 * fps, 1)
                    direction = motion['directions'][entry_idx - 1] if entry_idx - 1 < len(motion['directions']) else None

                frame_details[frame_idx]['tracks'].append({
                    'track_id': track_id,
                    'bbox': entry.bbox,
                    'velocity': velocity,
                    'direction': round(direction, 1) if direction else None,
                })

    return frame_details


def _cleanup_video_result(result) -> None:
    """
    Release GPU memory from VideoResult after extracting output data.

    Called after data has been copied to CPU and saved to R2.
    """
    from utils.memory import clear_gpu_cache

    # Clear frame tensors
    for frame in result.frames:
        frame.global_embedding = None
        frame.patch_features = None
        frame.dense_features = None
        frame.depth_cues = None
        frame.edge_map = None
        frame.image = None

        for obj in frame.objects:
            obj.dinov3_embedding = None
        frame.objects.clear()
    result.frames.clear()

    # Clear temporal tensors
    result.temporal_features = None
    result.motion_features = None
    result.clip_embedding = None

    # Clear scene graph tensors
    if isinstance(result.scene_graph, dict):
        result.scene_graph.clear()

    # Clear tracks
    for track in result.tracks.values():
        for entry in track.entries:
            entry.embedding = None
        track.entries.clear()
    result.tracks.clear()

    # Clear GPU cache
    clear_gpu_cache()
