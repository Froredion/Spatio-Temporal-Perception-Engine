"""
Video processing handlers.

Handles video and frame processing operations.
"""

import os
import tempfile
from typing import Dict, Any

from .utils import get_video_file, get_image, decode_base64_image
from .storage import save_analysis_to_r2
from utils.r2_storage import get_r2_client


def handle_process_video(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Process video file.

    Input:
        video_url: URL to download video from (optional)
        video_base64: Base64 encoded video (optional)
        fps: Frames per second (default 2.0)
        generate_captions: Whether to generate captions (default True)
        dense_captions: Whether to generate dense per-frame captions (default True)
        caption_interval: Generate captions every N frames (default 1 = every frame)
        caption_granularities: List of detail levels ['brief', 'normal', 'detailed'] (default all)
        processing_resolution: Optional [width, height] to resize frames before processing.
                               E.g., [1280, 720] for 720p, [854, 480] for 480p.
                               Lower resolution = faster processing but less detail.

    Returns:
        Video processing results with embeddings
    """
    print("[Handler] handle_process_video started")

    fps = data.get('fps', 2.0)
    generate_captions = data.get('generate_captions', True)
    dense_captions = data.get('dense_captions', True)
    caption_interval = data.get('caption_interval', 1)
    caption_granularities = data.get('caption_granularities', None)  # None = all levels

    # Parse processing_resolution (can be list or tuple)
    processing_resolution = data.get('processing_resolution', None)
    if processing_resolution is not None:
        processing_resolution = tuple(processing_resolution)

    print("[Handler] Getting video file...")
    video_path = get_video_file(data)

    if video_path is None:
        return {'error': 'No video provided. Use video_url or video_base64'}

    print(f"[Handler] Video path: {video_path}")
    if processing_resolution:
        print(f"[Handler] Processing resolution: {processing_resolution[0]}x{processing_resolution[1]}")
    print(f"[Handler] Caption settings: dense={dense_captions}, interval={caption_interval}, granularities={caption_granularities}")

    try:
        print("[Handler] Calling pipeline.process_video...")
        result = pipeline.process_video(
            video_path=video_path,
            fps=fps,
            generate_captions=generate_captions,
            dense_captions=dense_captions,
            caption_interval=caption_interval,
            caption_granularities=caption_granularities,
            processing_resolution=processing_resolution,
        )

        output = {
            'video_id': result.video_id,
            'num_frames': len(result.frames),
            'clip_embedding': result.clip_embedding.cpu().tolist(),
            'temporal_features_shape': list(result.temporal_features.shape),
            'num_tracks': len(result.tracks),
            'captions': result.captions,
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

        if data.get('include_frame_embeddings', False):
            output['frame_embeddings'] = [
                f.global_embedding.cpu().tolist() for f in result.frames
            ]

        if data.get('include_frame_details', True):
            frame_details = _build_frame_details(result, fps)
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

        analysis_url = save_analysis_to_r2(result.video_id, output)
        if analysis_url:
            output['analysis_url'] = analysis_url

        # Cleanup VideoResult GPU tensors (data already saved to R2)
        _cleanup_video_result(result)

        return {'output': output}

    finally:
        if video_path.startswith(tempfile.gettempdir()):
            os.unlink(video_path)


def _build_frame_details(result, fps: float) -> list:
    """Build per-frame details with objects and tracks."""
    frame_details = []

    for i, f in enumerate(result.frames):
        detail = {
            'frame_idx': i,
            'timestamp': round(i / fps, 2),
            'num_objects': len(f.objects),
            'has_dense': f.dense_features is not None,
            'frame_width': f.frame_width,
            'frame_height': f.frame_height,
        }

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


def handle_process_frame(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Process single frame.

    Input:
        image_base64: Base64 encoded image
        image_url: URL to download image (optional)

    Returns:
        Frame embeddings (DINOv3 only)
    """
    image = get_image(data)

    if image is None:
        return {'error': 'No image provided. Use image_base64 or image_url'}

    embeddings = pipeline.process_frame_single(image)

    return {
        'output': {
            'dinov3_embedding': embeddings['dinov3'].cpu().tolist(),
        }
    }


def handle_process_image(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Process single image through full pipeline (like video with 1 frame).

    Input:
        image_base64: Base64 encoded image
        image_url: URL to download image (optional)
        generate_caption: Whether to generate caption (default True)

    Returns:
        Full processing results similar to video output
    """
    import uuid

    image = get_image(data)
    generate_caption = data.get('generate_caption', True)

    if image is None:
        return {'error': 'No image provided. Use image_base64 or image_url'}

    print("[Handler] handle_process_image started")

    # Ensure vision pipeline is ready
    pipeline._ensure_vision_pipeline()

    image_id = str(uuid.uuid4())[:8]
    print(f"[Handler] Image ID: {image_id}")

    # Process through vision pipeline (Level 1 + Level 2 with SAM-3 labels)
    print("[Handler] Processing image through vision pipeline...")
    frame_results = pipeline.vision_pipeline.process_batch(
        [image],
        start_idx=0,
        extract_objects=True,
    )
    frame_result = frame_results[0]

    # Skip Level 3 dense features for images (expedite processing)
    # if pipeline.vision_pipeline.level3 is not None:
    #     print("[Handler] Adding Level 3 dense features...")
    #     level3_output = pipeline.vision_pipeline.level3.process(image, high_res=True)
    #     frame_result.dense_features = level3_output["features"].squeeze(0)
    #     frame_result.depth_cues = level3_output["depth_cues"]
    #     frame_result.edge_map = level3_output["edge_map"]

    # Skip caption generation for images (expedite processing)
    caption = None
    # if generate_caption and pipeline.models.vlm is not None:
    #     print("[Handler] Generating caption...")
    #     caption = pipeline.models.vlm.caption(image, detail_level='normal')

    # Upload image to R2 for preview
    frame_url = None
    thumbnail_url = None
    try:
        from io import BytesIO
        r2 = get_r2_client()

        # Save image as JPEG bytes
        img_buffer = BytesIO()
        image.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()

        # Upload as frame (use same path structure as videos for consistency)
        frame_key = f"videos/processed/{image_id}/frames/frame_0000.jpg"
        frame_url = r2.upload_bytes(img_bytes, frame_key, content_type='image/jpeg')
        print(f"[Handler] Uploaded frame to R2: {frame_url}")

        # Upload as thumbnail (same image for single frame)
        thumb_key = f"thumbnails/{image_id}_thumb.jpg"
        thumbnail_url = r2.upload_bytes(img_bytes, thumb_key, content_type='image/jpeg')
        print(f"[Handler] Uploaded thumbnail to R2: {thumbnail_url}")
    except Exception as e:
        print(f"[Handler] Failed to upload image to R2: {e}")

    # Build output similar to video format
    output = {
        'video_id': image_id,  # Use same field name for compatibility
        'num_frames': 1,
        'clip_embedding': frame_result.global_embedding.cpu().tolist(),
        'temporal_features_shape': [1, frame_result.global_embedding.shape[0]],
        'num_tracks': 0,
        'captions': {'summary': caption} if caption else None,
        'thumbnail_url': thumbnail_url,
        'frame_urls': [frame_url] if frame_url else [],
    }

    output['level_stats'] = {
        'level1_global': {
            'frames_processed': 1,
            'embedding_dim': frame_result.global_embedding.shape[0],
        },
        'level2_objects': {
            'total_objects_detected': len(frame_result.objects),
            'tracks_created': 0,
            'avg_objects_per_frame': len(frame_result.objects),
        },
        'level3_dense': {
            'frames_with_dense': 1 if frame_result.dense_features is not None else 0,
            'scene_graph_nodes': 0,
        },
    }

    # Include frame details
    if data.get('include_frame_details', True):
        frame_detail = {
            'frame_idx': 0,
            'timestamp': 0.0,
            'num_objects': len(frame_result.objects),
            'has_dense': frame_result.dense_features is not None,
            'frame_width': frame_result.frame_width,
            'frame_height': frame_result.frame_height,
        }
        if frame_result.objects:
            frame_detail['objects'] = [
                {
                    'bbox': obj.bbox,
                    'area': obj.area,
                    'confidence': round(obj.confidence, 2),
                    'label': obj.label,
                }
                for obj in frame_result.objects
            ]
        output['frame_details'] = [frame_detail]

    # Save analysis to R2
    analysis_url = save_analysis_to_r2(image_id, output)
    if analysis_url:
        output['analysis_url'] = analysis_url

    # Cleanup frame_result GPU tensors (data already saved to R2)
    _cleanup_frame_result(frame_result)

    print("[Handler] handle_process_image completed")
    return {'output': output}


def handle_compare_frames(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Compare two frames and describe the difference.

    Input:
        frame1_base64: Base64 encoded first frame
        frame2_base64: Base64 encoded second frame

    Returns:
        Description of changes
    """
    if pipeline.models.vlm is None:
        return {'error': 'Qwen3-VL not loaded (vision_only mode)'}

    frame1 = decode_base64_image(data.get('frame1_base64'))
    frame2 = decode_base64_image(data.get('frame2_base64'))

    if frame1 is None or frame2 is None:
        return {'error': 'Both frames required'}

    description = pipeline.models.vlm.compare_frames(frame1, frame2)

    return {'output': {'description': description}}


def _cleanup_video_result(result) -> None:
    """
    Release GPU memory from VideoResult after extracting output data.

    Called after data has been copied to CPU and saved to R2.
    Explicitly nullifies tensor references to help garbage collection.

    Args:
        result: VideoResult object to cleanup
    """
    from utils.memory import clear_gpu_cache

    # Clear frame tensors
    for frame in result.frames:
        frame.global_embedding = None
        frame.patch_features = None
        frame.dense_features = None
        frame.depth_cues = None
        frame.edge_map = None
        frame.image = None  # Release PIL image reference

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


def _cleanup_frame_result(frame_result) -> None:
    """
    Release GPU memory from a single FrameResult.

    Called after data has been copied to CPU and saved to R2.

    Args:
        frame_result: FrameResult object to cleanup
    """
    from utils.memory import clear_gpu_cache

    frame_result.global_embedding = None
    frame_result.patch_features = None
    frame_result.dense_features = None
    frame_result.depth_cues = None
    frame_result.edge_map = None
    frame_result.image = None

    for obj in frame_result.objects:
        obj.dinov3_embedding = None
    frame_result.objects.clear()

    clear_gpu_cache()
