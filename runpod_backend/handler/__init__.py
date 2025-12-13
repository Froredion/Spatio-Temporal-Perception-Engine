"""
STPE Request Handler

Core request handler for video processing operations.
Used by server.py for dedicated pod deployment.
"""

from typing import Dict, Any

from .pipeline import get_pipeline, get_r2_client
from .health import handle_health
from .video import handle_process_video, handle_process_frame, handle_process_image, handle_compare_frames
from .extracted_frames import handle_process_extracted_frames
from .embedding import handle_embed_text, handle_embed_image, handle_caption
from .storage import handle_get_analysis, handle_list_videos

# Re-export for backwards compatibility
__all__ = [
    'handler',
    'get_pipeline',
    'get_r2_client',
]

# Operations that require full cleanup after processing (GPU-intensive)
_CLEANUP_OPERATIONS = {
    'process_video',
    'process_frame',
    'process_image',
    'process_extracted_frames',
    'embed_image',
    'caption',
    'compare_frames',
}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler.

    Supported operations:
    - process_video: Process video and return embeddings
    - process_frame: Process single frame (embeddings only)
    - process_image: Process image through full pipeline (like video with 1 frame)
    - embed_text: Get text embedding
    - embed_image: Get image embedding
    - caption: Generate image caption
    - compare_frames: Compare two frames
    - get_analysis: Get saved video analysis
    - list_videos: List processed videos
    - memory_stats: Get memory statistics
    - health: Health check

    Request format:
    {
        "input": {
            "operation": "process_video" | ...,
            "data": { ... operation-specific data ... }
        }
    }

    Returns:
        Response dictionary with 'output' or 'error'
    """
    pipeline = None
    operation = None

    try:
        input_data = event.get('input', {})
        operation = input_data.get('operation', 'health')
        data = input_data.get('data', {})

        print(f"[Handler] Received operation: {operation}")

        # Health check (no pipeline needed)
        if operation == 'health':
            return handle_health()

        # Storage operations (no pipeline needed)
        if operation == 'get_analysis':
            return handle_get_analysis(data)

        if operation == 'list_videos':
            return handle_list_videos(data)

        # Operations requiring pipeline
        pipeline = get_pipeline()

        handlers = {
            'process_video': lambda: handle_process_video(pipeline, data),
            'process_frame': lambda: handle_process_frame(pipeline, data),
            'process_image': lambda: handle_process_image(pipeline, data),
            'process_extracted_frames': lambda: handle_process_extracted_frames(pipeline, data),
            'embed_text': lambda: handle_embed_text(pipeline, data),
            'embed_image': lambda: handle_embed_image(pipeline, data),
            'caption': lambda: handle_caption(pipeline, data),
            'compare_frames': lambda: handle_compare_frames(pipeline, data),
            'memory_stats': lambda: {'output': pipeline.get_memory_stats()},
        }

        if operation in handlers:
            result = handlers[operation]()

            # Cleanup after GPU-intensive operations (data saved to R2)
            if operation in _CLEANUP_OPERATIONS:
                print(f"[Handler] Running cleanup after {operation}")
                pipeline.cleanup_after_request()

            return result

        return {'error': f'Unknown operation: {operation}'}

    except Exception as e:
        import traceback
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
        }

    finally:
        # Always clear GPU cache on exit (even on error)
        try:
            from utils.memory import clear_gpu_cache
            clear_gpu_cache()
        except Exception:
            pass
