"""
Embedding handlers.

Handles image embedding and caption generation.
"""

from typing import Dict, Any

from .utils import get_image


def handle_embed_text(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Text embedding is no longer supported.
    """
    return {'error': 'Text embedding not supported. Use VLM for text understanding.'}


def handle_embed_image(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Get image embedding.

    Input:
        image_base64: Base64 encoded image

    Returns:
        DINOv3 image embeddings
    """
    image = get_image(data)

    if image is None:
        return {'error': 'No image provided'}

    dinov3_emb = pipeline.models.dinov3.encode(image)

    return {
        'output': {
            'dinov3_embedding': dinov3_emb['global_embedding'].cpu().tolist(),
        }
    }


def handle_caption(pipeline, data: Dict) -> Dict[str, Any]:
    """
    Generate caption for image.

    Input:
        image_base64: Base64 encoded image
        detail_level: 'brief', 'normal', or 'detailed' (default 'normal')

    Returns:
        Caption string
    """
    if pipeline.models.vlm is None:
        return {'error': 'Qwen3-VL not loaded (vision_only mode)'}

    image = get_image(data)
    detail_level = data.get('detail_level', 'normal')

    if image is None:
        return {'error': 'No image provided'}

    caption = pipeline.models.vlm.caption(image, detail_level)

    return {'output': {'caption': caption}}
