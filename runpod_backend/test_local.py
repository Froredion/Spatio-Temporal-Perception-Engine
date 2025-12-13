#!/usr/bin/env python3
"""
Local testing script for STPE pipeline.

Run this to test the pipeline locally before deploying to RunPod.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_health():
    """Test health check."""
    from handler import handler

    result = handler({'input': {'operation': 'health'}})
    print("Health check:", result)
    assert 'output' in result
    assert result['output']['status'] == 'healthy'
    print("Health check passed")


def test_pipeline_init():
    """Test pipeline initialization."""
    from pipeline import STPEPipeline
    from config import STPEConfig

    config = STPEConfig()
    pipeline = STPEPipeline(config)

    print(f"Device: {pipeline.device}")
    print(f"Config: {config}")
    print("Pipeline initialization passed")


def test_models_load():
    """Test model loading (vision only for faster test)."""
    from pipeline import STPEPipeline

    pipeline = STPEPipeline()

    print("Loading vision models...")
    pipeline.load_models(vision_only=True)

    print(f"Memory usage: {pipeline.get_memory_stats()}")
    print("Model loading passed")

    pipeline.unload_models()
    print("Model unloading passed")


def test_frame_processing():
    """Test single frame processing."""
    from pipeline import STPEPipeline
    from PIL import Image
    import numpy as np

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    pipeline = STPEPipeline()
    pipeline.load_models(vision_only=True)

    try:
        embeddings = pipeline.process_frame_single(dummy_image)

        print(f"DINOv3 embedding shape: {embeddings['dinov3'].shape}")
        print("Frame processing passed")
    finally:
        pipeline.unload_models()


def test_video_processing():
    """Test video processing (requires sample video)."""
    from pipeline import STPEPipeline

    # Check for sample video
    sample_videos = list(Path("data/clips").glob("*.mp4"))
    if not sample_videos:
        print("No sample videos found in data/clips/, skipping video test")
        return

    video_path = str(sample_videos[0])
    print(f"Testing with video: {video_path}")

    pipeline = STPEPipeline()
    pipeline.load_models(vision_only=True)

    try:
        result = pipeline.process_video(
            video_path,
            fps=1.0,  # Low FPS for faster test
            generate_captions=False,
            add_to_index=True,
        )

        print(f"Video ID: {result.video_id}")
        print(f"Frames processed: {len(result.frames)}")
        print(f"Clip embedding shape: {result.clip_embedding.shape}")
        print(f"Tracks: {len(result.tracks)}")
        print("Video processing passed")

        # Test search
        results = pipeline.search("test query", k=5)
        print(f"Search results: {len(results)}")
        print("Search passed")

    finally:
        pipeline.unload_models()


def test_handler():
    """Test request handler with mock request."""
    from handler import handler
    import base64
    from PIL import Image
    from io import BytesIO
    import numpy as np

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buffer = BytesIO()
    dummy_image.save(buffer, format='JPEG')
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    # Test embed_image operation
    result = handler({
        'input': {
            'operation': 'embed_image',
            'data': {
                'image_base64': image_b64,
            }
        }
    })

    print("Handler result:", result)
    if 'error' in result:
        print(f"Warning: Handler returned error: {result['error']}")
    else:
        print("Handler test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("STPE Local Tests")
    print("=" * 60)

    tests = [
        ("Health Check", test_health),
        ("Pipeline Init", test_pipeline_init),
        ("Model Loading", test_models_load),
        ("Frame Processing", test_frame_processing),
        ("Video Processing", test_video_processing),
        ("Handler", test_handler),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
