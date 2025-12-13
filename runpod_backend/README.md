# STPE - Spatio-Temporal Perception Engine

RunPod backend for video understanding using a 3-model AI stack.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOCAL PC (Development)                         │
│  • Write and test code                                                   │
│  • Send requests to RunPod for AI processing                             │
│  • Receive results (embeddings, captions, etc.)                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                    HTTP/WebSocket API Communication
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RUNPOD (AI Inference Server)                      │
│  • Run STPE models (DINOv3, SAM-3, Qwen3-VL)                            │
│  • Process video frames → return embeddings                              │
└─────────────────────────────────────────────────────────────────────────┘
```

## The 3-Model Stack

| Model | Model ID | Role | Embedding Dim |
|-------|----------|------|---------------|
| **DINOv3-7B** | `facebook/dinov3-vit7b16-pretrain-lvd1689m` | Vision Backbone (3-level processing) | 4096 |
| **SAM-3** | `facebook/sam3` | Instance Segmentation | - |
| **Qwen3-VL** | `Qwen/Qwen3-VL-8B-Instruct` | LLM Reasoning | - |

## 3-Level Vision Processing

1. **Level 1: Global** - Scene-level 4096-dim embedding
2. **Level 2: Objects** - Auto-segmentation + per-object embeddings
3. **Level 3: Dense** - High-resolution feature maps for tracking

## Temporal Modules

- Temporal Positional Encoding
- Object Tracking (DINOv3 feature-based)
- Motion Feature Extraction
- Cross-frame Temporal Attention
- Temporal Pooling for clip embeddings
- Scene Graph (object relationships)

## Quick Start

### Local Testing

```python
from runpod_backend import STPEPipeline

# Initialize pipeline
pipeline = STPEPipeline()
pipeline.load_models()

# Process video
result = pipeline.process_video("path/to/video.mp4")
print(f"Processed {len(result.frames)} frames")
print(f"Clip embedding shape: {result.clip_embedding.shape}")

# Cleanup
pipeline.unload_models()
```

### RunPod Deployment

1. Build Docker image:
```bash
docker build -t stpe-runpod .
```

2. Push to registry:
```bash
docker push your-registry/stpe-runpod:latest
```

3. Create RunPod Serverless endpoint with the image.

### API Usage

```python
import requests
import base64

RUNPOD_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
RUNPOD_KEY = "your-api-key"

# Process video
with open("video.mp4", "rb") as f:
    video_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    RUNPOD_URL,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
    json={
        "input": {
            "operation": "process_video",
            "data": {
                "video_base64": video_b64,
                "fps": 2.0,
                "generate_captions": True,
            }
        }
    }
)

result = response.json()["output"]
print(f"Video ID: {result['video_id']}")
print(f"Frames: {result['num_frames']}")
```

## API Operations

| Operation | Description |
|-----------|-------------|
| `health` | Health check |
| `process_video` | Process video file |
| `process_frame` | Process single frame |
| `embed_image` | Get image embedding |
| `caption` | Generate caption |
| `compare_frames` | Compare two frames |
| `memory_stats` | Get GPU memory stats |

## Hardware Requirements

| Setting | Recommended |
|---------|-------------|
| GPU | 1x A100 SXM 80GB |
| RAM | 117GB |
| Container Disk | 20 GB |
| Volume Disk | 100 GB |

### Memory Usage (FP16)

| Model | VRAM |
|-------|------|
| DINOv3-7B | ~18-22 GB |
| SAM-3 | ~8-12 GB |
| Qwen3-VL | ~16-20 GB |
| **Total** | **~42-54 GB** |

## Directory Structure

```
runpod_backend/
├── __init__.py
├── config.py              # Configuration dataclasses
├── pipeline.py            # Main STPE pipeline
├── handler.py             # RunPod serverless handler
├── models/
│   ├── __init__.py
│   ├── loader.py          # Unified model loader
│   ├── dinov3.py          # DINOv3 wrapper
│   ├── sam3/              # SAM-3 wrapper
│   └── qwen3_vision.py    # Qwen3-VL wrapper
├── processing/
│   ├── __init__.py
│   ├── level1_global.py   # Global embedding
│   ├── level2_objects.py  # Object segmentation
│   ├── level3_dense.py    # Dense feature maps
│   └── vision_pipeline.py # Unified vision pipeline
├── temporal/
│   ├── __init__.py
│   ├── positional_encoding.py
│   ├── tracker.py         # Object tracking
│   ├── motion.py          # Motion features
│   ├── attention.py       # Temporal attention
│   ├── pooling.py         # Temporal pooling
│   └── scene_graph.py     # Scene graph GNN
├── utils/
│   ├── __init__.py
│   ├── frame_extraction.py
│   ├── image_utils.py
│   └── clustering.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black runpod_backend/
isort runpod_backend/
```

## Notes

- Video processing results are saved to R2 cloud storage
