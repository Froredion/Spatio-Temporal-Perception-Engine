"""
HTTP Server for Dedicated GPU Pod

Run this instead of handler.py when using a dedicated RunPod pod.
Exposes the same API via FastAPI/HTTP.
"""

import os
import sys
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

# Add current directory to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from handler import handler, get_pipeline

app = FastAPI(title="STPE API", description="Spatio-Temporal Perception Engine")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
)

# Job storage for async processing
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
async def preload_models():
    """Preload all models on server startup."""
    import os
    print("[Startup] Preloading models...")
    pipeline = get_pipeline()

    # Check if vision_only mode
    vision_only = os.environ.get('VISION_ONLY', 'false').lower() == 'true'
    print(f"[Startup] Vision only mode: {vision_only}")

    # Load all models
    pipeline.load_models(vision_only=vision_only)

    # Verify models are loaded
    print(f"[Startup] Models loaded state:")
    print(f"  - DINOv3: {pipeline.models.dinov3 is not None}")
    print(f"  - SAM-3: {pipeline.models.sam3 is not None}")
    print(f"  - VLM: {pipeline.models.vlm is not None}")
    print(f"  - _models_loaded flag: {pipeline._models_loaded}")
    print("[Startup] All models preloaded and ready!")


def _run_job(job_id: str, operation: str, data: Dict[str, Any]):
    """Run a job in background thread."""
    try:
        _jobs[job_id]["status"] = "processing"
        print(f"[Job {job_id}] Starting {operation}...")

        result = handler({"input": {"operation": operation, "data": data}})

        if "error" in result:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = result["error"]
            print(f"[Job {job_id}] Failed: {result['error']}")
        else:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
            print(f"[Job {job_id}] Completed successfully")

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        print(f"[Job {job_id}] Exception: {e}")


@app.post("/jobs/submit")
async def submit_job(data: Dict[str, Any]):
    """
    Submit a long-running job (like video processing).

    Returns job_id immediately. Poll /jobs/{job_id} for status.

    Example:
        POST /jobs/submit
        {"operation": "process_video", "data": {"video_url": "...", "fps": 2.0}}

    Returns:
        {"job_id": "abc123", "status": "queued"}
    """
    operation = data.get("operation", "process_video")
    job_data = data.get("data", {})

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "operation": operation,
    }

    # Run in background thread
    _executor.submit(_run_job, job_id, operation, job_data)

    print(f"[Job {job_id}] Queued: {operation}")
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Get job status and result.

    Returns:
        - status: "queued", "processing", "completed", or "failed"
        - result: (only if completed) the operation result
        - error: (only if failed) error message
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    response = {"job_id": job_id, "status": job["status"]}

    if job["status"] == "completed":
        response["result"] = job.get("result")
    elif job["status"] == "failed":
        response["error"] = job.get("error")

    return response


class RequestInput(BaseModel):
    operation: str = "health"
    data: Optional[Dict[str, Any]] = {}


class RequestBody(BaseModel):
    input: RequestInput


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "service": "STPE"}


@app.post("/")
async def process(body: RequestBody):
    """
    Main API endpoint - same format as RunPod serverless.

    Example requests:

    Health check:
        {"input": {"operation": "health"}}

    Process frame:
        {"input": {"operation": "embed_image", "data": {"image_base64": "..."}}}

    Search:
        {"input": {"operation": "search", "data": {"query": "sword attack", "k": 10}}}
    """
    event = {"input": body.input.model_dump()}
    result = handler(event)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result)

    return result


@app.post("/health")
async def health():
    """Dedicated health endpoint."""
    result = handler({"input": {"operation": "health"}})
    return result


@app.post("/process_video")
async def process_video(data: Dict[str, Any]):
    """Convenience endpoint for video processing."""
    result = handler({"input": {"operation": "process_video", "data": data}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/process_extracted_frames")
async def process_extracted_frames(data: Dict[str, Any]):
    """
    Convenience endpoint for extracted frames processing.

    Processes pre-extracted frame datasets (ZIP files with PNG + JSON metadata).
    Supports Roblox game state metadata for enhanced captioning.

    Request body:
        zip_url: URL to download ZIP from (optional)
        zip_base64: Base64 encoded ZIP (optional)
        fps: Playback FPS for temporal calculations (default 10.0)
        include_roblox_metadata: Whether to parse frame JSON files (default True)
        generate_captions: Whether to generate captions (default True)
        dense_captions: Whether to generate dense per-frame captions (default True)
    """
    result = handler({"input": {"operation": "process_extracted_frames", "data": data}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/embed_text")
async def embed_text(data: Dict[str, Any]):
    """Convenience endpoint for text embedding."""
    result = handler({"input": {"operation": "embed_text", "data": data}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/embed_image")
async def embed_image(data: Dict[str, Any]):
    """Convenience endpoint for image embedding."""
    result = handler({"input": {"operation": "embed_image", "data": data}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.post("/caption")
async def caption(data: Dict[str, Any]):
    """Convenience endpoint for image captioning."""
    result = handler({"input": {"operation": "caption", "data": data}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/videos")
async def list_videos():
    """List all processed videos."""
    result = handler({"input": {"operation": "list_videos", "data": {}}})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result)
    return result


@app.get("/videos/{video_id}")
async def get_video_analysis(video_id: str):
    """Get analysis for a specific video."""
    result = handler({"input": {"operation": "get_analysis", "data": {"video_id": video_id}}})
    if "error" in result:
        raise HTTPException(status_code=404, detail=result)
    return result


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting STPE server on port {port}...")
    print(f"API docs available at: http://0.0.0.0:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
