"""
Handler utility functions.

Helper functions for processing request data (images, videos, ZIPs, base64 decoding).
"""

import base64
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from PIL import Image


def get_video_file(data: Dict) -> Optional[str]:
    """Get video file from request data."""
    if 'video_base64' in data:
        video_bytes = base64.b64decode(data['video_base64'])
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            return f.name

    if 'video_url' in data:
        import urllib.request
        url = data['video_url']
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            with urllib.request.urlopen(req) as response:
                f.write(response.read())
            return f.name

    if 'video_path' in data:
        return data['video_path']

    return None


def get_image(data: Dict) -> Optional[Image.Image]:
    """Get PIL Image from request data."""
    if 'image_base64' in data:
        return decode_base64_image(data['image_base64'])

    if 'image_url' in data:
        import urllib.request
        url = data['image_url']
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(req) as response:
            image_bytes = response.read()
            return Image.open(BytesIO(image_bytes)).convert('RGB')

    return None


def decode_base64_image(b64_string: str) -> Optional[Image.Image]:
    """Decode base64 string to PIL Image."""
    if not b64_string:
        return None

    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]

    image_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_bytes)).convert('RGB')


def get_zip_file(data: Dict) -> Optional[str]:
    """
    Get ZIP file from request data.

    Supports:
    - zip_url: URL to download ZIP from
    - zip_base64: Base64 encoded ZIP data
    - zip_path: Local file path (for testing)

    Returns:
        Path to temp file containing ZIP, or None if no ZIP provided
    """
    if 'zip_base64' in data:
        zip_bytes = base64.b64decode(data['zip_base64'])
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            f.write(zip_bytes)
            return f.name

    if 'zip_url' in data:
        import urllib.request
        url = data['zip_url']
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
            with urllib.request.urlopen(req) as response:
                f.write(response.read())
            return f.name

    if 'zip_path' in data:
        return data['zip_path']

    return None


def extract_frames_from_zip(
    zip_path: str,
    include_metadata: bool = True,
) -> Tuple[List[Image.Image], Optional[Dict[int, Dict]]]:
    """
    Extract frames and optional metadata from ZIP file.

    Expected ZIP structure:
        frame_0001/frame_0001.png
        frame_0001/frame_0001.json (optional)
        frame_0002/frame_0002.png
        ...

    Args:
        zip_path: Path to ZIP file
        include_metadata: Whether to parse JSON metadata files

    Returns:
        Tuple of:
        - frames: List of PIL Images (sorted by frame number)
        - metadata: Dict mapping frame_number (1-indexed) -> metadata dict,
                   or None if include_metadata is False
    """
    frames = []
    metadata = {} if include_metadata else None
    frame_data = []  # List of (frame_num, image, meta) for sorting

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Get all files in ZIP
        all_names = zf.namelist()

        # Find all frame folders (e.g., "frame_0001/")
        frame_dirs = set()
        for name in all_names:
            parts = Path(name).parts
            if parts and parts[0].startswith('frame_'):
                frame_dirs.add(parts[0])

        frame_dirs = sorted(frame_dirs)
        print(f"[extract_frames_from_zip] Found {len(frame_dirs)} frame directories")

        for frame_dir in frame_dirs:
            # Extract frame number from directory name
            try:
                frame_num = int(frame_dir.replace('frame_', ''))
            except ValueError:
                print(f"[extract_frames_from_zip] Skipping non-numeric frame dir: {frame_dir}")
                continue

            # Read PNG image
            png_path = f"{frame_dir}/{frame_dir}.png"
            image = None
            try:
                with zf.open(png_path) as img_file:
                    image = Image.open(img_file).convert('RGB')
                    image = image.copy()  # Copy to release file handle
            except KeyError:
                # Try alternative naming: frame_XXXX.png without nested folder
                alt_png_path = f"{frame_dir}.png"
                try:
                    with zf.open(alt_png_path) as img_file:
                        image = Image.open(img_file).convert('RGB')
                        image = image.copy()
                except KeyError:
                    print(f"[extract_frames_from_zip] Warning: No PNG found for {frame_dir}")
                    continue

            # Read JSON metadata if requested
            frame_meta = None
            if include_metadata:
                json_path = f"{frame_dir}/{frame_dir}.json"
                try:
                    with zf.open(json_path) as json_file:
                        frame_meta = json.load(json_file)
                except KeyError:
                    # JSON is optional
                    pass
                except json.JSONDecodeError as e:
                    print(f"[extract_frames_from_zip] Warning: Invalid JSON in {json_path}: {e}")

            frame_data.append((frame_num, image, frame_meta))

    # Sort by frame number
    frame_data.sort(key=lambda x: x[0])

    # Build output lists
    for frame_num, image, frame_meta in frame_data:
        frames.append(image)
        if include_metadata and frame_meta is not None:
            metadata[frame_num] = frame_meta

    print(f"[extract_frames_from_zip] Extracted {len(frames)} frames, {len(metadata) if metadata else 0} with metadata")

    return frames, metadata
