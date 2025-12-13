"""
Frame Extraction Utilities

Extract frames from video files at specified FPS.
"""

import os
from typing import List, Optional, Tuple
from PIL import Image
import subprocess
import tempfile


def extract_frames(
    video_path: str,
    fps: float = 2.0,
    max_frames: Optional[int] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> List[Image.Image]:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        max_frames: Maximum number of frames (None for all)
        output_size: Optional (width, height) to resize frames

    Returns:
        List of PIL Images
    """
    # Try to use decord (fastest)
    try:
        return _extract_with_decord(video_path, fps, max_frames, output_size)
    except ImportError:
        pass

    # Fallback to OpenCV
    try:
        return _extract_with_opencv(video_path, fps, max_frames, output_size)
    except ImportError:
        pass

    # Last resort: ffmpeg
    return _extract_with_ffmpeg(video_path, fps, max_frames, output_size)


def _extract_with_decord(
    video_path: str,
    fps: float,
    max_frames: Optional[int],
    output_size: Optional[Tuple[int, int]],
) -> List[Image.Image]:
    """Extract frames using decord (GPU accelerated)."""
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()

    # Calculate frame indices
    frame_interval = int(video_fps / fps)
    total_frames = len(vr)

    indices = list(range(0, total_frames, frame_interval))

    if max_frames:
        indices = indices[:max_frames]

    # Extract frames
    frames = vr.get_batch(indices).asnumpy()

    # Convert to PIL
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        if output_size:
            img = img.resize(output_size, Image.LANCZOS)
        pil_frames.append(img)

    return pil_frames


def _extract_with_opencv(
    video_path: str,
    fps: float,
    max_frames: Optional[int],
    output_size: Optional[Tuple[int, int]],
) -> List[Image.Image]:
    """Extract frames using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            if output_size:
                img = img.resize(output_size, Image.LANCZOS)

            frames.append(img)

            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def _extract_with_ffmpeg(
    video_path: str,
    fps: float,
    max_frames: Optional[int],
    output_size: Optional[Tuple[int, int]],
) -> List[Image.Image]:
    """Extract frames using ffmpeg subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
        ]

        if output_size:
            cmd.extend(["-s", f"{output_size[0]}x{output_size[1]}"])

        if max_frames:
            cmd.extend(["-frames:v", str(max_frames)])

        cmd.append(output_pattern)

        subprocess.run(cmd, capture_output=True, check=True)

        # Load extracted frames
        frames = []
        frame_files = sorted([f for f in os.listdir(tmpdir) if f.endswith('.jpg')])

        for frame_file in frame_files:
            frame_path = os.path.join(tmpdir, frame_file)
            img = Image.open(frame_path).convert('RGB')
            frames.append(img.copy())  # Copy to avoid file handle issues

        return frames


def extract_frames_batch(
    video_paths: List[str],
    fps: float = 2.0,
    max_frames_per_video: Optional[int] = None,
) -> List[List[Image.Image]]:
    """
    Extract frames from multiple videos.

    Args:
        video_paths: List of video file paths
        fps: Frames per second to extract
        max_frames_per_video: Maximum frames per video

    Returns:
        List of frame lists, one per video
    """
    all_frames = []

    for video_path in video_paths:
        frames = extract_frames(video_path, fps, max_frames_per_video)
        all_frames.append(frames)

    return all_frames


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info (fps, duration, width, height, frame_count)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0

        cap.release()
        return info

    except ImportError:
        # Fallback to ffprobe
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path,
        ]

        import json
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                fps_parts = stream.get('r_frame_rate', '0/1').split('/')
                fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else 0

                return {
                    'fps': fps,
                    'width': stream.get('width', 0),
                    'height': stream.get('height', 0),
                    'frame_count': int(stream.get('nb_frames', 0)),
                    'duration': float(stream.get('duration', 0)),
                }

        return {}
