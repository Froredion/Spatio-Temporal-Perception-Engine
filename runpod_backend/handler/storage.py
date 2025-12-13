"""
R2 storage handlers.

Handles saving/loading analysis results from Cloudflare R2.
"""

import json
from typing import Dict, Any, Optional

from .pipeline import get_r2_client


def save_analysis_to_r2(video_id: str, output: Dict[str, Any]) -> Optional[str]:
    """
    Save video analysis results to R2 for frontend retrieval.

    Saves to: videos/processed/{video_id}/analysis.json

    Returns:
        Public URL of saved analysis, or None if R2 not configured
    """
    r2 = get_r2_client()
    if r2 is None:
        print(f"[Handler] Skipping R2 save - client not configured")
        return None

    try:
        key = f"videos/processed/{video_id}/analysis.json"
        analysis_json = json.dumps(output, indent=2)
        url = r2.upload_bytes(
            analysis_json.encode('utf-8'),
            key,
            content_type='application/json'
        )
        print(f"[Handler] Saved analysis to R2: {key}")
        return url
    except Exception as e:
        print(f"[Handler] Failed to save analysis to R2: {e}")
        return None


def load_analysis_from_r2(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Load video analysis results from R2.

    Returns:
        Analysis dict, or None if not found
    """
    r2 = get_r2_client()
    if r2 is None:
        return None

    # Try multiple possible paths
    paths_to_try = [
        f"videos/processed/{video_id}/analysis.json",
        f"{r2.bucket_name}/videos/processed/{video_id}/analysis.json",
    ]

    for key in paths_to_try:
        try:
            print(f"[Handler] Trying to load analysis from: {key}")
            if r2.file_exists(key):
                print(f"[Handler] Found analysis at: {key}")
                data = r2.download_bytes(key)
                return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"[Handler] Error loading from {key}: {e}")
            continue

    print(f"[Handler] Analysis not found for video: {video_id}")
    return None


def handle_get_analysis(data: Dict) -> Dict[str, Any]:
    """
    Get saved video analysis from R2.

    Input:
        video_id: ID of the processed video

    Returns:
        Saved analysis data
    """
    video_id = data.get('video_id')

    if not video_id:
        return {'error': 'No video_id provided'}

    analysis = load_analysis_from_r2(video_id)

    if analysis is None:
        return {'error': f'Analysis not found for video: {video_id}'}

    return {'output': analysis}


def handle_list_videos(data: Dict) -> Dict[str, Any]:
    """
    List all processed videos in R2.

    Returns:
        List of video IDs with their analysis URLs
    """
    r2 = get_r2_client()
    if r2 is None:
        # R2 not configured - return empty list instead of error
        print("[Handler] R2 not configured, returning empty video list")
        return {'output': {'videos': [], 'count': 0}}

    try:
        # First, list root to debug the bucket structure
        print(f"[Handler] Debugging: listing root of bucket...")
        root_files = r2.list_files('', sort_by_modified=False)
        print(f"[Handler] Root files (first 20): {root_files[:20]}")

        # Try different possible prefixes
        prefixes_to_try = [
            'videos/processed/',
            f'{r2.bucket_name}/videos/processed/',
            'st-perception-engine/videos/processed/',
        ]

        files = []
        for prefix in prefixes_to_try:
            print(f"[Handler] Trying prefix: {prefix}")
            files = r2.list_files(prefix, sort_by_modified=True)
            if files:
                print(f"[Handler] Found {len(files)} files with prefix: {prefix}")
                break
            print(f"[Handler] No files found with prefix: {prefix}")

        videos = []
        for key in files:
            if key.endswith('/analysis.json'):
                # Path could be:
                # - videos/processed/{video_id}/analysis.json
                # - st-perception-engine/videos/processed/{video_id}/analysis.json
                # Extract video_id as the folder right before analysis.json
                parts = key.split('/')
                if len(parts) >= 2:
                    # video_id is the second-to-last part (folder containing analysis.json)
                    video_id = parts[-2]
                    print(f"[Handler] Found video: {video_id} from path: {key}")
                    videos.append({
                        'video_id': video_id,
                        'analysis_url': r2.get_public_url(key),
                    })

        print(f"[Handler] Returning {len(videos)} processed videos")
        return {'output': {'videos': videos, 'count': len(videos)}}

    except Exception as e:
        # Return empty list on error instead of failing
        print(f"[Handler] Error listing videos: {e}")
        return {'output': {'videos': [], 'count': 0}}
