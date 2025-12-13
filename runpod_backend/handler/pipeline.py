"""
Pipeline and R2 client management.

Lazy-loaded singletons for the STPE pipeline and R2 storage client.
"""

import os
from typing import Optional

_pipeline = None
_r2_client = None


def get_pipeline():
    """Lazy load the STPE pipeline (models load on first use)."""
    global _pipeline
    if _pipeline is None:
        from pipeline import STPEPipeline
        from config import STPEConfig

        config = STPEConfig()
        _pipeline = STPEPipeline(config)
        _pipeline._vision_only = os.environ.get('VISION_ONLY', 'false').lower() == 'true'

        index_path = os.environ.get('INDEX_PATH')
        if index_path and os.path.exists(index_path):
            _pipeline.load_index(index_path)

        print("Pipeline initialized - models will load on first request")

    return _pipeline


def get_r2_client():
    """Get R2 storage client (lazy loaded)."""
    global _r2_client
    if _r2_client is None:
        try:
            from utils.r2_storage import get_r2_client as _get_r2
            _r2_client = _get_r2()
        except Exception as e:
            print(f"[Handler] R2 client not available: {e}")
            return None
    return _r2_client
