"""
Health check handler.
"""

from typing import Dict, Any


def handle_health() -> Dict[str, Any]:
    """Health check endpoint."""
    import torch
    return {
        'output': {
            'status': 'healthy',
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
    }
