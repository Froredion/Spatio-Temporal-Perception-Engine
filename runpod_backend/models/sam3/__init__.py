"""
SAM-3 Integration Module

Contains:
- sam3_wrapper.py: Client that communicates with SAM-3 server
- sam3_server.py: Server running in dedicated conda env
- sam3_repo/: Cloned SAM-3 repository from facebook/sam3
- setup_sam3.sh: Setup script for SAM-3 conda environment
"""

from models.sam3.sam3_wrapper import SAM3Model

__all__ = ["SAM3Model"]
