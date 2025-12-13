#!/bin/bash
# SAM-3 Environment Setup Script
# Creates a dedicated conda environment for SAM-3

set -e

SAM3_ENV="sam3"
SAM3_DIR="/workspace/ClipSearchAI/runpod_backend/models/sam3/sam3_repo"

# Initialize conda
if [ -f "/workspace/miniconda/etc/profile.d/conda.sh" ]; then
    source /workspace/miniconda/etc/profile.d/conda.sh
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Create SAM-3 conda environment if it doesn't exist
if ! conda env list | grep -q "^$SAM3_ENV "; then
    echo "Creating SAM-3 conda environment..."
    conda create -n $SAM3_ENV python=3.12 -y
fi

# Activate SAM-3 environment
echo "Activating $SAM3_ENV environment..."
conda activate $SAM3_ENV

# Install PyTorch with CUDA 12.6 support (SAM-3 requirement)
echo "Installing PyTorch 2.7 with CUDA 12.6..."
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install SAM-3 from local repo
echo "Installing SAM-3..."
cd $SAM3_DIR
pip install -e .

# Install additional dependencies for inference
pip install matplotlib

echo "SAM-3 environment setup complete!"
echo "To activate: conda activate $SAM3_ENV"
