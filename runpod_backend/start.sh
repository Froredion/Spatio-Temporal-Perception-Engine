#!/bin/bash
# STPE RunPod Startup Script
# Run this after pod restart: /workspace/ClipSearchAI/runpod_backend/start.sh

set -e  # Exit on error

# Install Miniconda if not available
CONDA_DIR="/workspace/miniconda"
if ! command -v conda &> /dev/null && [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "Installing Miniconda to $CONDA_DIR..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $CONDA_DIR
    rm /tmp/miniconda.sh
    echo "Miniconda installed successfully"
fi

# Initialize conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source $CONDA_DIR/etc/profile.d/conda.sh
fi

# Ensure conda is in PATH
export PATH="$CONDA_DIR/bin:$PATH"

# Cache HuggingFace models to /workspace so they persist across restarts
export HF_HOME=/workspace/huggingface_cache
mkdir -p $HF_HOME

# HuggingFace authentication
if [ -z "$HF_TOKEN" ] && [ -f /workspace/.hf_token ]; then
    export HF_TOKEN=$(cat /workspace/.hf_token)
    echo "Loaded HF_TOKEN from /workspace/.hf_token"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Create /workspace/.hf_token with your token"
else
    export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
    echo "HuggingFace token configured"
fi

# Accept conda Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Go to project and pull latest
cd /workspace/ClipSearchAI
git pull

#############################################
# ENVIRONMENT 1: SAM-3 (PyTorch 2.7 / CUDA 12.6)
#############################################
SAM3_ENV="sam3"
SAM3_PID_FILE="/workspace/sam3_server.pid"
SAM3_LOG_FILE="/workspace/sam3_server.log"

# Clone SAM-3 repo if it doesn't exist
SAM3_REPO_DIR="/workspace/ClipSearchAI/runpod_backend/models/sam3/sam3_repo"
if [ ! -d "$SAM3_REPO_DIR" ] || [ ! -f "$SAM3_REPO_DIR/pyproject.toml" ]; then
    echo "Cloning SAM-3 repository..."
    rm -rf "$SAM3_REPO_DIR"
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_REPO_DIR"
    echo "SAM-3 repository cloned"
fi

# Create SAM-3 conda environment if it doesn't exist
if ! conda env list | grep -q "^$SAM3_ENV "; then
    echo "Creating SAM-3 conda environment..."
    conda create -n $SAM3_ENV python=3.12 -y
fi

# Always ensure SAM-3 dependencies are installed (in case env exists but packages missing)
echo "Ensuring SAM-3 dependencies are installed..."
if ! conda run -n $SAM3_ENV python -c "import sam3" 2>/dev/null; then
    echo "Installing PyTorch 2.7 with CUDA 12.6 for SAM-3..."
    conda run -n $SAM3_ENV pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

    echo "Installing SAM-3 from local repo with all dependencies..."
    conda run -n $SAM3_ENV pip install -e "$SAM3_REPO_DIR[notebooks,dev,train]"

    echo "SAM-3 environment setup complete"
else
    echo "SAM-3 already installed in $SAM3_ENV environment"
fi

# Kill existing SAM-3 server if running
if [ -f "$SAM3_PID_FILE" ]; then
    OLD_PID=$(cat "$SAM3_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing SAM-3 server (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$SAM3_PID_FILE"
fi

# Also kill any process holding port 9999 (in case PID file was stale)
echo "Checking for processes on port 9999..."
fuser -k 9999/tcp 2>/dev/null || true
sleep 2
# Double check and force kill if still in use
if lsof -i :9999 >/dev/null 2>&1; then
    echo "Port 9999 still in use, force killing..."
    lsof -t -i :9999 | xargs -r kill -9 2>/dev/null || true
    sleep 2
fi

# Start SAM-3 server in background
echo "Starting SAM-3 server in $SAM3_ENV environment..."
nohup conda run -n $SAM3_ENV --no-capture-output \
    python runpod_backend/models/sam3/sam3_server.py \
    --host 127.0.0.1 --port 9999 --device cuda \
    > "$SAM3_LOG_FILE" 2>&1 &
SAM3_PID=$!
disown $SAM3_PID
echo "$SAM3_PID" > "$SAM3_PID_FILE"
echo "SAM-3 server started with PID: $SAM3_PID"
echo "SAM-3 logs: $SAM3_LOG_FILE"

#############################################
# Install pip dependencies
#############################################
pip install -r runpod_backend/requirements.txt

#############################################
# Start Main Server
#############################################
cd runpod_backend
LOG_FILE="/workspace/server.log"
PID_FILE="/workspace/server.pid"

# Kill existing server if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID: $OLD_PID)..."
        kill "$OLD_PID"
        sleep 2
        kill -9 "$OLD_PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

echo "Starting main server in background..."
echo "Logs: $LOG_FILE"
nohup python server.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
disown $SERVER_PID
echo "$SERVER_PID" > "$PID_FILE"

echo ""
echo "============================================"
echo "All servers started!"
echo "============================================"
echo "Main server PID: $SERVER_PID (logs: $LOG_FILE)"
echo "SAM-3 server PID: $(cat $SAM3_PID_FILE) (logs: $SAM3_LOG_FILE)"
echo ""
echo "To view main logs: tail -f $LOG_FILE"
echo "To view SAM-3 logs: tail -f $SAM3_LOG_FILE"
echo "To stop all: kill \$(cat $PID_FILE) \$(cat $SAM3_PID_FILE)"
