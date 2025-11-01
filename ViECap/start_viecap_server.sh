#!/bin/bash

# Get the directory where this script is located
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: Virtual environment not found at .venv/bin/activate"
fi

# Set checkpoint directory if provided
if [ ! -z "$1" ]; then
    export VIECAP_CKPT_DIR="$1"
    echo "Checkpoint directory set to: $VIECAP_CKPT_DIR"
fi

# Start the server
echo "Starting ViECap server on port xxxx..."
python viecap_server.py