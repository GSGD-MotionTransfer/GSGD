#!/bin/bash

# CoTracker Setup Script
# This script creates a reproducible conda environment for CoTracker

set -e  # Exit on error

echo "=========================================="
echo "CoTracker Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^cotracker "; then
    echo "Error: Environment 'cotracker' already exists."
    echo "Please remove it first with: conda env remove -n cotracker"
    exit 1
fi

# Step 1: Create environment
echo ""
echo "Step 1: Creating conda environment 'cotracker' with Python 3.9..."
conda create -n cotracker python=3.9 -y

# Step 2: Activate environment
echo ""
echo "Step 2: Activating cotracker environment..."
eval "$(conda shell.bash hook)"
conda activate cotracker

# Step 3: Change to co-tracker directory
echo ""
echo "Step 3: Installing CoTracker package..."
cd co-tracker

# Step 4: Install CoTracker in editable mode
echo "  - Installing CoTracker in editable mode..."
pip install -e .

# Step 5: Install required Python packages
echo ""
echo "Step 4: Installing required Python packages..."
pip install matplotlib flow_vis tqdm tensorboard

# Step 6: Create checkpoints directory and download model
echo ""
echo "Step 5: Downloading model checkpoint..."
mkdir -p checkpoints
cd checkpoints

echo "  - Downloading scaled_offline.pth from Hugging Face..."
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
cd ..

# Step 7: Install additional packages
echo ""
echo "Step 6: Installing additional packages..."
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
pip install imageio imageio[ffmpeg]
pip install lpips
pip install rembg onnxruntime
pip install torchmetrics
pip install pyyaml
pip install einops
pip install wandb
pip install pandas

# Step 8: Verify installation
echo ""
echo "Step 7: Verifying installation..."
python -c "import cotracker; print('CoTracker imported successfully')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate cotracker"
echo ""
echo "Environment location:"
conda info --envs | grep cotracker