#!/bin/bash

# MT3D Setup Script
# This script creates a reproducible conda environment for MotionTransfer3D

set -e  # Exit on error

echo "=========================================="
echo "MotionTransfer3D Environment Setup"
echo "=========================================="
echo ""
echo "IMPORTANT: The GPU on which you intend to run must be available"
echo "during environment setup. The build process for diff-gaussian-rasterization"
echo "and other CUDA packages needs to detect the GPU architecture to compile"
echo "for the correct CUDA compute capability."
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Step 1: Create environment from environment.yml
echo ""
echo "Step 1: Creating conda environment from environment.yml..."
if conda env list | grep -q "^mt3d "; then
    echo "Error: Environment 'mt3d' already exists."
    echo "Please remove it first with: conda env remove -n mt3d"
    exit 1
fi
echo "  Creating new environment 'mt3d'..."
conda env create -f environment.yml

# Step 2: Activate environment
echo ""
echo "Step 2: Activating mt3d environment..."
eval "$(conda shell.bash hook)"
conda activate mt3d

# Step 3: Install local packages that require compilation
# These packages need --no-build-isolation because they require torch during build
echo ""
echo "Step 3: Installing local packages (diff-gaussian-rasterization, simple-knn, KNN_CUDA)..."
echo "  - Installing diff-gaussian-rasterization..."
pip install --no-build-isolation ./diff-gaussian-rasterization

echo "  - Installing simple-knn..."
pip install --no-build-isolation ./SC4D/simple-knn

echo "  - Installing KNN_CUDA..."
pip install --no-build-isolation ./KNN_CUDA

# Step 4: Install packages from git repositories
echo ""
echo "Step 4: Installing packages from git repositories..."
echo "  - Installing kiui..."
pip install git+https://github.com/ashawkey/kiuikit

echo "  - Installing pytorch3d..."
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

# Step 5: Install chamferdist (requires PyTorch to be installed first)
echo ""
echo "Step 5: Installing chamferdist package..."
echo "  - Cloning chamferdist repository..."
if [ -d "chamferdist" ]; then
    echo "    chamferdist directory already exists, skipping clone..."
else
    git clone https://github.com/krrish94/chamferdist.git
fi
cd chamferdist
pip install --no-build-isolation .
cd ..

# Step 6: Verify installation
echo ""
echo "Step 6: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate mt3d"
echo ""
