#!/bin/bash
# Motion Control Video Generation System - Automated Setup Script
# This script initializes the project environment and downloads necessary dependencies

set -e  # Exit on error

echo "=========================================="
echo "Motion Control Setup Script"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check if running on HPC
if command -v module &> /dev/null; then
    echo "✓ Detected HPC environment with module system"

    # Load required modules
    echo "Loading CUDA and Conda modules..."
    module load cuda/12.8.1 2>/dev/null || module load cuda/13.0.2 || echo "Warning: Could not load CUDA module"
    module load conda 2>/dev/null || echo "Warning: Could not load conda module"
else
    echo "! Not running on HPC - using locally installed tools"
fi

# Check for GPU availability
echo ""
echo "Checking GPU availability..."
if command -v sinfo &> /dev/null; then
    echo "SLURM partitions with GPUs:"
    sinfo -o "%P %G" | grep gpu || echo "No GPU partitions found"
else
    echo "SLURM not available - assuming direct GPU access"
fi

# Create __init__.py files for Python packages
echo ""
echo "Creating Python package structure..."
touch src/__init__.py
touch src/pipeline/__init__.py
touch src/utils/__init__.py

# Check for existing conda environment
echo ""
echo "Checking for motion_control conda environment..."
if conda env list | grep -q "^motion_control "; then
    echo "✓ motion_control environment exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n motion_control -y
        conda env create -f environment/environment.yml
    fi
else
    echo "Creating motion_control environment..."
    conda env create -f environment/environment.yml
fi

# Activate environment and install additional dependencies
echo ""
echo "Installing additional Python dependencies..."
eval "$(conda shell.bash hook)"
conda activate motion_control

pip install -r environment/requirements.txt

# Clone Wan2.1 repository if not exists
echo ""
echo "Setting up Wan2.1 model repository..."
if [ ! -d "models/wan21/.git" ]; then
    echo "Cloning Wan2.1 repository..."
    git clone https://github.com/Wan-Video/Wan2.1.git models/wan21
    cd models/wan21
    pip install -r requirements.txt 2>/dev/null || echo "Note: Install Wan2.1 requirements manually if needed"
    cd "$PROJECT_ROOT"
else
    echo "✓ Wan2.1 repository already cloned"
fi

# Clone CodeFormer for identity preservation
echo ""
echo "Setting up CodeFormer for identity preservation..."
if [ ! -d "models/CodeFormer/.git" ]; then
    echo "Cloning CodeFormer repository..."
    cd models
    git clone https://github.com/sczhou/CodeFormer.git
    cd CodeFormer
    pip install -r requirements.txt 2>/dev/null || echo "Note: Install CodeFormer requirements manually if needed"

    # Download pretrained models
    echo "Downloading CodeFormer pretrained models..."
    python scripts/download_pretrained_models.py facelib 2>/dev/null || echo "Note: Download CodeFormer models manually if needed"
    python scripts/download_pretrained_models.py CodeFormer 2>/dev/null || echo "Note: Download CodeFormer models manually if needed"
    cd "$PROJECT_ROOT"
else
    echo "✓ CodeFormer repository already cloned"
fi

# Check for Real-ESRGAN
echo ""
echo "Installing Real-ESRGAN for upscaling..."
pip install realesrgan 2>/dev/null || echo "Note: Install realesrgan manually if needed"

# Check for ffmpeg
echo ""
echo "Checking for ffmpeg..."
if command -v ffmpeg &> /dev/null; then
    echo "✓ ffmpeg is available"
else
    echo "! ffmpeg not found - load module or install manually"
    echo "  On HPC: module load ffmpeg"
fi

# Create example batch manifest
echo ""
echo "Creating example batch manifest..."
cat > data/batch_manifest.json <<EOF
{
  "jobs": [
    {
      "id": "example_001",
      "character_image": "data/inputs/character1.png",
      "reference_video": "data/inputs/reference1.mp4",
      "output_video": "data/outputs/result1.mp4",
      "duration": 10,
      "preserve_identity": true,
      "preserve_audio": true
    }
  ]
}
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Build PyTorch with SM_100 support (if using B200 GPUs):"
echo "   sbatch environment/build_pytorch_sm100.sh"
echo ""
echo "2. Download Wan2.1 I2V-14B-720P checkpoint:"
echo "   # Visit: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P"
echo "   # Download to: models/checkpoints/"
echo ""
echo "3. Activate the environment:"
echo "   conda activate motion_control"
echo ""
echo "4. Test the pipeline:"
echo "   python scripts/generate.py --help"
echo ""
echo "For more information, see README.md"
