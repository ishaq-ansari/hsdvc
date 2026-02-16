#!/bin/bash
#SBATCH --job-name=build_pytorch_sm100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-b200
#SBATCH --account=pinaki.sarder
#SBATCH --gres=gpu:b200:1
#SBATCH --output=logs/pytorch_build_%j.out
#SBATCH --error=logs/pytorch_build_%j.err

# PyTorch Source Build Script for NVIDIA B200 (SM_100)
# This builds PyTorch, Triton, and xFormers with full Blackwell architecture support

set -e

echo "=========================================="
echo "Building PyTorch with SM_100 Support"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module purge
module load cuda/12.8.1  # Use CUDA 12.4+
module load conda

echo "CUDA Version: $(nvcc --version | grep release)"
echo "Conda Version: $(conda --version)"

# Determine CUDA home
if [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
fi
echo "CUDA_HOME: $CUDA_HOME"

# Force correct NCCL + CUDA libs at runtime
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export USE_SYSTEM_NCCL=1
export NCCL_INCLUDE_DIR=$CUDA_HOME/include
export NCCL_LIB_DIR=$CUDA_HOME/lib64

# Create build environment
BUILD_ENV="pytorch_build_sm100"
echo ""
echo "Creating build environment: $BUILD_ENV"

if conda env list | grep -q "^${BUILD_ENV} "; then
    echo "Removing existing build environment..."
    conda env remove -n $BUILD_ENV -y
fi

conda create -n $BUILD_ENV python=3.11 cmake ninja mkl mkl-include -y
eval "$(conda shell.bash hook)"
conda activate $BUILD_ENV

# Install build dependencies
echo ""
echo "Installing build dependencies..."
conda install -y numpy pyyaml setuptools cffi typing_extensions future six requests dataclasses

# Set build directory
BUILD_DIR="${HOME}/pytorch_sm100_build"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# ============================================
# Step 1: Build PyTorch from Source
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Building PyTorch"
echo "=========================================="

if [ ! -d "pytorch" ]; then
    echo "Cloning PyTorch repository..."
    git clone --recursive --depth 1 --branch v2.5.1 https://github.com/pytorch/pytorch
fi

cd pytorch

# Update submodules
echo "Updating submodules..."
git submodule sync
git submodule update --init --recursive

# Fix protobuf CMake compatibility issue
echo "Patching protobuf CMake version requirement..."
if [ -f "third_party/protobuf/cmake/CMakeLists.txt" ]; then
    sed -i 's/cmake_minimum_required(VERSION [0-9.]*)/cmake_minimum_required(VERSION 3.10)/' third_party/protobuf/cmake/CMakeLists.txt || true
    echo "✓ Protobuf CMake patched"
fi

# Set PyTorch build flags for SM_100 (Blackwell B200)
export TORCH_CUDA_ARCH_LIST="10.0"  # sm_100 for B200
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
export USE_CUDA=1
export USE_CUDNN=1
export USE_MKLDNN=1
export BUILD_TEST=0
export BUILD_CAFFE2=0
export USE_DISTRIBUTED=1
export USE_NCCL=1
export MAX_JOBS=16

echo "Build configuration:"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

# Clean previous build
echo ""
echo "Cleaning previous build artifacts..."
python setup.py clean

# Build and install PyTorch
echo ""
echo "Building PyTorch (this will take 1-2 hours)..."
python setup.py develop

echo ""
echo "✓ PyTorch build complete"

# ============================================
# Step 2: Build Triton with SM_100
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Building Triton"
echo "=========================================="

cd $BUILD_DIR

if [ ! -d "triton" ]; then
    echo "Cloning Triton repository..."
    git clone https://github.com/openai/triton.git
fi

cd triton/python

# Set Triton build flags for SM_100
export TRITON_CODEGEN_INTEL_XPU_BACKEND=0
export CMAKE_BUILD_TYPE=Release

# Build with SM_100 support
echo "Building Triton with SM_100 support..."
pip install -v -e . --no-build-isolation

echo ""
echo "✓ Triton build complete"

# ============================================
# Step 3: Build xFormers with SM_100
# ============================================
echo ""
echo "=========================================="
echo "Step 3: Building xFormers"
echo "=========================================="

cd $BUILD_DIR

if [ ! -d "xformers" ]; then
    echo "Cloning xFormers repository..."
    git clone --recursive https://github.com/facebookresearch/xformers.git
fi

cd xformers

# Update submodules
git submodule update --init --recursive

# Set xFormers build flags for SM_100
export TORCH_CUDA_ARCH_LIST="10.0"
export XFORMERS_BUILD_TYPE="Release"
export XFORMERS_ENABLE_DEBUG_ASSERTIONS=0

echo "Building xFormers with SM_100 support..."
pip install -v -e . --no-build-isolation

echo ""
echo "✓ xFormers build complete"

# ============================================
# Step 4: Verification
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Verifying Installation"
echo "=========================================="

echo "Testing PyTorch installation..."
python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
print("✓ Verification complete")
EOF

# ============================================
# Step 5: Export to Motion Control Environment
# ============================================
echo ""
echo "=========================================="
echo "Step 5: Installing to motion_control env"
echo "=========================================="

# Check if motion_control environment exists
if conda env list | grep -q "^motion_control "; then
    echo "Installing built packages to motion_control environment..."
    conda activate motion_control

    cd $BUILD_DIR/pytorch
    pip install -e .

    cd $BUILD_DIR/triton/python
    pip install -e .

    cd $BUILD_DIR/xformers
    pip install -e .

    echo "✓ Packages installed to motion_control environment"
else
    echo "! motion_control environment not found - run scripts/setup.sh first"
    echo "  Note: Packages installed in $BUILD_ENV environment"
fi

# ============================================
# Completion
# ============================================
echo ""
echo "=========================================="
echo "Build Complete!"
echo "End Time: $(date)"
echo "=========================================="
echo ""
echo "Build artifacts location: $BUILD_DIR"
echo ""
echo "To use the built PyTorch:"
echo "  conda activate motion_control"
echo "  python -c 'import torch; print(torch.__version__)'"
echo ""
echo "To verify SM_100 support on a GPU node:"
echo "  srun --gres=gpu:b200:1 python -c 'import torch; print(torch.cuda.get_device_capability())'"
echo "  # Should output: (10, 0)"
