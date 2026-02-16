#!/bin/bash
#SBATCH --job-name=motion_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/motion_%j.out
#SBATCH --error=logs/motion_%j.err

# Motion Control Video Generation - SLURM Job Template
# Customize this script for your specific needs

echo "=========================================="
echo "Motion Control Video Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load modules
module purge
module load cuda/12.8.1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate motion_control

# Verify GPU
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Configuration
CHARACTER_IMAGE="${CHARACTER_IMAGE:-data/inputs/character.png}"
REFERENCE_VIDEO="${REFERENCE_VIDEO:-data/inputs/reference.mp4}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-data/outputs/result.mp4}"
DURATION="${DURATION:-10}"
MODEL_PATH="${MODEL_PATH:-models/checkpoints}"

echo ""
echo "Configuration:"
echo "  Character: $CHARACTER_IMAGE"
echo "  Reference: $REFERENCE_VIDEO"
echo "  Output: $OUTPUT_VIDEO"
echo "  Duration: ${DURATION}s"
echo "  Model: $MODEL_PATH"
echo ""

# Run generation
python scripts/generate.py \
    --character "$CHARACTER_IMAGE" \
    --reference "$REFERENCE_VIDEO" \
    --output "$OUTPUT_VIDEO" \
    --model-path "$MODEL_PATH" \
    --duration $DURATION \
    --fps 24 \
    --preserve-identity \
    --preserve-audio \
    --target-resolution 1920x1080 \
    --upscale-method lanczos \
    --temporal-smoothing \
    --device cuda \
    --dtype float16

# Check exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
