#!/bin/bash
#SBATCH --job-name=wan21_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=b200:1

# Load environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.conda/envs/motion_control/etc/profile.d/conda.sh 2>/dev/null || true
conda activate motion_control

# Set paths
cd /blue/pinaki.sarder/iansari/motion_control

# Test generation with minimal settings
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --output data/outputs/test.mp4 \
    --duration 3.375 \
    --fps 24 \
    --guidance-scale 5.0 \
    --num-inference-steps 40

echo "Test completed. Check output at data/outputs/test.mp4"
