#!/bin/bash
#SBATCH --job-name=wan21_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=b200:1

# Test different prompts with Wan2.1 I2V
cd /blue/pinaki.sarder/iansari/motion_control

# Test 1: Simple motion
echo "=== Test 1: Simple walking motion ==="
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --output data/outputs/test_walk.mp4 \
    --prompt "person walking forward slowly, smooth natural motion" \
    --duration 2.0 \
    --guidance-scale 5.0 \
    --num-inference-steps 40

# Test 2: Dynamic motion
echo "=== Test 2: Dancing motion ==="
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --output data/outputs/test_dance.mp4 \
    --prompt "person dancing energetically, fluid body movements" \
    --duration 2.0 \
    --guidance-scale 7.0 \
    --num-inference-steps 40

# Test 3: Subtle motion
echo "=== Test 3: Portrait subtle motion ==="
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --output data/outputs/test_subtle.mp4 \
    --prompt "portrait with subtle head movement, gentle breathing" \
    --duration 2.0 \
    --guidance-scale 3.0 \
    --num-inference-steps 40

# Test 4: Different guidance scales
echo "=== Test 4: High guidance ==="
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --output data/outputs/test_high_guidance.mp4 \
    --prompt "smooth natural motion" \
    --duration 2.0 \
    --guidance-scale 10.0 \
    --num-inference-steps 40

echo "=== All tests complete! ==="
ls -lh data/outputs/test_*.mp4
