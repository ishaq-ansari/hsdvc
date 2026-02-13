# Getting Started with HSDVC

This guide will help you get up and running with HSDVC in minutes.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Training](#training)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

Before installing HSDVC, ensure you have:

- **Python**: 3.10 or higher
- **CUDA**: 11.8+ (for GPU support)
- **GPU**: 12GB+ VRAM (24GB recommended)
- **Disk Space**: 50GB+ free

Check your setup:

```bash
python --version  # Should be 3.10+
nvidia-smi        # Should show CUDA 11.8+
```

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/smcdr.git
cd smcdr
```

### Step 2: Create Environment

Using conda (recommended):

```bash
conda create -n hsdvc python=3.10
conda activate hsdvc
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install HSDVC
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"
```

### Step 4: Verify Installation

```bash
python -c "import hsdvc; print(f'HSDVC version: {hsdvc.__version__}')"
```

Expected output: `HSDVC version: 0.1.0`

## Quick Start

### 5-Minute Demo

Let's replace a character in a video!

#### 1. Prepare Data

Create the data directory structure:

```bash
mkdir -p data/videos data/characters
```

Download sample data:

```bash
# TODO: Add actual download script
# For now, place your own:
# - Video: data/videos/sample.mp4
# - Character image: data/characters/new_character.jpg
```

#### 2. Run Demo

```bash
python scripts/demo.py
```

This will:
1. Extract motion from the video
2. Compile a video-specific model
3. Replace the character
4. Save output to `outputs/demo/output.mp4`

Expected time: ~10 minutes (on RTX 4090)

### Step-by-Step Example

#### Step 1: Extract Motion

```python
from hsdvc import MotionExtractor

# Initialize extractor
extractor = MotionExtractor()

# Extract motion from video
motion_data = extractor.extract("data/videos/sample.mp4")

print(f"Extracted {motion_data.num_frames} frames")
print(f"Pose shape: {motion_data.poses_3d.shape}")
```

#### Step 2: Compile Video

```python
from hsdvc import VideoCompiler

# Load compiler with base model
compiler = VideoCompiler.from_pretrained("cogvideox-5b")

# Compile video (adapts model to this specific video)
results = compiler.compile_video(
    video_path="data/videos/sample.mp4",
    num_steps=500,  # More steps = better quality
    save_dir="outputs/compiled"
)
```

#### Step 3: Replace Character

```python
from hsdvc import CharacterReplacer

# Initialize replacer with compiled video
replacer = CharacterReplacer(compiler)

# Replace character
output = replacer.replace(
    new_character_image="data/characters/new_character.jpg",
    preserve_motion=True,
    output_path="outputs/output.mp4"
)

print("Done! Output saved to outputs/output.mp4")
```

## Basic Usage

### Command-Line Interface

#### Compile a Video

```bash
python scripts/compile_video.py \
    --video data/videos/sample.mp4 \
    --output_dir outputs/compiled \
    --num_steps 500
```

#### Replace Character

```bash
python scripts/replace_character.py \
    --compiled_dir outputs/compiled \
    --new_character data/characters/new_character.jpg \
    --output outputs/output.mp4
```

### Python API

#### Full Example

```python
import torch
from hsdvc import VideoCompiler, CharacterReplacer

# Setup
device = torch.device("cuda")

# Load compiler
compiler = VideoCompiler.from_pretrained("cogvideox-5b")
compiler = compiler.to(device)

# Compile video
compiler.compile_video(
    video_path="input.mp4",
    num_steps=500,
    save_dir="compiled_data"
)

# Replace character
replacer = CharacterReplacer(compiler)
output = replacer.replace(
    new_character_image="new_character.jpg",
    preserve_motion=True,
    identity_strength=1.0,
    num_inference_steps=50,
    output_path="output.mp4"
)
```

#### Working with Motion Data

```python
from hsdvc.models.motion import MotionExtractor

extractor = MotionExtractor()
motion = extractor.extract("video.mp4")

# Access motion components
poses = motion.poses_3d        # [T, N, 3] 3D keypoints
depth = motion.depth_maps      # [T, H, W] depth maps
flow = motion.optical_flow     # [T-1, H, W, 2] optical flow
camera = motion.camera_trajectory  # [T, 6] camera path

# Save motion data
torch.save(motion, "motion.pt")

# Load motion data
motion = torch.load("motion.pt")
```

#### Working with Identity

```python
from hsdvc.models.identity import IdentityEncoder

encoder = IdentityEncoder()

# Encode identity from image
identity = encoder.encode_from_path("character.jpg")

# Access factorized components
shape = identity.shape           # [512] shape embedding
appearance = identity.appearance # [512] appearance embedding
texture = identity.texture       # [1024] texture embedding

# Interpolate between identities
identity1 = encoder.encode_from_path("char1.jpg")
identity2 = encoder.encode_from_path("char2.jpg")

interpolated = encoder.interpolate(
    identity1, identity2,
    alpha=0.5  # 50% blend
)

# Modify specific attributes
custom_identity = IdentityEmbedding(
    shape=identity1.shape,         # Use shape from char1
    appearance=identity2.appearance,  # Use appearance from char2
    texture=identity1.texture      # Use texture from char1
)
```

## Advanced Features

### Character Interpolation

Smoothly transition between multiple characters:

```python
replacer = CharacterReplacer(compiler)

video = replacer.interpolate_characters(
    character_images=[
        "char1.jpg",
        "char2.jpg",
        "char3.jpg"
    ],
    num_frames_per_transition=30,
    output_path="interpolated.mp4"
)
```

### Style Transfer

Replace character with style transfer:

```python
video = replacer.replace_with_style_transfer(
    new_character_image="character.jpg",
    style_image="style.jpg",
    style_strength=0.7,
    output_path="stylized.mp4"
)
```

### Custom Control Signals

Fine-grained control over generation:

```python
# Extract motion
motion = extractor.extract("video.mp4")

# Modify control signals
motion.poses_3d[:, 0, :] *= 1.5  # Scale head position
motion.depth_maps *= 0.8          # Adjust depth

# Generate with modified motion
output = compiler.generate(
    motion_data=motion,
    identity_embedding=identity,
    num_inference_steps=50
)
```

### Batch Processing

Process multiple videos:

```python
from pathlib import Path

video_dir = Path("data/videos")
output_dir = Path("outputs")

for video_path in video_dir.glob("*.mp4"):
    print(f"Processing {video_path.name}...")
    
    # Compile
    compiler.compile_video(
        video_path=str(video_path),
        save_dir=str(output_dir / video_path.stem)
    )
    
    # Replace character
    replacer = CharacterReplacer(compiler)
    replacer.replace(
        new_character_image="character.jpg",
        output_path=str(output_dir / f"{video_path.stem}_output.mp4")
    )
```

## Training

### Training Your Own Model

#### Stage 1: Pre-train Structure Modules

Train motion extraction and identity encoding:

```bash
python scripts/train_stage1.py \
    --data_dir data/training \
    --output_dir outputs/stage1 \
    --num_epochs 100 \
    --batch_size 4 \
    --use_wandb
```

#### Stage 2: Train CogVideoX Adapter

Train the full model with conditioning:

```bash
python scripts/train_stage2.py \
    --data_dir data/training \
    --stage1_checkpoint outputs/stage1/best_model.pt \
    --output_dir outputs/stage2 \
    --num_epochs 50 \
    --batch_size 2
```

### Training on Custom Data

Organize your data:

```
data/training/
├── video1/
│   ├── video.mp4
│   └── metadata.json
├── video2/
│   ├── video.mp4
│   └── metadata.json
...
```

Metadata format:

```json
{
  "identity": "person_id",
  "tags": ["dance", "indoor"],
  "resolution": [1080, 1920],
  "fps": 30
}
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch_size 1`
- Reduce video length: `--video_length 25`
- Use gradient checkpointing
- Use mixed precision: `--mixed_precision fp16`

#### 2. MediaPipe Installation Issues

**Error**: `ImportError: No module named 'mediapipe'`

**Solution**:
```bash
pip install mediapipe
```

#### 3. Slow Motion Extraction

**Issue**: Motion extraction takes too long

**Solutions**:
- Use GPU for depth/flow estimation
- Reduce resolution: `--resolution 384 512`
- Use faster models: `--depth_model midas`

#### 4. Poor Quality Output

**Issue**: Generated video quality is poor

**Solutions**:
- Increase compilation steps: `--num_steps 1000`
- Increase inference steps: `--num_inference_steps 100`
- Use higher resolution
- Ensure good quality input video

### Getting Help

- **Documentation**: See [docs/](docs/) folder
- **Issues**: [GitHub Issues](https://github.com/yourusername/smcdr/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/smcdr/discussions)
- **Discord**: [Join our server](#) (TODO)

### Performance Tips

1. **Use fp16**: Enables mixed precision for 2x speedup
   ```python
   compiler = compiler.half()  # Convert to fp16
   ```

2. **Optimize motion extraction**: Cache motion data
   ```python
   motion = extractor.extract("video.mp4")
   torch.save(motion, "motion_cache.pt")
   # Later: motion = torch.load("motion_cache.pt")
   ```

3. **Batch inference**: Process multiple frames together
   ```python
   compiler.generate(num_frames=49)  # Better than 49x1
   ```

4. **Use compiled models**: PyTorch 2.0 compilation
   ```python
   compiler = torch.compile(compiler)
   ```

## Next Steps

Now that you're set up, explore:

1. **Tutorials**: See [notebooks/](notebooks/) for interactive guides
2. **Examples**: Check [examples/](examples/) for more use cases
3. **Architecture**: Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical details
4. **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

Happy coding! 🚀
