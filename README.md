# Motion Control Video Generation System

A Kling-style motion control system using Wan2.1 I2V for generating high-quality videos (1080p+, 10-30s) where a character image performs motions from a reference video.

## Features

- **Wan2.1 I2V Model**: State-of-the-art image-to-video generation
- **High Quality**: 1080p+ output resolution with temporal smoothing
- **Long Videos**: Generate 10-30 second clips through segment stitching
- **Audio Preservation**: Extract and merge audio from reference videos
- **Identity Preservation**: LoRA + CodeFormer face restoration
- **Batch Processing**: Process multiple videos with SLURM array jobs
- **B200 Optimized**: Built with SM_100 support for NVIDIA Blackwell GPUs

## System Requirements

### Hardware

- NVIDIA B200 GPU (or other CUDA-capable GPU)
- 64GB+ RAM
- 100GB+ storage

### Software

- CUDA 12.4+ (12.8.1 or 13.0.2 recommended)
- Python 3.11
- SLURM (for HPC batch processing)

## Installation

### 1. Quick Start

Clone the repository and run the setup script:

```bash
cd /blue/pinaki.sarder/iansari/motion_control
bash scripts/setup.sh
```

This will:

- Create the conda environment
- Install dependencies
- Clone Wan2.1 and CodeFormer repositories
- Set up directory structure

### 2. Build PyTorch with SM_100 Support (for B200 GPUs)

```bash
# Submit build job (takes ~3 hours)
sbatch environment/build_pytorch_sm100.sh

# Or build locally if not using SLURM
bash environment/build_pytorch_sm100.sh
```

### 3. Download Wan2.1 Model

Download the I2V-14B-720P checkpoint from Hugging Face:

```bash
# Visit: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
# Download to: models/checkpoints/

# Or use huggingface-cli
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir models/checkpoints
```

### 4. Activate Environment

```bash
conda activate motion_control
```

## Usage

### Single Video Generation

```bash
python scripts/generate.py \
    --character data/inputs/character.png \
    --reference data/inputs/reference.mp4 \
    --output data/outputs/result.mp4 \
    --duration 10 \
    --preserve-identity \
    --preserve-audio \
    --target-resolution 1920x1080
```

### Using SLURM

```bash
# Edit submit_job.sh to specify your inputs
# Set environment variables
export CHARACTER_IMAGE="data/inputs/character.png"
export REFERENCE_VIDEO="data/inputs/reference.mp4"
export OUTPUT_VIDEO="data/outputs/result.mp4"
export DURATION="30"

# Submit job
sbatch scripts/submit_job.sh

# Check status
squeue -u $USER

# View output
tail -f logs/motion_*.out
```

### Batch Processing

Create a manifest file (`data/batch_manifest.json`):

```json
{
  "jobs": [
    {
      "id": "job_001",
      "character_image": "data/inputs/character1.png",
      "reference_video": "data/inputs/dance1.mp4",
      "output_video": "data/outputs/result1.mp4",
      "duration": 10,
      "preserve_identity": true,
      "preserve_audio": true
    },
    {
      "id": "job_002",
      "character_image": "data/inputs/character2.png",
      "reference_video": "data/inputs/dance2.mp4",
      "output_video": "data/outputs/result2.mp4",
      "duration": 15,
      "preserve_identity": true,
      "preserve_audio": true
    }
  ]
}
```

Process batch:

```bash
# Sequential (local)
python scripts/generate.py --batch data/batch_manifest.json --model-path models/checkpoints

# Parallel (SLURM array jobs)
python -c "from src.batch_processor import BatchProcessor; BatchProcessor('data/batch_manifest.json').process_with_slurm('models/checkpoints')"
```

## CLI Options

```
usage: generate.py [-h] [--character CHARACTER] [--reference REFERENCE]
                   [--output OUTPUT] [--batch BATCH] [--job-index JOB_INDEX]
                   [--model-path MODEL_PATH] [--duration DURATION] [--fps FPS]
                   [--guidance-scale GUIDANCE_SCALE]
                   [--num-inference-steps NUM_INFERENCE_STEPS] [--seed SEED]
                   [--preserve-identity] [--preserve-audio]
                   [--target-resolution TARGET_RESOLUTION]
                   [--upscale-method {realesrgan,bicubic,lanczos}]
                   [--temporal-smoothing] [--device DEVICE]
                   [--dtype {float16,float32}]

Options:
  --character           Path to character image
  --reference           Path to reference video
  --output              Path for output video
  --duration            Video duration in seconds (default: 5.0)
  --fps                 Frames per second (default: 24)
  --preserve-identity   Apply CodeFormer face restoration
  --preserve-audio      Extract and merge audio from reference
  --target-resolution   Target resolution WIDTHxHEIGHT (default: 1920x1080)
  --upscale-method      Upscaling method (default: lanczos)
  --temporal-smoothing  Apply temporal smoothing
  --device              Device for inference (default: cuda)
  --dtype               Data type for inference (default: float16)
```

## Project Structure

```
motion_control/
├── environment/          # Environment setup
│   ├── build_pytorch_sm100.sh
│   ├── environment.yml
│   └── requirements.txt
├── models/               # Model repositories and checkpoints
│   ├── wan21/           # Wan2.1 repository (cloned)
│   ├── CodeFormer/      # CodeFormer repository (cloned)
│   └── checkpoints/     # Model weights
├── src/                  # Source code
│   ├── pipeline/        # Core pipeline
│   │   ├── video_generator.py
│   │   ├── preprocessing.py
│   │   ├── postprocessing.py
│   │   └── identity_preserving.py
│   ├── utils/           # Utilities
│   │   ├── audio_handler.py
│   │   ├── video_utils.py
│   │   └── slurm_utils.py
│   └── batch_processor.py
├── scripts/              # Scripts
│   ├── setup.sh
│   ├── generate.py
│   └── submit_job.sh
├── configs/              # Configuration
│   └── generation_config.yaml
├── data/                 # Data
│   ├── inputs/
│   └── outputs/
└── logs/                 # Job logs
```

## Performance

### Expected Performance on B200

- **5s video (720p → 1080p)**: ~30-60 seconds
- **30s video (720p → 1080p)**: ~3-5 minutes
- **VRAM usage**: 16-32GB (depends on settings)

### Optimization Tips

1. **Use FP16**: `--dtype float16` (default, 2x faster)
2. **Reduce steps**: `--num-inference-steps 30` (faster, slightly lower quality)
3. **Skip temporal smoothing**: Remove `--temporal-smoothing` (faster)
4. **Use lanczos upscaling**: `--upscale-method lanczos` (faster than realesrgan)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce resolution
--target-resolution 1280x720

# Use float32 (if FP16 causes issues)
--dtype float32

# Process shorter segments
--duration 5
```

### Model Not Loading

```bash
# Verify model path
ls -lh models/checkpoints/

# Check for required files
# Should contain: model weights, config files, etc.
```

### Audio Sync Issues

```bash
# Check original video
ffprobe data/inputs/reference.mp4

# Manually adjust audio
ffmpeg -i output.mp4 -itsoffset 0.1 -i audio.m4a -c copy output_synced.mp4
```

### Face Restoration Not Working

```bash
# Verify CodeFormer installation
cd models/CodeFormer
python inference_codeformer.py --help

# Download pretrained models
python scripts/download_pretrained_models.py facelib
python scripts/download_pretrained_models.py CodeFormer
```

## Verification

Test the installation:

```bash
# On compute node with GPU
srun --gres=gpu:b200:1 --pty bash

# Activate environment
conda activate motion_control

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Capability: {torch.cuda.get_device_capability(0)}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA B200
# Capability: (9, 0)

# Test generation (use small test images/videos)
python scripts/generate.py \
    --character data/inputs/test_char.png \
    --reference data/inputs/test_ref.mp4 \
    --output data/outputs/test.mp4 \
    --duration 2
```

## Advanced Usage

### Custom Configuration

Edit `configs/generation_config.yaml` to set default parameters:

```yaml
generation:
  duration: 10.0
  fps: 30
  guidance_scale: 8.0

postprocessing:
  target_resolution:
    width: 2560
    height: 1440
  upscale_method: "realesrgan"
```

### Identity Preservation Tuning

```python
from src.pipeline.identity_preserving import IdentityPreserver

preserver = IdentityPreserver(
    method="codeformer",
    face_enhance_strength=0.7  # Increase for stronger restoration
)
```

### LoRA Fine-Tuning (Advanced)

```python
# Train character-specific LoRA
from src.pipeline.identity_preserving import LoRATrainer

trainer = LoRATrainer(base_model=generator.pipeline, rank=16)
lora_weights = trainer.train_lora(
    character_images=["char1.png", "char2.png", "char3.png"],
    num_steps=1000
)
```

## Contributing

This is a research/development project. Contributions welcome!

## License

- Code: MIT License
- Wan2.1: Apache 2.0
- CodeFormer: S-Lab License
- Real-ESRGAN: BSD-3

## Acknowledgments

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Image-to-Video model
- [CodeFormer](https://github.com/sczhou/CodeFormer) - Face restoration
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Video upscaling

## Citation

If you use this system in your research, please cite:

```bibtex
@software{motion_control_2026,
  title = {Motion Control Video Generation System},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/motion_control}
}
```

## Support

For issues or questions:

- Check the troubleshooting section above
- Review logs in `logs/` directory
- Consult Wan2.1 documentation
- Check CUDA/PyTorch compatibility

## Roadmap

- [ ] Gradio web interface
- [ ] ControlNet fallback for precise motion control
- [ ] Multi-GPU distributed generation
- [ ] Fine-tuning scripts for specific styles
- [ ] Stable Video Diffusion alternative backend
- [ ] Frame interpolation (RIFE integration)
- [ ] Real-time preview mode
