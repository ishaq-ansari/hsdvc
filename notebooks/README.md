# HSDVC Quick Start Examples

This directory contains Jupyter notebooks demonstrating HSDVC capabilities.

## Notebooks

### 1. `01_quick_start.ipynb`
Basic usage: compile a video and replace character.

### 2. `02_motion_extraction.ipynb`
Detailed motion extraction and visualization.

### 3. `03_identity_encoding.ipynb`
Identity encoding and factorization exploration.

### 4. `04_character_interpolation.ipynb`
Interpolating between multiple characters.

### 5. `05_style_transfer.ipynb`
Character replacement with style transfer.

### 6. `06_training.ipynb`
Training pipeline walkthrough.

## Running Notebooks

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter server
cd notebooks
jupyter notebook
```

## Requirements

All notebooks require the HSDVC package installed:

```bash
cd ..
pip install -e .
```

Some notebooks may require additional data:
- Sample videos (place in `data/videos/`)
- Character images (place in `data/characters/`)

Download sample data:
```bash
# TODO: Add download script
python scripts/download_sample_data.py
```
