# Contributing to HSDVC

Thank you for your interest in contributing to HSDVC! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title**: Describe the bug in one sentence
2. **Environment**: OS, Python version, CUDA version, GPU
3. **Steps to reproduce**: Minimal code to reproduce the bug
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens
6. **Logs/errors**: Full error messages and stack traces

### Suggesting Features

Feature requests are welcome! Please create an issue with:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches you considered
4. **Additional context**: Examples, mockups, etc.

### Pull Requests

We love pull requests! Here's the process:

1. **Fork and clone** the repository
2. **Create a branch** for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow code style guidelines
   - Add tests for new features
   - Update documentation
4. **Test your changes**
   ```bash
   pytest tests/
   ```
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 50GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/smcdr.git
cd smcdr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hsdvc --cov-report=html

# Run specific test file
pytest tests/test_motion.py

# Run specific test
pytest tests/test_motion.py::test_pose_estimation
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black hsdvc/
isort hsdvc/

# Check linting
flake8 hsdvc/

# Type check
mypy hsdvc/
```

These are automatically run by pre-commit hooks.

## Code Guidelines

### Python Style

Follow PEP 8 and these guidelines:

```python
# Good: Clear function with docstring
def extract_motion(video: torch.Tensor) -> MotionData:
    """
    Extract motion from video.
    
    Args:
        video: [T, C, H, W] video tensor
        
    Returns:
        Extracted motion data
    """
    # Implementation
    pass


# Good: Type hints and clear names
def compute_similarity(
    identity1: IdentityEmbedding,
    identity2: IdentityEmbedding
) -> float:
    """Compute similarity between identities."""
    return cosine_similarity(identity1.concat(), identity2.concat())


# Bad: No docstring, unclear names
def process(x, y):
    return x @ y.T
```

### Documentation

- **Docstrings**: All public functions/classes must have docstrings
- **Type hints**: Use type hints for function signatures
- **Comments**: Explain *why*, not *what*
- **Examples**: Include usage examples in docstrings

```python
def replace_character(
    video_path: str,
    new_character: str,
    preserve_motion: bool = True
) -> torch.Tensor:
    """
    Replace character in video while preserving motion.
    
    This function performs character replacement by:
    1. Extracting motion from source video
    2. Encoding new character identity
    3. Generating video with new character and source motion
    
    Args:
        video_path: Path to input video
        new_character: Path to new character image
        preserve_motion: Whether to preserve exact motion (default: True)
        
    Returns:
        Generated video tensor [T, C, H, W]
        
    Example:
        >>> output = replace_character(
        ...     "dance.mp4",
        ...     "character.jpg",
        ...     preserve_motion=True
        ... )
        >>> save_video(output, "output.mp4")
    
    Note:
        Video must be compiled first using VideoCompiler.compile_video()
    """
    pass
```

### Testing

Write tests for all new features:

```python
import pytest
import torch
from hsdvc import MotionExtractor


def test_motion_extraction():
    """Test motion extraction from video."""
    # Setup
    extractor = MotionExtractor()
    video = torch.randn(1, 30, 3, 256, 256)
    
    # Execute
    motion = extractor(video)
    
    # Assert
    assert motion.poses_3d.shape == (30, 33, 3)
    assert motion.depth_maps.shape == (30, 256, 256)
    assert motion.num_frames == 30


def test_identity_encoding():
    """Test identity encoding and factorization."""
    from hsdvc import IdentityEncoder
    
    encoder = IdentityEncoder()
    image = torch.randn(1, 3, 224, 224)
    
    identity = encoder(image)
    
    # Check dimensions
    assert identity.shape.shape[-1] == 512
    assert identity.appearance.shape[-1] == 512
    assert identity.texture.shape[-1] == 1024
    
    # Check disentanglement
    similarity = torch.cosine_similarity(
        identity.shape,
        identity.appearance,
        dim=-1
    )
    assert similarity.abs() < 0.3  # Should be orthogonal
```

## Project Structure

```
smcdr/
├── hsdvc/                  # Main package
│   ├── models/            # Model implementations
│   │   ├── motion/        # Motion extraction
│   │   ├── identity/      # Identity encoding
│   │   ├── geometry/      # 3D representations
│   │   ├── cogvideox/     # CogVideoX integration
│   │   └── controlnet/    # Control conditioning
│   ├── data/              # Data loading
│   ├── training/          # Training utilities
│   └── utils/             # Helper functions
├── configs/               # Configuration files
├── scripts/               # Command-line scripts
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── notebooks/             # Jupyter notebooks
```

## Areas for Contribution

We especially welcome contributions in:

### High Priority

- [ ] **Performance optimization**: Speed up motion extraction
- [ ] **Memory efficiency**: Reduce VRAM usage
- [ ] **Multi-GPU training**: Improve distributed training
- [ ] **Better depth estimation**: Integrate newer models
- [ ] **Tests**: Increase test coverage

### Medium Priority

- [ ] **Web UI**: Gradio or Streamlit interface
- [ ] **Docker support**: Containerization
- [ ] **Pre-trained models**: Share trained weights
- [ ] **Tutorials**: More example notebooks
- [ ] **Benchmarks**: Comprehensive evaluation suite

### Research Directions

- [ ] **Zero-shot generalization**: Skip per-video compilation
- [ ] **Multi-subject scenes**: Handle multiple characters
- [ ] **Higher resolution**: 1080p, 4K support
- [ ] **Real-time inference**: Faster generation
- [ ] **New architectures**: Experiment with alternatives

## Questions?

- **Discord**: [Join our server](#) (TODO)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/smcdr/discussions)
- **Email**: your.email@example.com

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for contributing to HSDVC! 🎉
