from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hsdvc",
    version="0.1.0",
    author="Ishaq Ansari",
    author_email="ishaq.ansari@medicine.ufl.edu",
    description="Hybrid Structured-Diffusion Video Compiler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ishaq-ansari/smcdr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "accelerate>=0.25.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "scipy>=1.11.0",
        "scikit-image>=0.22.0",
        "pillow>=10.0.0",
        "tqdm>=4.66.0",
        "tensorboard>=2.15.0",
        "wandb>=0.16.0",
        "omegaconf>=2.3.0",
        "hydra-core>=1.3.0",
        "safetensors>=0.4.0",
        "peft>=0.7.0",  # For LoRA
        "xformers>=0.0.22",  # For efficient attention
        # Motion and 3D
        "trimesh>=4.0.0",
        # Video processing
        "imageio[ffmpeg]>=2.31.0",
        # Pose estimation
        "mediapipe>=0.10.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "3d": [
            "open3d>=0.18.0",
            # pytorch3d needs manual install: pip install "git+https://github.com/facebookresearch/pytorch3d.git"
        ],
        "video": [
            "av>=11.0.0",
            "decord>=0.6.0",
        ],
        "motion": [
            # Depth-Anything V2: pip install git+https://github.com/DepthAnything/Depth-Anything-V2.git
            # RAFT: pip install git+https://github.com/princeton-vl/RAFT.git
        ],
        "gaussian_splatting": [
            "diff-gaussian-rasterization>=0.0.1",
        ],
        "nerf": [
            "nerfstudio>=0.3.0",
        ],
    },
)
