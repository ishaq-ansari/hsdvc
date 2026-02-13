#!/usr/bin/env python3
"""
Installation verification script.
Tests that all components are properly installed and working.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_imports():
    """Check required imports."""
    print("\nChecking required packages...")
    
    required = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "opencv-python": "OpenCV",
        "numpy": "NumPy",
        "pillow": "Pillow",
        "tqdm": "tqdm",
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            # Handle opencv-python -> cv2
            import_name = "cv2" if module == "opencv-python" else module
            __import__(import_name)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (install with: pip install {module})")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available ({device_count} device(s))")
            print(f"     Device: {device_name}")
            return True
        else:
            print("  ⚠️  CUDA not available (CPU only)")
            return False
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        return False


def check_hsdvc():
    """Check HSDVC installation."""
    print("\nChecking HSDVC installation...")
    try:
        import hsdvc
        print(f"  ✅ HSDVC version {hsdvc.__version__}")
        
        # Check main components
        from hsdvc import VideoCompiler, CharacterReplacer, MotionExtractor, IdentityEncoder
        print("  ✅ All main components importable")
        return True
    except Exception as e:
        print(f"  ❌ HSDVC import error: {e}")
        return False


def check_structure():
    """Check project structure."""
    print("\nChecking project structure...")
    
    required_paths = [
        "hsdvc",
        "hsdvc/models",
        "hsdvc/data",
        "hsdvc/utils",
        "scripts",
        "configs",
        "docs",
        "README.md",
        "setup.py",
        "requirements.txt",
    ]
    
    all_ok = True
    for path in required_paths:
        if Path(path).exists():
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path} (missing)")
            all_ok = False
    
    return all_ok


def run_basic_test():
    """Run a basic functionality test."""
    print("\nRunning basic functionality test...")
    try:
        import torch
        from hsdvc.config import HSDVCConfig
        from hsdvc.models.motion import MotionExtractor
        from hsdvc.models.identity import IdentityEncoder
        
        # Create config
        config = HSDVCConfig()
        print("  ✅ Config initialization")
        
        # Create motion extractor
        motion_extractor = MotionExtractor(config.motion)
        print("  ✅ Motion extractor initialization")
        
        # Create identity encoder
        identity_encoder = IdentityEncoder(config.identity)
        print("  ✅ Identity encoder initialization")
        
        # Test forward pass with dummy data
        dummy_video = torch.randn(1, 10, 3, 224, 224)
        motion_data = motion_extractor(dummy_video)
        print("  ✅ Motion extraction forward pass")
        
        dummy_image = torch.randn(1, 3, 224, 224)
        identity = identity_encoder(dummy_image)
        print("  ✅ Identity encoding forward pass")
        
        print("\n  ✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("HSDVC Installation Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_imports),
        ("CUDA", check_cuda),
        ("HSDVC", check_hsdvc),
        ("Project Structure", check_structure),
        ("Basic Functionality", run_basic_test),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed! HSDVC is ready to use.")
        print("\nNext steps:")
        print("  1. Run demo: python scripts/demo.py")
        print("  2. Check tutorials: see notebooks/")
        print("  3. Read docs: docs/GETTING_STARTED.md")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nFor help:")
        print("  - See docs/GETTING_STARTED.md")
        print("  - Open an issue: GitHub Issues")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
