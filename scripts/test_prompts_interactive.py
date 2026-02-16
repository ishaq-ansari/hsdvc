#!/usr/bin/env python
"""
Interactive test script for Wan2.1 I2V with different prompts.
Tests various prompt styles to understand model capabilities.
"""

import subprocess
import sys
from pathlib import Path

# Test configurations
TESTS = [
    {
        "name": "walking",
        "prompt": "person walking forward slowly, smooth natural motion",
        "guidance": 5.0,
        "duration": 2.0
    },
    {
        "name": "dancing",
        "prompt": "person dancing energetically, fluid body movements",
        "guidance": 7.0,
        "duration": 2.0
    },
    {
        "name": "waving",
        "prompt": "person waving hand, friendly gesture, smiling",
        "guidance": 6.0,
        "duration": 2.0
    },
    {
        "name": "portrait_subtle",
        "prompt": "portrait with subtle head movement, gentle breathing, natural expression",
        "guidance": 3.0,
        "duration": 2.0
    },
    {
        "name": "talking",
        "prompt": "person talking, mouth moving, natural facial expressions",
        "guidance": 5.0,
        "duration": 2.0
    },
    {
        "name": "turning_around",
        "prompt": "person turning around slowly, full body rotation",
        "guidance": 6.0,
        "duration": 2.0
    },
]

def run_test(test_config, character_image, output_dir):
    """Run a single test."""
    output_path = output_dir / f"test_{test_config['name']}.mp4"

    cmd = [
        "python", "scripts/generate.py",
        "--character", str(character_image),
        "--output", str(output_path),
        "--prompt", test_config['prompt'],
        "--duration", str(test_config['duration']),
        "--guidance-scale", str(test_config['guidance']),
        "--num-inference-steps", "40",
        "--target-resolution", "1920x1080"
    ]

    print(f"\n{'='*80}")
    print(f"Test: {test_config['name']}")
    print(f"Prompt: {test_config['prompt']}")
    print(f"Guidance: {test_config['guidance']}, Duration: {test_config['duration']}s")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, check=False)
    return result.returncode == 0

def main():
    character_image = Path("data/inputs/test_char.png")
    output_dir = Path("data/outputs/prompt_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not character_image.exists():
        print(f"Error: Character image not found: {character_image}")
        sys.exit(1)

    print("="*80)
    print("Wan2.1 I2V Prompt Testing")
    print("="*80)
    print(f"Character: {character_image}")
    print(f"Output directory: {output_dir}")
    print(f"Number of tests: {len(TESTS)}")
    print()

    # Ask user which tests to run
    print("Available tests:")
    for i, test in enumerate(TESTS, 1):
        print(f"  {i}. {test['name']}: \"{test['prompt']}\"")
    print(f"  {len(TESTS)+1}. ALL tests")

    choice = input(f"\nSelect test (1-{len(TESTS)+1}): ").strip()

    tests_to_run = []
    if choice == str(len(TESTS)+1):
        tests_to_run = TESTS
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(TESTS):
                tests_to_run = [TESTS[idx]]
            else:
                print("Invalid choice")
                sys.exit(1)
        except ValueError:
            print("Invalid choice")
            sys.exit(1)

    # Run selected tests
    results = []
    for test in tests_to_run:
        success = run_test(test, character_image, output_dir)
        results.append((test['name'], success))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")

    print("\nOutput videos saved to:", output_dir)
    print("="*80)

if __name__ == "__main__":
    main()
