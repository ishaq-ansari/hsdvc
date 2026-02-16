"""
Batch processing utilities for motion control pipeline.
Handles multiple character-video pairs with SLURM integration.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.slurm_utils import SLURMJobSubmitter, create_gpu_job_script, is_slurm_available


class BatchProcessor:
    """Process multiple video generation jobs in batch."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize batch processor.

        Args:
            manifest_path: Path to JSON manifest file with job specifications
            output_dir: Output directory for results. If None, uses paths from manifest.
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir) if output_dir else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.jobs = self._load_manifest()

    def _load_manifest(self) -> List[Dict]:
        """
        Load job manifest from JSON file.

        Returns:
            List of job specifications
        """
        with open(self.manifest_path, 'r') as f:
            data = json.load(f)

        if 'jobs' not in data:
            raise ValueError("Manifest must contain 'jobs' key")

        jobs = data['jobs']
        print(f"Loaded {len(jobs)} jobs from manifest")

        # Validate jobs
        for i, job in enumerate(jobs):
            required_keys = ['character_image', 'reference_video', 'output_video']
            missing = [k for k in required_keys if k not in job]
            if missing:
                raise ValueError(f"Job {i} missing required keys: {missing}")

            # Apply output_dir if specified
            if self.output_dir:
                job['output_video'] = str(
                    self.output_dir / Path(job['output_video']).name
                )

        return jobs

    def process_sequential(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        **generation_kwargs
    ):
        """
        Process all jobs sequentially on current machine.

        Args:
            model_path: Path to Wan2.1 model
            device: Device for processing
            **generation_kwargs: Additional generation parameters
        """
        from pipeline.video_generator import Wan21VideoGenerator
        from pipeline.postprocessing import postprocess_video
        from pipeline.identity_preserving import preserve_identity
        from utils.audio_handler import AudioHandler

        print(f"Processing {len(self.jobs)} jobs sequentially...")

        # Initialize generator once
        generator = Wan21VideoGenerator(model_path=model_path, device=device)
        audio_handler = AudioHandler()

        for i, job in enumerate(self.jobs):
            print(f"\n{'='*60}")
            print(f"Job {i+1}/{len(self.jobs)}: {job.get('id', f'job_{i}')}")
            print(f"{'='*60}")

            try:
                character_image = Path(job['character_image'])
                reference_video = Path(job['reference_video'])
                output_video = Path(job['output_video'])
                duration = job.get('duration', 5)

                # Generate video
                print(f"Generating video ({duration}s)...")
                if duration <= 5:
                    frames = generator.generate_video(
                        character_image=character_image,
                        num_frames=int(duration * 24),
                        fps=24
                    )
                else:
                    frames = generator.generate_long_video(
                        character_image=character_image,
                        target_duration=duration,
                        fps=24
                    )

                # Identity preservation
                if job.get('preserve_identity', False):
                    print("Applying identity preservation...")
                    frames = preserve_identity(frames, character_image, device=device)

                # Postprocessing
                target_res = job.get('target_resolution', (1920, 1080))
                print(f"Postprocessing (target: {target_res})...")
                frames = postprocess_video(
                    frames,
                    target_resolution=target_res,
                    device=device
                )

                # Save video (without audio first)
                from utils.video_utils import save_frames_as_video

                temp_video = output_video.parent / f"{output_video.stem}_temp.mp4"
                save_frames_as_video(frames, temp_video, fps=24)

                # Add audio if requested
                if job.get('preserve_audio', False):
                    print("Extracting and merging audio...")
                    audio_path = audio_handler.extract_audio(reference_video)
                    audio_handler.merge_audio_video(temp_video, audio_path, output_video)
                    temp_video.unlink()  # Remove temp file
                    audio_path.unlink()  # Remove temp audio
                else:
                    temp_video.rename(output_video)

                print(f"✓ Job completed: {output_video}")

            except Exception as e:
                print(f"✗ Job failed: {e}")
                continue

        print(f"\n{'='*60}")
        print("Batch processing complete!")
        print(f"{'='*60}")

    def process_with_slurm(
        self,
        model_path: Union[str, Path],
        partition: str = "gpu",
        gres: str = "gpu:b200:1",
        time_limit: str = "02:00:00",
        mem: str = "64GB",
        wait: bool = False
    ) -> List[int]:
        """
        Process all jobs using SLURM array jobs.

        Args:
            model_path: Path to Wan2.1 model
            partition: SLURM partition
            gres: GPU resource specification
            time_limit: Time limit per job
            mem: Memory per job
            wait: Wait for all jobs to complete

        Returns:
            List of job IDs
        """
        if not is_slurm_available():
            raise RuntimeError("SLURM not available on this system")

        print(f"Submitting {len(self.jobs)} jobs to SLURM as array job...")

        # Save manifest where SLURM can access it
        manifest_copy = Path("data") / f"batch_{int(time.time())}.json"
        manifest_copy.parent.mkdir(parents=True, exist_ok=True

        with open(manifest_copy, 'w') as f:
            json.dump({'jobs': self.jobs}, f, indent=2)

        # Create array job script
        script_path = Path("scripts") / f"batch_array_{int(time.time())}.sh"

        script_content = f"""#!/bin/bash
#SBATCH --job-name=motion_batch
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --array=0-{len(self.jobs)-1}
#SBATCH --output=logs/batch_%A_%a.out
#SBATCH --error=logs/batch_%A_%a.err

module load cuda/12.8.1
eval "$(conda shell.bash hook)"
conda activate motion_control

# Get job index
JOB_IDX=$SLURM_ARRAY_TASK_ID

echo "Processing job $JOB_IDX"

# Run generation script with array index
python scripts/generate.py \\
    --batch {manifest_copy} \\
    --job-index $JOB_IDX \\
    --model-path {model_path}
"""

        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)

        # Submit array job
        submitter = SLURMJobSubmitter(partition=partition)
        job_id = submitter.submit_job(script_path)

        if wait:
            print("Waiting for jobs to complete...")
            submitter.wait_for_job(job_id)

        return [job_id]

    def get_progress(self) -> Dict[str, int]:
        """
        Get progress statistics.

        Returns:
            Dictionary with completion statistics
        """
        completed = 0
        failed = 0
        pending = 0

        for job in self.jobs:
            output_path = Path(job['output_video'])
            if output_path.exists():
                completed += 1
            # Additional logic to check for failed jobs could be added

        pending = len(self.jobs) - completed - failed

        return {
            'total': len(self.jobs),
            'completed': completed,
            'failed': failed,
            'pending': pending
        }


# Convenience function
def process_batch(
    manifest_path: Union[str, Path],
    model_path: Union[str, Path],
    use_slurm: bool = False,
    **kwargs
):
    """
    Process batch of video generation jobs.

    Args:
        manifest_path: Path to job manifest
        model_path: Path to Wan2.1 model
        use_slurm: Use SLURM for parallel processing
        **kwargs: Additional arguments
    """
    processor = BatchProcessor(manifest_path)

    if use_slurm and is_slurm_available():
        processor.process_with_slurm(model_path, **kwargs)
    else:
        processor.process_sequential(model_path, **kwargs)
