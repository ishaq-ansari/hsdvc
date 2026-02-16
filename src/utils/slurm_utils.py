"""
SLURM utilities for HPC job submission and management.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union


class SLURMJobSubmitter:
    """Helper class for submitting SLURM jobs."""

    def __init__(
        self,
        partition: str = "gpu",
        account: Optional[str] = None,
        qos: Optional[str] = None
    ):
        """
        Initialize SLURM job submitter.

        Args:
            partition: SLURM partition name
            account: SLURM account name
            qos: QOS (Quality of Service)
        """
        self.partition = partition
        self.account = account
        self.qos = qos

    def submit_job(
        self,
        script_path: Union[str, Path],
        job_name: Optional[str] = None,
        dependencies: Optional[List[int]] = None,
        **sbatch_args
    ) -> int:
        """
        Submit a SLURM job.

        Args:
            script_path: Path to SLURM job script
            job_name: Job name
            dependencies: List of job IDs this job depends on
            **sbatch_args: Additional sbatch arguments

        Returns:
            Job ID

        Raises:
            RuntimeError: If job submission fails
        """
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        cmd = ["sbatch"]

        # Add job name
        if job_name:
            cmd.extend(["--job-name", job_name])

        # Add dependencies
        if dependencies:
            dep_str = ":".join(map(str, dependencies))
            cmd.extend(["--dependency", f"afterok:{dep_str}"])

        # Add additional arguments
        for key, value in sbatch_args.items():
            key = key.replace("_", "-")
            if value is True:
                cmd.append(f"--{key}")
            elif value is not False and value is not None:
                cmd.extend([f"--{key}", str(value)])

        cmd.append(str(script_path))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from output
            job_id = int(result.stdout.split()[-1])
            print(f"✓ Job submitted: {job_id}")
            return job_id

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Job submission failed: {e.stderr}") from e

    def submit_array_job(
        self,
        script_path: Union[str, Path],
        array_spec: str,
        job_name: Optional[str] = None,
        **sbatch_args
    ) -> int:
        """
        Submit a SLURM array job.

        Args:
            script_path: Path to job script
            array_spec: Array specification (e.g., "1-10", "1-100:10")
            job_name: Job name
            **sbatch_args: Additional sbatch arguments

        Returns:
            Job ID
        """
        return self.submit_job(
            script_path=script_path,
            job_name=job_name,
            array=array_spec,
            **sbatch_args
        )

    def get_job_status(self, job_id: int) -> Dict[str, str]:
        """
        Get status of a SLURM job.

        Args:
            job_id: Job ID

        Returns:
            Dictionary with job status information
        """
        try:
            result = subprocess.run(
                ["scontrol", "show", "job", str(job_id)],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output
            status = {}
            for line in result.stdout.split('\n'):
                for item in line.split():
                    if '=' in item:
                        key, value = item.split('=', 1)
                        status[key.lower()] = value

            return status

        except subprocess.CalledProcessError:
            return {"jobstate": "NOT_FOUND"}

    def cancel_job(self, job_id: int):
        """
        Cancel a SLURM job.

        Args:
            job_id: Job ID to cancel
        """
        try:
            subprocess.run(
                ["scancel", str(job_id)],
                check=True
            )
            print(f"✓ Job {job_id} cancelled")
        except subprocess.CalledProcessError as e:
            print(f"Failed to cancel job {job_id}: {e}")

    def wait_for_job(self, job_id: int, check_interval: int = 30):
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            check_interval: Check interval in seconds
        """
        import time

        print(f"Waiting for job {job_id}...")
        while True:
            status = self.get_job_status(job_id)
            state = status.get("jobstate", "UNKNOWN")

            if state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NOT_FOUND"]:
                print(f"Job {job_id} finished with state: {state}")
                break

            time.sleep(check_interval)


def create_gpu_job_script(
    output_path: Union[str, Path],
    command: str,
    job_name: str = "motion_gen",
    partition: str = "gpu",
    gres: str = "gpu:b200:1",
    ntasks: int = 1,
    cpus_per_task: int = 8,
    mem: str = "64GB",
    time: str = "02:00:00",
    conda_env: str = "motion_control",
    modules: Optional[List[str]] = None,
    setup_commands: Optional[List[str]] = None
) -> Path:
    """
    Create a SLURM GPU job script.

    Args:
        output_path: Path for job script
        command: Command to run
        job_name: Job name
        partition: SLURM partition
        gres: GPU resource specification
        ntasks: Number of tasks
        cpus_per_task: CPUs per task
        mem: Memory requirement
        time: Time limit
        conda_env: Conda environment name
        modules: List of modules to load
        setup_commands: Additional setup commands

    Returns:
        Path to created script
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if modules is None:
        modules = ["cuda/12.8.1"]

    if setup_commands is None:
        setup_commands = []

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# Load modules
"""

    for module in modules:
        script_content += f"module load {module}\n"

    script_content += f"""
# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate {conda_env}

# Setup commands
"""

    for cmd in setup_commands:
        script_content += f"{cmd}\n"

    script_content += f"""
# Run command
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

{command}

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
"""

    with open(output_path, 'w') as f:
        f.write(script_content)

    # Make executable
    output_path.chmod(0o755)

    return output_path


def get_available_gpus() -> List[str]:
    """
    Get list of available GPU types on the cluster.

    Returns:
        List of GPU types
    """
    try:
        result = subprocess.run(
            ["sinfo", "-o", "%G", "--noheader"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = set()
        for line in result.stdout.strip().split('\n'):
            if "gpu:" in line:
                # Parse format like "gpu:a100:4"
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    gpus.add(parts[1])

        return sorted(gpus)

    except subprocess.CalledProcessError:
        return []


def get_gpu_queue_status(partition: str = "gpu") -> Dict[str, int]:
    """
    Get GPU queue status.

    Args:
        partition: Partition name

    Returns:
        Dictionary with queue statistics
    """
    try:
        result = subprocess.run(
            ["squeue", "-p", partition, "-o", "%T", "--noheader"],
            capture_output=True,
            text=True,
            check=True
        )

        states = {"RUNNING": 0, "PENDING": 0, "OTHER": 0}

        for line in result.stdout.strip().split('\n'):
            state = line.strip()
            if state in states:
                states[state] += 1
            elif state:
                states["OTHER"] += 1

        return states

    except subprocess.CalledProcessError:
        return {}


# Convenience function
def is_slurm_available() -> bool:
    """Check if SLURM is available on the system."""
    try:
        subprocess.run(["which", "sbatch"], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False
