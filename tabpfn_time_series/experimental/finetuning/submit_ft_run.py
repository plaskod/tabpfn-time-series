#!/usr/bin/env python
"""
Script to submit run_ft.py as an SBATCH job using submitit.
"""

import argparse
import submitit
from datetime import datetime
from pathlib import Path

from tabpfn_time_series.experimental.finetuning.experiment_args import common_parse_args

SCRIPT_PATH = Path(__file__).parent / "run_ft.py"


def parse_args():
    # First get the base parser from run_ft.py
    ft_parser = common_parse_args(return_parser=True)

    # Create a new parser for submitit-specific arguments
    parser = argparse.ArgumentParser(
        description="Submit run_ft.py as an SBATCH job using submitit",
        parents=[ft_parser],  # Inherit all arguments from run_ft.py
        conflict_handler="resolve",  # Handle any conflicts
    )

    # Add SLURM job parameters
    slurm_group = parser.add_argument_group("SLURM Parameters")
    slurm_group.add_argument(
        "--job_name", type=str, default="tabpfn-ts-ft", help="Name for the SLURM job"
    )
    slurm_group.add_argument(
        "--partition", type=str, default="gpua100", help="SLURM partition to use"
    )
    slurm_group.add_argument(
        "--time", type=int, default=24, help="Maximum job runtime in hours"
    )
    slurm_group.add_argument(
        "--mem", type=int, default=64, help="Memory allocation in GB"
    )
    slurm_group.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs to request"
    )
    slurm_group.add_argument(
        "--cpus", type=int, default=6, help="Number of CPUs to request"
    )

    return parser.parse_args()


class TrainingJob:
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import sys
        import os

        # Print some debug information
        print(f"Job running on node: {os.uname().nodename}")
        print(f"CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        # Construct command to run run_ft.py
        cmd = [sys.executable, "-u", str(SCRIPT_PATH)]

        # Add all arguments that were passed, excluding SLURM-specific ones
        slurm_args = {"job_name", "partition", "time", "mem", "gpus", "cpus"}

        # Get all arguments from the namespace
        for arg_name, arg_value in vars(self.args).items():
            # Skip SLURM-specific arguments
            if arg_name in slurm_args:
                continue

            # Skip None values
            if arg_value is None:
                continue

            # Handle boolean flags
            if isinstance(arg_value, bool):
                if arg_value:
                    cmd.append(f"--{arg_name}")
            # Handle lists (like tags, train_datasets, etc.)
            elif isinstance(arg_value, list):
                cmd.extend([f"--{arg_name}"] + [str(v) for v in arg_value])
            # Handle all other arguments
            else:
                cmd.extend([f"--{arg_name}", str(arg_value)])

        # Execute the command
        import subprocess

        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True)
        return process.returncode


def main():
    args = parse_args()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=str(log_dir))

    # Set slurm parameters
    executor.update_parameters(
        name=args.job_name,
        slurm_partition=args.partition,
        timeout_min=args.time * 60,  # Convert hours to minutes
        slurm_mem=f"{args.mem}GB",
        gpus_per_node=args.gpus,
        cpus_per_task=args.cpus,
        slurm_additional_parameters={"comment": "TabPFN Time Series Fine-tuning"},
    )

    # Create the job
    job = TrainingJob(args)

    # Submit the job
    submitted_job = executor.submit(job)
    print(f"Submitted job with ID: {submitted_job.job_id}")
    print(f"Logs will be saved to: {log_dir}")


if __name__ == "__main__":
    main()
