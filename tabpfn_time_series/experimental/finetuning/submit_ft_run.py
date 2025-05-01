#!/usr/bin/env python
"""
Script to submit run_ft.py as an SBATCH job using submitit.
"""

import argparse
import submitit
from datetime import datetime
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "run_ft.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit run_ft.py as an SBATCH job using submitit"
    )
    # SLURM job parameters
    parser.add_argument(
        "--job_name", type=str, default="tabpfn-ts-ft", help="Name for the SLURM job"
    )
    parser.add_argument(
        "--partition", type=str, default="gpua100", help="SLURM partition to use"
    )
    parser.add_argument(
        "--time", type=int, default=24, help="Maximum job runtime in hours"
    )
    parser.add_argument("--mem", type=int, default=64, help="Memory allocation in GB")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request")
    parser.add_argument("--cpus", type=int, default=6, help="Number of CPUs to request")

    # run_ft.py parameters
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment configuration to use",
    )
    parser.add_argument(
        "--method", type=str, help="Name of the method configuration to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )
    parser.add_argument(
        "--num_workers", type=int, help="Number of worker processes for data loading"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-ft",
        help="Weights & Biases project name",
    )
    parser.add_argument("--run_name", type=str, help="Custom name for the W&B run")
    parser.add_argument("--tags", type=str, nargs="+", help="Tags for the W&B run")
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--train_datasets", type=str, nargs="+", help="Names of training datasets"
    )
    parser.add_argument(
        "--test_datasets", type=str, nargs="+", help="Names of test datasets"
    )
    parser.add_argument(
        "--log_model", type=bool, default=True, help="Log the model to Weights & Biases"
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

        # Add all arguments that were passed
        if self.args.experiment:
            cmd.extend(["--experiment", self.args.experiment])

        if self.args.method:
            cmd.extend(["--method", self.args.method])

        if self.args.device:
            cmd.extend(["--device", self.args.device])

        if self.args.seed is not None:
            cmd.extend(["--seed", str(self.args.seed)])

        if self.args.debug:
            cmd.append("--debug")

        if self.args.num_workers is not None:
            cmd.extend(["--num_workers", str(self.args.num_workers)])

        if self.args.wandb_project:
            cmd.extend(["--wandb_project", self.args.wandb_project])

        if self.args.run_name:
            cmd.extend(["--run_name", self.args.run_name])

        if self.args.tags:
            cmd.extend(["--tags"] + self.args.tags)

        if self.args.max_epochs is not None:
            cmd.extend(["--max_epochs", str(self.args.max_epochs)])

        if self.args.lr is not None:
            cmd.extend(["--lr", str(self.args.lr)])

        if self.args.train_datasets:
            cmd.extend(["--train_datasets"] + self.args.train_datasets)

        if self.args.test_datasets:
            cmd.extend(["--test_datasets"] + self.args.test_datasets)

        if self.args.log_model is not None:
            cmd.extend(["--log_model", str(self.args.log_model)])

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
