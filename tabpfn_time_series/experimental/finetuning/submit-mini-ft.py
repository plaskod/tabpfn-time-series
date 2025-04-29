#!/usr/bin/env python
"""
Script to submit mini-ft-lightning.py as an SBATCH job using submitit.
"""

import os
import argparse
import submitit
from datetime import datetime
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "mini-ft-lightning.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Submit mini-ft-lightning.py as an SBATCH job using submitit")
    parser.add_argument("--job_name", type=str, default="tabpfn-ts-ft", help="Name for the SLURM job")
    parser.add_argument("--partition", type=str, default="gpua100", help="SLURM partition to use")
    parser.add_argument("--time", type=int, default=24, help="Maximum job runtime in hours")
    parser.add_argument("--mem", type=int, default=64, help="Memory allocation in GB")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request")
    parser.add_argument("--cpus", type=int, default=6, help="Number of CPUs to request")
    parser.add_argument("--wandb_project", type=str, default="tabpfn-ts-ft", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the W&B run")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs to train")
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
        
        # Construct command to run mini-ft-lightning.py
        cmd = [sys.executable, "-u", SCRIPT_PATH, "--device", "cuda"]
        
        # Add optional arguments if provided
        if self.args.wandb_project:
            cmd.extend(["--wandb_project", self.args.wandb_project])
        
        if self.args.wandb_entity:
            cmd.extend(["--wandb_entity", self.args.wandb_entity])
        
        if self.args.run_name:
            cmd.extend(["--run_name", self.args.run_name])
        
        if self.args.max_epochs:
            cmd.extend(["--max_epochs", str(self.args.max_epochs)])
        
        # Execute the command
        import subprocess
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
        slurm_additional_parameters={
            'comment': 'TabPFN Time Series Fine-tuning'
        }
    )
    
    # Create the job
    job = TrainingJob(args)
    
    # Submit the job
    submitted_job = executor.submit(job)
    print(f"Submitted job with ID: {submitted_job.job_id}")
    print(f"Logs will be saved to: {log_dir}")


if __name__ == "__main__":
    main()
