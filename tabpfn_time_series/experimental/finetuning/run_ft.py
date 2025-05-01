#!/usr/bin/env python3
"""
Fine-tuning runner for TabPFN Time Series models with experiment types.
"""

import os
import logging
import argparse
import random
from functools import partial
import numpy as np
from dotenv import load_dotenv
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf, DictConfig

from tabpfn.preprocessing import DatasetCollectionWithPreprocessing
from tabpfn.utils import collate_for_tabpfn_dataset

from tabpfn_time_series.experimental.finetuning.dataset import (
    TabPFNTimeSeriesPretrainDataset,
    load_all_ts_datasets,
)
from tabpfn_time_series.experimental.finetuning.configs.config import ConfigManager
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    TabPFNTimeSeriesModule,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN Time Series models")

    # Configuration
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment configuration to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Name of the method configuration to use (overrides experiment's method)",
    )

    # Runtime settings
    parser.add_argument(
        "--device", type=str, help="Device to use for training (cuda or cpu)"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced dataset size",
    )

    # Logging settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-ft",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="dummy-run-name",
        help="Custom name for the W&B run",
    )
    parser.add_argument("--tags", type=str, nargs="+", help="Tags for the W&B run")

    # Training settings
    parser.add_argument(
        "--max_epochs", type=int, help="Maximum number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--log_model", type=bool, default=True, help="Log the model to Weights & Biases"
    )

    # Dataset settings
    parser.add_argument(
        "--train_datasets", type=str, nargs="+", help="Names of training datasets"
    )
    parser.add_argument(
        "--test_datasets", type=str, nargs="+", help="Names of test datasets"
    )

    return parser.parse_args()


def ts_splitfn(
    X: np.ndarray, y: np.ndarray, prediction_length: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time series data into train and test sets."""
    X_train = X[:-prediction_length]
    X_test = X[-prediction_length:]
    y_train = y[:-prediction_length]
    y_test = y[-prediction_length:]

    return X_train, X_test, y_train, y_test


def setup_data_loaders(
    train_datasets_collection: DatasetCollectionWithPreprocessing,
    test_datasets_collection: DatasetCollectionWithPreprocessing,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create and configure data loaders for training and validation."""

    # Debug
    num_workers = 0

    train_dl = DataLoader(
        train_datasets_collection,
        batch_size=1,
        collate_fn=collate_for_tabpfn_dataset,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    val_dl = DataLoader(
        test_datasets_collection,
        batch_size=1,
        collate_fn=collate_for_tabpfn_dataset,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    return train_dl, val_dl


def setup_trainer(config: DictConfig, wandb_logger):
    """Configure and initialize the PyTorch Lightning trainer."""
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="tabpfn-ts-{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.optimization.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.optimization.n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.optimization.gradient_accumulation_steps,
        log_every_n_steps=10,
    )

    return trainer, checkpoint_callback


def main():
    """Main entry point for the fine-tuning script."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    args = parse_args()

    debug_mode = args.debug

    # Use medium mixed precision for faster training
    torch.set_float32_matmul_precision("medium")

    # Load experiment configuration
    config: DictConfig = ConfigManager.load_experiment(args.experiment)
    if config is None:
        logger.error(f"Failed to load experiment configuration: {args.experiment}")
        return 1

    # Override method if specified
    if args.method:
        method_config = ConfigManager.load_method(args.method)
        if method_config is None:
            logger.error(f"Failed to load method configuration: {args.method}")
            return 1

        # Update config with method settings
        config = OmegaConf.merge(config, method_config)

        # Update experiment name to reflect the method change
        config.experiment_name = f"{config.experiment_name}-{args.method}"

    # Update configuration with command line arguments
    config = ConfigManager.update_from_args(config, args)

    # Set debug logging if in debug mode
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Get HF cache directory from environment
    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    if hf_cache_dir is None:
        logger.warning(
            "HF_CACHE_DIR environment variable not set. Using default cache directory."
        )

    # Set random seed for reproducibility
    seed = config.seed
    set_seed(seed)

    # Setup tags for W&B
    tags = config.tags
    if debug_mode:
        tags.append("debug")

    # Setup Weights & Biases logger
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.run_name,
        log_model=config.log_model,
        tags=tags if tags else None,
    )

    # Log hyperparameters
    wandb_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # Initialize datasets
    train_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=config.train_datasets.dataset_repo_name,
        dataset_names=config.train_datasets.dataset_names,
        max_context_length=config.train_datasets.max_context_length,
        hf_cache_dir=hf_cache_dir,
    )

    test_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=config.test_datasets.dataset_repo_name,
        dataset_names=config.test_datasets.dataset_names,
        max_context_length=config.test_datasets.max_context_length,
        hf_cache_dir=hf_cache_dir,
    )

    # Prepare datasetsd
    train_max_length = 20 if debug_mode else None
    test_max_length = 5 if debug_mode else None

    all_train_X, all_train_y = load_all_ts_datasets(
        train_dataset, max_length=train_max_length
    )
    all_test_X, all_test_y = load_all_ts_datasets(
        test_dataset, max_length=test_max_length
    )

    logger.info(
        f"Loaded {len(all_train_X)} training samples and {len(all_test_X)} test samples"
    )

    # Setup regressor
    precision_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    regressor_config = {
        "ignore_pretraining_limits": config.base_model.ignore_pretraining_limits,
        "n_estimators": config.base_model.n_estimators,
        "random_state": config.seed,
        "inference_precision": precision_map.get(config.base_model.inference_precision),
        "device": "cuda",
    }

    # Setup training configuration
    training_config = {
        "lr": config.optimization.lr,
        "finetune_space": config.optimization.finetune_space,
    }

    # Initialize PyTorch Lightning module
    lightning_model = TabPFNTimeSeriesModule(
        training_config=training_config,
        tabpfn_model_config=regressor_config,
    )

    # Preprocess datasets for TabPFN
    train_datasets_collection = lightning_model.regressor.get_preprocessed_datasets(
        all_train_X,
        all_train_y,
        partial(ts_splitfn, prediction_length=config.train_datasets.prediction_length),
        max_data_size=config.tabpfn_split_max_data_size,
    )
    test_datasets_collection = lightning_model.regressor.get_preprocessed_datasets(
        all_test_X,
        all_test_y,
        partial(ts_splitfn, prediction_length=config.test_datasets.prediction_length),
        max_data_size=config.tabpfn_split_max_data_size,
    )

    # Setup data loaders
    num_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Using {num_workers} workers for data loading")

    train_dl, val_dl = setup_data_loaders(
        train_datasets_collection, test_datasets_collection, num_workers=num_workers
    )

    # Setup trainer and start training
    trainer, checkpoint_callback = setup_trainer(config, wandb_logger)
    trainer.fit(lightning_model, train_dl, val_dl)

    # Log best model path
    if checkpoint_callback.best_model_path:
        logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        wandb_logger.experiment.log(
            {"best_model_path": checkpoint_callback.best_model_path}
        )

    # Log final skipped steps statistics
    logger.info(f"Total training steps skipped: {lightning_model.train_skipped_steps}")
    logger.info(f"Total validation steps skipped: {lightning_model.val_skipped_steps}")

    # Close wandb run
    wandb_logger.experiment.finish()

    return 0


if __name__ == "__main__":
    exit(main())
