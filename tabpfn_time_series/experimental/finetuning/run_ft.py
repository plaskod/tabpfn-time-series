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
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf, DictConfig

from tabpfn.preprocessing import DatasetCollectionWithPreprocessing
from tabpfn.utils import collate_for_tabpfn_dataset

from tabpfn_time_series.experimental.finetuning.dataset import (
    TabPFNTimeSeriesPretrainDataset,
    load_all_ts_datasets,
    filter_constant_series,
    XTrainType,
    YTrainType,
    XTestType,
    YTestType,
)
from tabpfn_time_series.experimental.finetuning.configs.config import ConfigManager
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    TabPFNTimeSeriesModule,
)
from tabpfn_time_series.experimental.finetuning.experiment_args import common_parse_args


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


def ts_splitfn(
    X: np.ndarray, y: np.ndarray, prediction_length: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time series data into train and test sets."""
    X_train = X[:-prediction_length]
    X_test = X[-prediction_length:]
    y_train = y[:-prediction_length]
    y_test = y[-prediction_length:]

    return X_train, X_test, y_train, y_test


def setup_datasets(
    config: DictConfig,
    debug_mode: bool = False,
    hf_cache_dir: str = None,
) -> Tuple[List[XTrainType], List[YTrainType], List[XTestType], List[YTestType]]:
    """Setup datasets for training and validation."""

    train_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=config.train_datasets.dataset_repo_name,
        dataset_names=config.train_datasets.dataset_names,
        max_context_length=config.train_datasets.max_context_length,
        hf_cache_dir=hf_cache_dir,
    )

    # Get dataset length limits from config or use defaults
    train_max_length = config.get("train_max_length", None)
    test_max_length = config.get("test_max_length", None)

    # Override with debug values if in debug mode
    if debug_mode:
        train_max_length = 10
        test_max_length = 1

    logger.debug(
        f"Using train_max_length={train_max_length}, test_max_length={test_max_length}"
    )

    if config.split_train_to_val or config.use_train_as_val:
        logger.info("Loading train datasets only")
        logger.info(
            f"Train dataset: repo={train_dataset.dataset_repo_name}, names={train_dataset.dataset_names}"
        )

        all_X, all_y = load_all_ts_datasets(
            train_dataset,
            shuffle=True,
            max_length=train_max_length,
            preprocess_fn=filter_constant_series,
            prefix="train",
        )

        # Split into train and test sets
        if config.split_train_to_val:
            all_train_X, all_test_X, all_train_y, all_test_y = train_test_split(
                all_X, all_y, test_size=0.2, random_state=config.seed
            )
        elif config.use_train_as_val:
            all_train_X, all_train_y = all_X, all_y
            all_test_X, all_test_y = all_train_X, all_train_y

    else:
        logger.info("Loading train and test datasets")

        test_dataset = TabPFNTimeSeriesPretrainDataset(
            dataset_repo_name=config.test_datasets.dataset_repo_name,
            dataset_names=config.test_datasets.dataset_names,
            max_context_length=config.test_datasets.max_context_length,
            hf_cache_dir=hf_cache_dir,
        )

        logger.info(
            f"Train dataset: repo={train_dataset.dataset_repo_name}, names={train_dataset.dataset_names}"
        )
        logger.info(
            f"Test dataset: repo={test_dataset.dataset_repo_name}, names={test_dataset.dataset_names}"
        )

        all_train_X, all_train_y = load_all_ts_datasets(
            train_dataset,
            shuffle=True,
            max_length=train_max_length,
            preprocess_fn=filter_constant_series,
            prefix="train",
        )

        all_test_X, all_test_y = load_all_ts_datasets(
            test_dataset,
            shuffle=True,
            max_length=test_max_length,
            preprocess_fn=filter_constant_series,
            prefix="test",
        )

    return all_train_X, all_train_y, all_test_X, all_test_y


def setup_data_loaders(
    train_datasets_collection: DatasetCollectionWithPreprocessing,
    test_datasets_collection: DatasetCollectionWithPreprocessing,
    num_workers: int = 0,
    debug_mode: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create and configure data loaders for training and validation."""

    # Debug
    num_workers = 0

    train_dl = DataLoader(
        train_datasets_collection,
        batch_size=1,
        collate_fn=collate_for_tabpfn_dataset,
        shuffle=False,  # Already shuffled in the dataset
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
    metric_to_monitor = "val/out_of_sample/mae"

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config.experiment_name}/{config.run_name}",
        filename="tabpfn-ts-{epoch:02d}-{val_mae:.4f}",
        monitor=metric_to_monitor,
        mode="min",
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor=metric_to_monitor,
        patience=config.optimization.early_stopping_patience,
        mode="min",
        verbose=True,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.optimization.max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.optimization.gradient_accumulation_steps,
        log_every_n_steps=8,
    )

    return trainer, checkpoint_callback


def setup_config(args: argparse.Namespace) -> DictConfig:
    """Load configurations from files and override with command line arguments."""
    experiment_config = ConfigManager.load_experiment(args.experiment)
    base_model_config = ConfigManager.load_base_model(args.base_model_name)
    method_config = ConfigManager.load_method(args.method)

    for config in [experiment_config, base_model_config, method_config]:
        if config is None:
            raise ValueError(f"Failed to load configuration: {config}")

    all_configs = OmegaConf.merge(experiment_config, base_model_config, method_config)
    all_configs.experiment_name = (
        f"{all_configs.experiment_name}-{args.base_model_name}-{args.method}"
    )

    # Update configuration with command line arguments
    all_configs = ConfigManager.update_from_args(all_configs, args)

    return all_configs


def main():
    """Main entry point for the fine-tuning script."""
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    args = common_parse_args()

    debug_mode = args.debug

    # Use medium mixed precision for faster training
    torch.set_float32_matmul_precision("medium")

    # Setup configuration
    config = setup_config(args)
    logger.info(f"Configuration: \n{ConfigManager.pretty_print(config)}")

    # Set debug logging if in debug mode
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    # Debug
    # Set logging level to DEBUG
    # logging.getLogger().setLevel(logging.DEBUG)

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

    # Update run name if resuming from checkpoint
    if config.resume_from_checkpoint:
        config.run_name = f"{config.run_name}-resumed"
        if "resumed" not in tags:
            tags.append("resumed")

    # Setup Weights & Biases logger
    if args.no_wandb:
        logger.info("Weights & Biases logging disabled")
        wandb_logger = None
    else:
        wandb_logger = WandbLogger(
            project=config.wandb_project,
            name=config.run_name,
            log_model=not config.no_log_model,
            tags=tags if tags else None,
        )

        # Log hyperparameters
        wandb_logger.log_hyperparams(OmegaConf.to_container(config, resolve=True))

    # Setup datasets
    all_train_X, all_train_y, all_test_X, all_test_y = setup_datasets(
        config, debug_mode, hf_cache_dir
    )

    logger.info(
        f"Loaded {len(all_train_X)} training samples and {len(all_test_X)} test samples"
    )

    # Setup regressor
    regressor_config = {
        **OmegaConf.to_container(config.base_model, resolve=True),
        "random_state": config.seed,
        "device": "cuda",
    }

    # Setup training configuration
    training_config = {
        "lr": config.optimization.lr,
        "finetune_space": config.optimization.finetune_space,
        "ignore_negative_loss": config.loss_config.ignore_negative_loss,
        "ignore_inf_loss": config.loss_config.ignore_inf_loss,
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
        override_random_state=config.seed,
    )
    test_datasets_collection = lightning_model.regressor.get_preprocessed_datasets(
        all_test_X,
        all_test_y,
        partial(
            ts_splitfn,
            prediction_length=config.test_datasets.prediction_length
            if hasattr(config, "test_datasets")
            else config.train_datasets.prediction_length,
        ),
        max_data_size=config.tabpfn_split_max_data_size,
        override_random_state=config.seed,
    )

    # Setup data loaders
    num_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Using {num_workers} workers for data loading")

    train_dl, val_dl = setup_data_loaders(
        train_datasets_collection,
        test_datasets_collection,
        num_workers=num_workers,
        debug_mode=debug_mode,
    )

    # Setup trainer and start training
    trainer, checkpoint_callback = setup_trainer(config, wandb_logger)
    if config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")

    # import warnings
    # from lightning_fabric.utilities.warnings import PossibleUserWarning

    # warnings.filterwarnings("error", category=UserWarning)
    # warnings.filterwarnings("ignore", category=PossibleUserWarning)
    trainer.fit(
        lightning_model, train_dl, val_dl, ckpt_path=config.resume_from_checkpoint
    )

    # Log best model path
    if checkpoint_callback.best_model_path:
        logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        if wandb_logger:
            wandb_logger.experiment.log(
                {"best_model_path": checkpoint_callback.best_model_path}
            )

    # Log final skipped steps statistics
    logger.info(f"Total training steps skipped: {lightning_model.train_skipped_steps}")
    logger.info(f"Total validation steps skipped: {lightning_model.val_skipped_steps}")

    # Close wandb run
    if wandb_logger:
        wandb_logger.experiment.finish()

    return 0


if __name__ == "__main__":
    exit(main())
