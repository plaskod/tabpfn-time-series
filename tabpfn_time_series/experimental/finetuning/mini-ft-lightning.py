"""
Mini fine-tuning script for TabPFN Time Series models using PyTorch Lightning with Weights & Biases logging.
"""

import os
import logging
import argparse
from typing import Tuple, Any
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tabpfn import TabPFNRegressor
from tabpfn.utils import collate_for_tabpfn_dataset
from tabpfn.finetune_utils import _prepare_eval_model
from tabpfn_time_series.experimental.finetuning.dataset import (
    TabPFNTimeSeriesPretrainDataset,
    load_all_ts_datasets,
)
from tabpfn_time_series.experimental.finetuning.ft_config import FinetuneConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def ts_splitfn(
    X: np.ndarray, y: np.ndarray, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split time series data into train and test sets."""
    prediction_length = 100
    X_train = X[:-prediction_length]
    X_test = X[-prediction_length:]
    y_train = y[:-prediction_length]
    y_test = y[-prediction_length:]

    return X_train, X_test, y_train, y_test


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    mse: float
    mae: float
    r2: float


class TabPFNTimeSeriesModule(pl.LightningModule):
    """PyTorch Lightning module for TabPFN Time Series fine-tuning."""

    def __init__(
        self,
        regressor: TabPFNRegressor,
        opt_config: Any,
        model_config: dict,
    ):
        super().__init__()
        self.regressor = regressor
        self.model = regressor.model_  # for Lightning to track
        self.opt_config = opt_config
        self.model_config = model_config
        self.save_hyperparameters(ignore=["regressor"])
        self.train_skipped_steps = 0
        self.val_skipped_steps = 0
        self.current_epoch_train_skipped = 0
        self.current_epoch_val_skipped = 0

    def forward(self, X_tests_preprocessed):
        """Forward pass through the regressor."""
        return self.regressor.forward(X_tests_preprocessed)

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        # Unpack batch
        batch_data = self._unpack_batch(batch)
        if batch_data is None:
            return None

        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            renormalized_criterion,
            x_test_raw,
            y_test_raw,
            x_train_raw,
            y_train_raw,
        ) = batch_data

        # Forward pass
        self.regressor.fit_from_preprocessed(
            X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
        )
        averaged_pred_logits, _, _ = self.forward(X_tests_preprocessed)

        # Calculate loss
        loss_fn = renormalized_criterion
        loss = self._calculate_loss(
            averaged_pred_logits, y_test_standardized, loss_fn, "train"
        )
        if loss is None:
            logging.warning("Train loss is None")
            return None

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        # Unpack batch
        batch_data = self._unpack_batch(batch)
        if batch_data is None:
            return None

        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            renormalized_criterion,
            x_test_raw,
            y_test_raw,
            x_train_raw,
            y_train_raw,
        ) = batch_data

        # Get a copy of TabPFNRegressor
        eval_model = _prepare_eval_model(
            self.regressor, self.model_config, TabPFNRegressor
        )

        # Forward pass
        eval_model.fit_from_preprocessed(
            X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
        )
        averaged_pred_logits, _, _ = eval_model.forward(X_tests_preprocessed)

        # Calculate loss
        loss_fn = renormalized_criterion
        loss = self._calculate_loss(
            averaged_pred_logits, y_test_standardized, loss_fn, "val"
        )
        if loss is None:
            logging.warning("Validation loss is None")
            return None

        # Calculate regression metrics
        metrics = self._calculate_regression_metrics(
            eval_model, x_train_raw, y_train_raw, x_test_raw, y_test_raw
        )

        # Log metrics
        self.log("val/mse", metrics.mse, on_epoch=True, batch_size=1)
        self.log("val/mae", metrics.mae, on_epoch=True, batch_size=1)
        self.log("val/r2", metrics.r2, on_epoch=True, batch_size=1)
        self.log("val/loss", loss, on_epoch=True, batch_size=1)

        return {"loss": loss, "mse": metrics.mse, "mae": metrics.mae, "r2": metrics.r2}

    def _unpack_batch(self, batch):
        """Unpack and preprocess batch data."""
        try:
            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_preprocessed,
                y_test_standardized,
                cat_ixs,
                confs,
                renormalized_criterion,
                batch_x_test_raw,
                batch_y_test_raw,
                batch_x_train_raw,
                batch_y_train_raw,
            ) = batch

            assert len(renormalized_criterion) == 1
            renormalized_criterion = renormalized_criterion[0]

            assert batch_x_test_raw.shape[0] == 1
            assert batch_y_test_raw.shape[0] == 1
            x_test_raw = batch_x_test_raw[0]
            y_test_raw = batch_y_test_raw[0]

            assert batch_x_train_raw.shape[0] == 1
            assert batch_y_train_raw.shape[0] == 1
            x_train_raw = batch_x_train_raw[0]
            y_train_raw = batch_y_train_raw[0]

            return (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_preprocessed,
                y_test_standardized,
                cat_ixs,
                confs,
                renormalized_criterion,
                x_test_raw,
                y_test_raw,
                x_train_raw,
                y_train_raw,
            )

        except Exception as e:
            logging.error(f"Error unpacking batch: {e}")
            return None

    def _check_numerical_issues(self, tensor, prefix):
        """Check for NaN/Inf values in tensors."""
        has_issues = False

        if torch.isnan(tensor).any():
            logging.warning(f"NaN detected in {prefix} tensor")
            self.log(f"{prefix}/nan_detected", 1.0)
            has_issues = True

        if torch.isinf(tensor).any():
            logging.warning(f"Inf detected in {prefix} tensor")
            self.log(f"{prefix}/inf_detected", 1.0)
            has_issues = True

        return has_issues

    def _calculate_loss(self, pred_logits, targets, loss_fn, prefix):
        """Calculate loss and handle numerical issues."""
        nll_loss_per_sample = loss_fn(pred_logits, targets.to(self.device))
        loss = nll_loss_per_sample.mean()

        # Skip infinite loss values
        if torch.isinf(loss).any():
            self.log(f"{prefix}/inf_loss_detected", 1.0)
            if prefix == "train":
                self.train_skipped_steps += 1
                self.current_epoch_train_skipped += 1
            else:
                self.val_skipped_steps += 1
                self.current_epoch_val_skipped += 1
            return None

        return loss

    @staticmethod
    def _calculate_regression_metrics(
        model: TabPFNRegressor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> EvalResult:
        """Calculate standard regression metrics for model evaluation."""

        # Convert tensors to CPU for sklearn compatibility
        X_train_cpu = (
            X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        )
        y_train_cpu = (
            y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        )
        X_test_cpu = (
            X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        )
        y_test_cpu = (
            y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
        )

        # Fit model and generate predictions
        model.fit(X_train_cpu, y_train_cpu)
        predictions = model.predict(X_test_cpu)

        # Calculate metrics
        mse = mean_squared_error(y_test_cpu, predictions)
        mae = mean_absolute_error(y_test_cpu, predictions)
        r2 = r2_score(y_test_cpu, predictions)

        return EvalResult(mse=mse, mae=mae, r2=r2)

        # except Exception as e:
        #     logging.error(f"Error calculating metrics: {e}")
        #     return EvalResult(mse=float('nan'), mae=float('nan'), r2=float('nan'))

    def on_train_epoch_end(self):
        """Log training statistics at the end of each epoch."""
        self.log("train/skipped_steps_epoch", self.current_epoch_train_skipped)
        self.log("train/total_skipped_steps", self.train_skipped_steps)
        logging.info(
            f"Epoch {self.current_epoch}: Skipped {self.current_epoch_train_skipped} training steps"
        )
        self.current_epoch_train_skipped = 0

    def on_validation_epoch_end(self):
        """Log validation statistics at the end of each epoch."""
        self.log("val/skipped_steps_epoch", self.current_epoch_val_skipped)
        self.log("val/total_skipped_steps", self.val_skipped_steps)
        logging.info(
            f"Epoch {self.current_epoch}: Skipped {self.current_epoch_val_skipped} validation steps"
        )
        self.current_epoch_val_skipped = 0

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        trainable_params = sum(
            p.numel() for p in self.regressor.model_.parameters() if p.requires_grad
        )
        logging.info(
            f"Debug, optimizer:Model has {trainable_params} trainable parameters"
        )
        return torch.optim.Adam(
            self.regressor.model_.parameters(), lr=self.opt_config.lr
        )


def prepare_datasets(train_dataset, test_dataset, debug_mode=False):
    """Prepare and potentially truncate datasets based on debug mode."""
    # Load all time series datasets
    logging.info("Loading all time series datasets...")

    train_max_length = 100 if debug_mode else None
    test_max_length = 20 if debug_mode else None
    all_train_X, all_train_y = load_all_ts_datasets(
        train_dataset, max_length=train_max_length
    )
    all_test_X, all_test_y = load_all_ts_datasets(
        test_dataset, max_length=test_max_length
    )

    logging.info(
        f"Loaded {len(all_train_X)} training samples and {len(all_test_X)} test samples"
    )
    logging.info(
        f"Lengths: (all_train_X, {len(all_train_X)}), (all_train_y, {len(all_train_y)}), "
        f"(all_test_X, {len(all_test_X)}), (all_test_y, {len(all_test_y)})"
    )

    return all_train_X, all_train_y, all_test_X, all_test_y


def parse_model_config(model_config: FinetuneConfig.Model) -> dict:
    """Parse the model config into a dictionary."""
    precision_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    config = asdict(model_config)
    config["inference_precision"] = precision_map[config["inference_precision"]]
    return config


def setup_data_loaders(
    train_datasets_collection, test_datasets_collection, num_workers=0
):
    """Create and configure data loaders for training and validation."""
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


def setup_trainer(opt_config, wandb_logger):
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
        monitor="val/loss", patience=5, mode="min", verbose=True
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=opt_config.n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        gradient_clip_val=1.0,
        accumulate_grad_batches=opt_config.gradient_accumulation_steps,
        log_every_n_steps=10,
    )

    return trainer, checkpoint_callback


DEBUG_MESSAGE = """
#########################################################
#                                                       #
#                      DEBUG MODE                       #
#                                                       #
#########################################################
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning script for TabPFN Time Series models"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-ft",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Custom name for the W&B run"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--tags", type=str, nargs="+", default=None, help="Tags for the W&B run"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading",
    )
    return parser.parse_args()


def main():
    """Main entry point for the fine-tuning script."""
    # Load environment variables
    load_dotenv()
    HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
    if HF_CACHE_DIR is None:
        logging.warning(
            "HF_CACHE_DIR environment variable not set. Using default cache directory."
        )

    # Parse command line arguments
    args = parse_args()
    device = args.device
    debug_mode = args.debug
    num_workers = args.num_workers

    if debug_mode:
        logging.basicConfig(level=logging.DEBUG)
        print(DEBUG_MESSAGE)

    # Load hyperparams
    all_config = FinetuneConfig()
    opt_config = all_config.optimization

    # Override epochs if specified in command line
    if args.max_epochs is not None:
        opt_config.n_epochs = args.max_epochs
    elif debug_mode:
        # Reduce epochs in debug mode if not explicitly set
        opt_config.n_epochs = min(2, opt_config.n_epochs)

    # Setup tags for W&B
    tags = args.tags or []
    if debug_mode:
        tags.append("debug")

    # Setup Weights & Biases logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        log_model=True,
        tags=tags if tags else None,
    )

    # Log hyperparameters
    config_dict = all_config.to_dict()
    config_dict["debug_mode"] = debug_mode
    wandb_logger.log_hyperparams(config_dict)

    # Initialize datasets
    train_dataset_config = all_config.train_dataset
    train_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=train_dataset_config.dataset_repo_name,
        dataset_names=train_dataset_config.dataset_names,
        max_context_length=train_dataset_config.max_context_length,
        hf_cache_dir=HF_CACHE_DIR,
    )

    test_dataset_config = all_config.test_dataset
    test_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=test_dataset_config.dataset_repo_name,
        dataset_names=test_dataset_config.dataset_names,
        max_context_length=test_dataset_config.max_context_length,
        hf_cache_dir=HF_CACHE_DIR,
    )

    # Prepare datasets
    all_train_X, all_train_y, all_test_X, all_test_y = prepare_datasets(
        train_dataset, test_dataset, debug_mode
    )

    # for debug purpose
    # Debug: Override all target values with random values between 0.4 and 0.5
    import numpy as np

    # Process all training datasets
    for i in range(len(all_train_y)):
        if isinstance(all_train_y[i], np.ndarray):
            # Generate random values between 0.4 and 0.5
            random_values = np.random.uniform(0.4, 0.5, size=all_train_y[i].shape)
            all_train_y[i] = random_values

    # Process all test datasets
    for i in range(len(all_test_y)):
        if isinstance(all_test_y[i], np.ndarray):
            # Generate random values between 0.4 and 0.5
            random_values = np.random.uniform(0.4, 0.5, size=all_test_y[i].shape)
            all_test_y[i] = random_values

    logging.info(
        "Debug mode: Overrode all target values with random values between 0.4 and 0.5"
    )

    # Setup regressor
    model_config = parse_model_config(all_config.model)
    model_config["device"] = device
    reg = TabPFNRegressor(**model_config)

    # Preprocess datasets for TabPFN
    preprocessing_config = all_config.preprocessing
    train_datasets_collection = reg.get_preprocessed_datasets(
        all_train_X,
        all_train_y,
        ts_splitfn,
        max_data_size=preprocessing_config.max_data_size,
    )
    test_datasets_collection = reg.get_preprocessed_datasets(
        all_test_X,
        all_test_y,
        ts_splitfn,
        max_data_size=preprocessing_config.max_data_size,
    )

    # Setup data loaders
    train_dl, val_dl = setup_data_loaders(
        train_datasets_collection, test_datasets_collection, num_workers=num_workers
    )

    # Setup trainer
    trainer, checkpoint_callback = setup_trainer(opt_config, wandb_logger)

    # Initialize PyTorch Lightning module
    model = TabPFNTimeSeriesModule(reg, opt_config, model_config)

    # Train the model
    trainer.fit(model, train_dl, val_dl)

    # Log best model path
    if checkpoint_callback.best_model_path:
        logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        wandb_logger.experiment.log(
            {"best_model_path": checkpoint_callback.best_model_path}
        )

    # Log final skipped steps statistics
    logging.info(f"Total training steps skipped: {model.train_skipped_steps}")
    logging.info(f"Total validation steps skipped: {model.val_skipped_steps}")

    # Close wandb run
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
