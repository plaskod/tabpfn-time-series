import logging
from typing import Any
from dataclasses import dataclass

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tabpfn import TabPFNRegressor
from tabpfn.regressor import _logits_to_output
from tabpfn.finetune_utils import _prepare_eval_model


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    mse: float
    mae: float
    r2: float


TABPFN_ENABLE_FINETUNING_KWARGS = {
    "differentiable_input": False,
    "fit_mode": "batched",
}


class TabPFNTimeSeriesModule(pl.LightningModule):
    """PyTorch Lightning module for TabPFN Time Series fine-tuning."""

    def __init__(
        self,
        training_config: Any,
        tabpfn_model_config: dict,
    ):
        super().__init__()
        self.tabpfn_model_config = tabpfn_model_config
        self.regressor = TabPFNRegressor(
            **self.tabpfn_model_config, **TABPFN_ENABLE_FINETUNING_KWARGS
        )
        self.model = None  # Only used for tracking, will be initialized before fitting
        self.training_config = training_config
        self.save_hyperparameters(ignore=["regressor"])
        self.train_skipped_steps = 0
        self.val_skipped_steps = 0
        self.current_epoch_train_skipped = 0
        self.current_epoch_val_skipped = 0

    def on_fit_start(self):
        """Initialize the model."""
        # model_ of tabpfn is only initialized after fit_from_preprocessed
        # hacky, consider ignoring this.
        self.model = self.regressor.model_

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
            bar_distribution,
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
        loss = self._calculate_loss(
            renormalized_criterion=renormalized_criterion,
            bar_distribution=bar_distribution,
            pred_logits=averaged_pred_logits,
            targets=y_test_standardized,
            prefix="train",
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
            bar_distribution,
            x_test_raw,
            y_test_raw,
            x_train_raw,
            y_train_raw,
        ) = batch_data

        # Get a copy of TabPFNRegressor
        # Need to use FINETUNING mode to allow loss calculation
        eval_model = _prepare_eval_model(
            original_model=self.regressor,
            eval_init_args={
                **self.tabpfn_model_config,
                **TABPFN_ENABLE_FINETUNING_KWARGS,
            },
            model_class=TabPFNRegressor,
        )

        # Forward pass
        eval_model.fit_from_preprocessed(
            X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
        )
        averaged_pred_logits, individual_pred_logits, borders = eval_model.forward(
            X_tests_preprocessed,
        )

        # Calculate loss
        loss = self._calculate_loss(
            renormalized_criterion=renormalized_criterion,
            bar_distribution=bar_distribution,
            pred_logits=averaged_pred_logits,
            targets=y_test_standardized,
            prefix="val",
        )
        if loss is None:
            logging.warning("Validation loss is None")
            return None

        # Transform logits to target
        transformed_logits = self.regressor.transform_logits(
            outputs=individual_pred_logits,
            borders=borders,
            bardist_borders=self.regressor.bardist_.borders,
            device=self.device,
        )
        median_pred = _logits_to_output(
            output_type="median",
            logits=transformed_logits,
            criterion=renormalized_criterion.cpu(),
            quantiles=None,
        )

        # Calculate regression metrics
        metrics = self._calculate_regression_metrics(
            median_predictions=median_pred,
            y_test=y_test_raw.cpu().numpy(),
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
                bar_distribution,
                batch_x_test_raw,
                batch_y_test_raw,
                batch_x_train_raw,
                batch_y_train_raw,
            ) = batch

            assert len(renormalized_criterion) == 1
            renormalized_criterion = renormalized_criterion[0]

            assert len(bar_distribution) == 1
            bar_distribution = bar_distribution[0]

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
                bar_distribution,
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

    def _calculate_loss(
        self,
        renormalized_criterion,
        bar_distribution,
        pred_logits,
        targets,
        prefix,
    ):
        """Calculate loss and handle numerical issues."""

        # Select loss function based on optimization space
        if self.training_config["finetune_space"] == "raw":
            loss_fn = renormalized_criterion
        elif self.training_config["finetune_space"] == "preprocessed":
            loss_fn = bar_distribution
        else:
            raise ValueError(
                f"Invalid optimization space: {self.training_config['finetune_space']}"
            )

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
        median_predictions: np.ndarray,
        y_test: np.ndarray,
    ) -> EvalResult:
        """Calculate standard regression metrics for model evaluation."""

        # Calculate metrics
        mse = mean_squared_error(y_test, median_predictions)
        mae = mean_absolute_error(y_test, median_predictions)
        r2 = r2_score(y_test, median_predictions)

        return EvalResult(mse=mse, mae=mae, r2=r2)

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
        return torch.optim.Adam(
            self.regressor.model_.parameters(), lr=self.training_config["lr"]
        )
