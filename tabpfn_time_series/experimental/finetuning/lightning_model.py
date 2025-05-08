import os
import logging
from dataclasses import dataclass
from functools import partial
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, r2_score
from gluonts.evaluation.metrics import smape, mse
from tabpfn_time_series.experimental.finetuning.metrics import nmae

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import _prepare_eval_model


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    mse: float
    mae: float
    r2: float
    smape: float
    nmae: float


TABPFN_ENABLE_FINETUNING_KWARGS = {
    "differentiable_input": False,
    "fit_mode": "batched",
}

TABPFN_FINETUNING_FIXED_BATCH_SIZE = 1

PRECISION_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
}


class TabPFNTimeSeriesModule(pl.LightningModule):
    """PyTorch Lightning module for TabPFN Time Series fine-tuning."""

    def __init__(
        self,
        training_config: dict,
        tabpfn_model_config: dict,
    ):
        super().__init__()
        self.tabpfn_model_config = self._parse_model_config(tabpfn_model_config)
        self.regressor = TabPFNRegressor(
            **self.tabpfn_model_config,
            **TABPFN_ENABLE_FINETUNING_KWARGS,
        )
        self.training_config = training_config
        self.save_hyperparameters(ignore=["regressor"])

        self.num_total_ignored_train_target_points = 0
        self.num_current_epoch_ignored_train_target_points = 0
        self.num_total_ignored_val_target_points = 0
        self.num_current_epoch_ignored_val_target_points = 0
        self.train_skipped_steps = 0
        self.current_epoch_train_skipped = 0
        self.val_skipped_steps = 0
        self.current_epoch_val_skipped = 0

        self.eval_model = None
        # Log with batch size 1
        self.log = partial(self.log, batch_size=1)

    def forward(self, X_tests_preprocessed):
        """Forward pass through the regressor."""
        return self.regressor.forward(X_tests_preprocessed)

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        # Unpack batch
        logging.debug(f"Training step batch: {batch_idx}")
        batch_data = self._unpack_batch(batch)
        if batch_data is None:
            raise ValueError(f"Batch {batch_idx} is None")

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

        # Debug compute hash of the data
        # self._report_batch_hash(batch_data)

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
            targets_standardized=y_test_standardized,
            targets_raw=y_test_raw,
            prefix="train",
            batch_idx=batch_idx,
        )
        if loss is None:
            logging.warning(f"Batch: {batch_idx}, Train loss is None")

            if os.environ.get("DEBUG_SAVE_RAW_DATA"):
                self._save_debug_data(
                    batch_idx,
                    x_train_raw,
                    y_train_raw,
                    x_test_raw,
                    y_test_raw,
                    "train_bad",
                )

            self.train_skipped_steps += 1
            self.current_epoch_train_skipped += 1
            return None

        else:
            logging.debug(
                f"Current epoch: {self.current_epoch}, batch: {batch_idx}, train loss: {loss}"
            )

        # Only log if loss is finite
        if not torch.isinf(loss).any():
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

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

        # # Get a copy of TabPFNRegressor
        # # Need to use FINETUNING mode to allow loss calculation
        # eval_model = _prepare_eval_model(
        #     original_model=self.regressor,
        #     eval_init_args={
        #         **self.tabpfn_model_config,
        #         **TABPFN_ENABLE_FINETUNING_KWARGS,
        #     },
        #     model_class=TabPFNRegressor,
        # )

        # # Forward pass
        # eval_model.fit_from_preprocessed(
        #     X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
        # )
        # averaged_pred_logits, individual_pred_logits, borders = eval_model.forward(
        #     X_tests_preprocessed,
        # )

        # # Calculate loss
        # loss = self._calculate_loss(
        #     renormalized_criterion=renormalized_criterion,
        #     bar_distribution=bar_distribution,
        #     pred_logits=averaged_pred_logits,
        #     targets_standardized=y_test_standardized,
        #     targets_raw=y_test_raw,
        #     prefix="val",
        #     batch_idx=batch_idx,
        # )
        # if loss is None:
        #     logging.warning("Validation loss is None")
        #     return None

        # # Transform logits to target
        # transformed_logits = self.regressor.transform_logits(
        #     outputs=individual_pred_logits,
        #     borders=borders,
        #     bardist_borders=self.regressor.bardist_.borders,
        #     device=self.device,
        # )
        # median_pred = _logits_to_output(
        #     output_type="median",
        #     logits=transformed_logits,
        #     criterion=renormalized_criterion.cpu(),
        #     quantiles=None,
        # )

        x_train_raw_numpy = x_train_raw.cpu().numpy()
        y_train_raw_numpy = y_train_raw.cpu().numpy()
        x_test_raw_numpy = x_test_raw.cpu().numpy()
        y_test_raw_numpy = y_test_raw.cpu().numpy()

        self.eval_model.fit(x_train_raw_numpy, y_train_raw_numpy)
        full_pred_on_test = self.eval_model.predict(
            x_test_raw_numpy, output_type="full"
        )

        metrics_on_test = self._calculate_regression_metrics(
            median_predictions=full_pred_on_test,
            y_test=y_test_raw_numpy,
        )

        # Log metrics and prepare return dictionary
        result_dict = {}
        for field_name in EvalResult.__dataclass_fields__:
            field_value = getattr(metrics_on_test, field_name)
            self.log(
                f"val/{field_name}",
                field_value,
                on_step=True,
                on_epoch=True,
            )
            result_dict[field_name] = field_value

        if self.training_config["also_validate_on_train"]:
            full_pred_on_train = self.eval_model.predict(
                x_train_raw_numpy, output_type="full"
            )
            metrics_on_train = self._calculate_regression_metrics(
                median_predictions=full_pred_on_train,
                y_test=y_train_raw_numpy,
            )
            for field_name in EvalResult.__dataclass_fields__:
                field_value = getattr(metrics_on_train, field_name)
                self.log(
                    f"val/on_train_{field_name}",
                    field_value,
                    on_step=True,
                    on_epoch=True,
                )

        return result_dict

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

            assert len(renormalized_criterion) == TABPFN_FINETUNING_FIXED_BATCH_SIZE
            renormalized_criterion = renormalized_criterion[0]

            assert len(bar_distribution) == TABPFN_FINETUNING_FIXED_BATCH_SIZE
            bar_distribution = bar_distribution[0]

            assert batch_x_test_raw.shape[0] == TABPFN_FINETUNING_FIXED_BATCH_SIZE
            assert batch_y_test_raw.shape[0] == TABPFN_FINETUNING_FIXED_BATCH_SIZE
            x_test_raw = batch_x_test_raw[0]
            y_test_raw = batch_y_test_raw[0]

            assert batch_x_train_raw.shape[0] == TABPFN_FINETUNING_FIXED_BATCH_SIZE
            assert batch_y_train_raw.shape[0] == TABPFN_FINETUNING_FIXED_BATCH_SIZE
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

    def _calculate_loss(
        self,
        renormalized_criterion,
        bar_distribution,
        pred_logits,
        targets_standardized,
        targets_raw,
        prefix,
        batch_idx,
    ):
        """Calculate loss and handle numerical issues."""

        # Select loss function based on optimization space
        if self.training_config["finetune_space"] == "raw":
            loss_fn = renormalized_criterion
            targets = targets_raw
        elif self.training_config["finetune_space"] == "preprocessed":
            loss_fn = bar_distribution
            targets = targets_standardized
        else:
            raise ValueError(
                f"Invalid optimization space: {self.training_config['finetune_space']}"
            )

        nll_loss_per_sample, ignore_loss_mask = loss_fn(
            pred_logits,
            targets.to(self.device),
            ignore_negative_loss=self.training_config["ignore_negative_loss"],
            ignore_inf_loss=self.training_config["ignore_inf_loss"],
        )
        loss = nll_loss_per_sample[~ignore_loss_mask].mean()

        if torch.sum(ignore_loss_mask) > 0:
            num_ignored_points = torch.sum(ignore_loss_mask).item()
            logging.warning(
                f"Batch {batch_idx}, {prefix} step ignored {num_ignored_points} target points"
            )
            if prefix == "train":
                self.log("train/ignored_points", num_ignored_points)
                self.num_current_epoch_ignored_train_target_points += num_ignored_points
                self.num_total_ignored_train_target_points += num_ignored_points
            elif prefix == "val":
                self.log("val/ignored_points", num_ignored_points)
                self.num_current_epoch_ignored_val_target_points += num_ignored_points
                self.num_total_ignored_val_target_points += num_ignored_points

        if torch.isinf(loss).any():
            self.log(f"{prefix}/inf_loss_detected", 1.0)
            logging.warning(
                f"Epoch {self.current_epoch}, batch {batch_idx}, {prefix} step detected inf loss"
            )
            if prefix == "train":
                self.train_skipped_steps += 1
                self.current_epoch_train_skipped += 1
            else:
                self.val_skipped_steps += 1
                self.current_epoch_val_skipped += 1

        return loss

    @staticmethod
    def _calculate_regression_metrics(
        median_predictions: np.ndarray,
        y_test: np.ndarray,
    ) -> EvalResult:
        """Calculate standard regression metrics for model evaluation."""

        pred_median = median_predictions["median"]
        pred_mean = median_predictions["mean"]

        return EvalResult(
            mse=mse(y_test, pred_mean),
            r2=r2_score(y_test, pred_mean),
            mae=mean_absolute_error(y_test, pred_median),
            smape=smape(y_test, pred_median),
            nmae=nmae(y_test, pred_median),
        )

    def on_train_epoch_end(self):
        """Log training statistics at the end of each epoch."""
        # Access the epoch loss through the trainer's logged metrics
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        logging.info(f"Current epoch: {self.current_epoch}, train loss: {train_loss}")

        self.log(
            "train/total_ignored_target_points",
            self.num_total_ignored_train_target_points,
        )
        self.log(
            "train/epoch_ignored_target_points",
            self.num_current_epoch_ignored_train_target_points,
        )
        self.num_current_epoch_ignored_train_target_points = 0

        self.log("train/skipped_steps", self.train_skipped_steps)
        self.log("train/skipped_epochs", self.current_epoch_train_skipped)
        self.train_skipped_steps = 0
        self.current_epoch_train_skipped = 0

    def on_validation_epoch_start(self):
        """Prepare evaluation model at the start of each validation epoch."""
        self.eval_model = _prepare_eval_model(
            original_model=self.regressor,
            eval_init_args=self.tabpfn_model_config,
            model_class=TabPFNRegressor,
        )

    def on_validation_epoch_end(self):
        """Log validation statistics at the end of each epoch."""
        self.log(
            "val/total_ignored_target_points", self.num_total_ignored_val_target_points
        )
        self.log(
            "val/epoch_ignored_target_points",
            self.num_current_epoch_ignored_val_target_points,
        )
        self.num_current_epoch_ignored_val_target_points = 0

        self.log("val/skipped_steps", self.val_skipped_steps)
        self.log("val/skipped_epochs", self.current_epoch_val_skipped)
        self.val_skipped_steps = 0
        self.current_epoch_val_skipped = 0

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.Adam(
            self.regressor.model_.parameters(), lr=self.training_config["lr"]
        )

    @staticmethod
    def _save_debug_data(
        batch_idx, x_train_raw, y_train_raw, x_test_raw, y_test_raw, prefix
    ):
        """Save debug data to CSV files."""
        import pandas as pd

        # Create debug directory if it doesn't exist
        debug_dir = os.path.join("debug_bad_data_new", prefix, f"batch_{batch_idx}")
        os.makedirs(debug_dir, exist_ok=True)

        # Save raw data to CSV files
        pd.DataFrame(x_train_raw.cpu()).to_csv(
            os.path.join(debug_dir, "x_train_raw.csv")
        )
        pd.DataFrame(y_train_raw.cpu()).to_csv(
            os.path.join(debug_dir, "y_train_raw.csv")
        )
        pd.DataFrame(x_test_raw.cpu()).to_csv(os.path.join(debug_dir, "x_test_raw.csv"))
        pd.DataFrame(y_test_raw.cpu()).to_csv(os.path.join(debug_dir, "y_test_raw.csv"))

        logging.info(f"Saved debug data for batch {batch_idx} to {debug_dir}")

    @staticmethod
    def _parse_model_config(raw_model_config: dict) -> dict:
        new_model_config = raw_model_config.copy()
        new_model_config["inference_precision"] = PRECISION_MAP[
            raw_model_config["inference_precision"]
        ]
        return new_model_config

    def _compute_param_hash(self):
        """Compute a hash of all model parameters."""
        param_tensors = []
        for _, param in self.regressor.model_.named_parameters():
            param_tensors.append(param.detach().cpu().numpy().tobytes())

        import hashlib

        combined = b"".join(param_tensors)
        return hashlib.md5(combined).hexdigest()

    def _report_batch_hash(self, batch_data):
        """Compute a hash of preprocessed training and testing data to track if the same data is being processed across epochs."""
        import hashlib

        if batch_data is None:
            return "None"

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

        # Hash X_trains_preprocessed (list of tensors)
        x_train_hash = []
        for i, x_train in enumerate(X_trains_preprocessed):
            x_train_hash.append(x_train.detach().cpu().numpy().tobytes())
            # logging.info(
            #     f"Hash for x_train[{i}]: {hashlib.md5(x_train_bytes).hexdigest()}"
            # )

        # Hash y_trains_preprocessed (list of tensors)
        y_train_hash = []
        for y_train in y_trains_preprocessed:
            y_train_hash.append(y_train.detach().cpu().numpy().tobytes())

        # Hash X_tests_preprocessed
        x_test_hash = []
        for i, x_test in enumerate(X_tests_preprocessed):
            x_test_hash.append(x_test.detach().cpu().numpy().tobytes())
            # logging.info(
            #     f"Hash for x_test[{i}]: {hashlib.md5(x_test.detach().cpu().numpy().tobytes()).hexdigest()}"
            # )

        # Hash y_test_standardized
        y_test_hash = []
        for y_test in y_test_standardized:
            y_test_hash.append(y_test.detach().cpu().numpy().tobytes())

        # Hash x_train_raw
        x_train_raw_hash = x_train_raw.detach().cpu().numpy().tobytes()
        logging.info(
            f"Hash for x_train_raw: {hashlib.md5(x_train_raw_hash).hexdigest()}"
        )

        # Hash x_test_raw
        x_test_raw_hash = x_test_raw.detach().cpu().numpy().tobytes()
        logging.info(f"Hash for x_test_raw: {hashlib.md5(x_test_raw_hash).hexdigest()}")

        # Hash y_train_raw
        y_train_raw_hash = y_train_raw.detach().cpu().numpy().tobytes()
        logging.info(
            f"Hash for y_train_raw: {hashlib.md5(y_train_raw_hash).hexdigest()}"
        )

        # Hash y_test_raw
        y_test_raw_hash = y_test_raw.detach().cpu().numpy().tobytes()
        logging.info(f"Hash for y_test_raw: {hashlib.md5(y_test_raw_hash).hexdigest()}")

        # Hash cat_ixs (could be None or a list)
        cat_ixs_hash = []
        if cat_ixs is not None:
            if isinstance(cat_ixs, list):
                for cat_ix in cat_ixs:
                    if cat_ix is not None:
                        cat_ixs_hash.append(str(cat_ix).encode())
            else:
                cat_ixs_hash.append(str(cat_ixs).encode())

        # Hash confs (ensemble configurations)
        from tabpfn.preprocessing import RegressorEnsembleConfig

        confs_hash = []
        if confs is not None:
            if isinstance(confs, list):
                for conf in confs:
                    if isinstance(conf, RegressorEnsembleConfig):
                        confs_hash.append(str(conf.__dict__).encode())
                    else:
                        confs_hash.append(str(conf).encode())
            else:
                confs_hash.append(str(confs).encode())

        for name, data in [
            ("x_train", x_train_hash),
            ("y_train", y_train_hash),
            ("x_test", x_test_hash),
            ("y_test", y_test_hash),
            ("cat_ixs", cat_ixs_hash),
            ("confs", confs_hash),
        ]:
            logging.info(f"Hash for {name}: {hashlib.md5(b''.join(data)).hexdigest()}")
