"""
Mini fine-tuning script for TabPFN Time Series models using PyTorch Lightning with Weights & Biases logging.
"""

import os
import logging
import argparse
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
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
    load_all_ts_datasets
)
from tabpfn_time_series.experimental.finetuning.ft_config import FinetuneConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def ts_splitfn(
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction_length = 100
    X_train = X[:-prediction_length]
    X_test = X[-prediction_length:]
    y_train = y[:-prediction_length]
    y_test = y[-prediction_length:]

    return X_train, X_test, y_train, y_test


@dataclass
class EvalResult:
    mse: float
    mae: float
    r2: float


class TabPFNTimeSeriesModule(pl.LightningModule):
    def __init__(self, regressor: TabPFNRegressor, opt_config: Any):
        super().__init__()
        self.regressor = regressor
        self.opt_config = opt_config
        self.save_hyperparameters(ignore=["regressor"])
        self.train_skipped_steps = 0
        self.val_skipped_steps = 0
        self.current_epoch_train_skipped = 0
        self.current_epoch_val_skipped = 0
        
    def forward(self, X_tests_preprocessed):
        return self.regressor.forward(X_tests_preprocessed)
    
    def get_loss_fn(self):
        if self.opt_config.space == "raw_label_space":
            return self.regressor.bardist_.to(self.device)    
        elif self.opt_config.space == "preprocessed":
            return self.regressor.renormalized_criterion_.to(self.device)
        else: 
            raise ValueError("Need to define optimization space")
    
    def training_step(self, batch, batch_idx):
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            renormalized_criterion, 
            batch_x_test_raw,
            batch_y_test_raw
        ) = batch
        
        # Quick hack to fix different dtypes
        batch_y_test_raw = batch_y_test_raw.to(torch.float32)
        
        # Forward pass
        self.regressor.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
        averaged_pred_logits, _, _ = self.forward(X_tests_preprocessed)
        
        # Check for NaN/Inf values
        if torch.isnan(averaged_pred_logits).any():
            self.log("train/nan_detected", 1.0)
            
        if torch.isinf(averaged_pred_logits).any():
            self.log("train/inf_detected", 1.0)
        
        # Calculate loss
        loss_fn = self.get_loss_fn()
        nll_loss_per_sample = loss_fn(averaged_pred_logits, batch_y_test_raw.to(self.device))
        loss = nll_loss_per_sample.mean()
        
        # Skip infinite loss values
        if torch.isinf(loss).any():
            self.log("train/inf_loss_detected", 1.0)
            self.train_skipped_steps += 1
            self.current_epoch_train_skipped += 1
            return None
        
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_standardized,
            cat_ixs,
            confs,
            renormalized_criterion, 
            batch_x_test_raw,
            batch_y_test_raw
        ) = batch
        
        # Quick hack to fix different dtypes
        batch_y_test_raw = batch_y_test_raw.to(torch.float32)
        
        # Forward pass
        self.regressor.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
        averaged_pred_logits, _, _ = self.forward(X_tests_preprocessed)
        
        # Calculate loss
        loss_fn = self.get_loss_fn()
        nll_loss_per_sample = loss_fn(averaged_pred_logits, batch_y_test_raw.to(self.device))
        loss = nll_loss_per_sample.mean()
        
        # Skip infinite loss values
        if torch.isinf(loss).any():
            self.log("val/inf_loss_detected", 1.0)
            self.val_skipped_steps += 1
            self.current_epoch_val_skipped += 1
            return None
        
        self.log("val/loss", loss, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log("train/skipped_steps_epoch", self.current_epoch_train_skipped)
        self.log("train/total_skipped_steps", self.train_skipped_steps)
        logging.info(f"Epoch {self.current_epoch}: Skipped {self.current_epoch_train_skipped} training steps")
        self.current_epoch_train_skipped = 0
    
    def on_validation_epoch_end(self):
        self.log("val/skipped_steps_epoch", self.current_epoch_val_skipped)
        self.log("val/total_skipped_steps", self.val_skipped_steps)
        logging.info(f"Epoch {self.current_epoch}: Skipped {self.current_epoch_val_skipped} validation steps")
        self.current_epoch_val_skipped = 0
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.regressor.model_.parameters(), 
            lr=self.opt_config.lr
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for TabPFN Time Series models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu)")
    parser.add_argument("--wandb_project", type=str, default="tabpfn-ts-ft", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--max_epochs", type=int, default=None, help="Maximum number of epochs to train")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for the W&B run")
    return parser.parse_args()


def main():
    load_dotenv()
    HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
    
    args = parse_args()
    device = args.device

    # Load hyperparams
    all_config = FinetuneConfig()
    opt_config = all_config.Optimization
    
    # Override epochs if specified in command line
    if args.max_epochs is not None:
        opt_config.n_epochs = args.max_epochs

    train_dataset_config = all_config.TrainDataset
    train_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=train_dataset_config.dataset_repo_name,
        dataset_names=train_dataset_config.dataset_names,
        max_context_length=train_dataset_config.max_context_length,
        hf_cache_dir=HF_CACHE_DIR,
    )

    test_dataset_config = all_config.TestDataset
    test_dataset = TabPFNTimeSeriesPretrainDataset(
        dataset_repo_name=test_dataset_config.dataset_repo_name,
        dataset_names=test_dataset_config.dataset_names,
        max_context_length=test_dataset_config.max_context_length,
        hf_cache_dir=HF_CACHE_DIR,
    )

    # Load all time series datasets
    all_train_X, all_train_y = load_all_ts_datasets(train_dataset)
    all_test_X, all_test_y = load_all_ts_datasets(test_dataset, shuffle=False)

    logging.info(f"Loaded {len(all_train_X)} training samples and {len(all_test_X)} test samples")

    # Initialize TabPFN regressor
    model_config = all_config.Model
    regressor_args = dict(
        ignore_pretraining_limits=model_config.ignore_pretraining_limits,
        n_estimators=model_config.n_estimators,
        random_state=model_config.random_state,
        device=device,
        differentiable_input=model_config.differentiable_input,
        inference_precision=torch.float32,
    )
        
    reg = TabPFNRegressor(**regressor_args)
    splitfn = ts_splitfn

    # Preprocess datasets
    train_datasets_collection = reg.get_preprocessed_datasets(all_train_X, all_train_y, splitfn, max_data_size=10000)
    test_datasets_collection = reg.get_preprocessed_datasets(all_test_X, all_test_y, splitfn, max_data_size=10000)
    
    # Create data loaders
    train_dl = DataLoader(
        train_datasets_collection, 
        batch_size=1, 
        collate_fn=collate_for_tabpfn_dataset,
        shuffle=True,
        num_workers=0,
    )
    
    val_dl = DataLoader(
        test_datasets_collection, 
        batch_size=1, 
        collate_fn=collate_for_tabpfn_dataset,
        num_workers=0,
    )
    
    # Initialize PyTorch Lightning module
    model = TabPFNTimeSeriesModule(reg, opt_config)
    
    # Setup Weights & Biases logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        log_model=True,
    )
    
    # Log hyperparameters
    wandb_logger.log_hyperparams(all_config.to_dict())
    
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
        patience=5,
        mode="min",
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=opt_config.n_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        gradient_clip_val=1.0,
        accumulate_grad_batches=opt_config.gradient_accumulation_steps,
        log_every_n_steps=10,
    )
    
    # Train the model
    trainer.fit(model, train_dl, val_dl)
    
    # Log best model path
    if checkpoint_callback.best_model_path:
        logging.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
        wandb_logger.experiment.log({"best_model_path": checkpoint_callback.best_model_path})
    
    # Log final skipped steps statistics
    logging.info(f"Total training steps skipped: {model.train_skipped_steps}")
    logging.info(f"Total validation steps skipped: {model.val_skipped_steps}")
    
    # Close wandb run
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
