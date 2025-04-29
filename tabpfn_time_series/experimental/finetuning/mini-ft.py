"""
Mini fine-tuning script for TabPFN Time Series models.
"""

import os
import logging
import argparse
from tqdm import tqdm
from functools import partial
from typing import Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from tabpfn.utils import collate_for_tabpfn_dataset
from tabpfn.finetune_utils import _prepare_eval_model
from tabpfn_time_series.experimental.finetuning.dataset import (
    TabPFNTimeSeriesPretrainDataset,
    TimeSeriesPretrainConfig,
    load_all_ts_datasets
)
from tabpfn_time_series.experimental.finetuning.ft_config import FinetuneConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Dataset configuration
HF_DATASET_REPO_NAME = "liamsbhoo/GiftEvalPretrain"


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


def eval_model(
    model: TabPFNRegressor,
    eval_model_init_args: dict,
    *,
    X_train_raw: list[np.ndarray],
    y_train_raw: list[np.ndarray],
    X_test_raw: list[np.ndarray],
    y_test_raw: list[np.ndarray],
) -> EvalResult:
    eval_model = _prepare_eval_model(model, eval_model_init_args, TabPFNRegressor)

    # Get a random subset to evaluate on
    n_use_ratio = 0.1
    n_use = int(n_use_ratio * len(X_train_raw))

    # Get a random subset of indices
    random_indices = np.random.choice(len(X_train_raw), size=n_use, replace=False)

    all_res = {}

    for set_name, (set_X, set_y) in \
        [("in-sample", (X_train_raw, y_train_raw)),
        ("out-of-sample", (X_test_raw, y_test_raw))]:
            
        for i in random_indices:
            single_X = set_X[i]
            single_y = set_y[i]

            eval_model.fit(single_X, single_y)
            predictions = eval_model.predict(single_X)

            try:
                res = EvalResult(
                    mse=mean_squared_error(set_y, predictions),
                    mae=mean_absolute_error(set_y, predictions),
                    r2=r2_score(set_y, predictions),
                )
            except Exception as e:
                logging.warning(f"Error during evaluation prediction/metric calculation: {e}")
                res = EvalResult(
                    mse=np.nan,
                    mae=np.nan,
                    r2=np.nan,
                )

            all_res[set_name] = res

    return all_res


def eval_model_on_train(
    model: TabPFNRegressor,
    eval_model_init_args: dict,
    all_X: list[np.ndarray],
    all_y: list[np.ndarray],
) -> EvalResult:

    all_X_train, all_X_test, all_y_train, all_y_test = ts_splitfn(all_X, all_y)

    return eval_model(
        model,
        eval_model_init_args,
        X_train_raw=all_X_train,
        y_train_raw=all_y_train,
        X_test_raw=all_X_test,
        y_test_raw=all_y_test,
    )


def get_loss_fn(opt_config, model: TabPFNRegressor, device: str):
    if opt_config.space == "raw_label_space":
        return model.bardist_.to(device)    
    elif opt_config.space == "preprocessed":
        return model.renormalized_criterion_.to(device)
    else: 
        raise ValueError("Need to define optimization space")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script for TabPFN Time Series models")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    load_dotenv()
    HF_CACHE_DIR = os.getenv("HF_CACHE_DIR")
    
    args = parse_args()
    device = args.device

    # Load hyperparams
    all_config = FinetuneConfig()
    opt_config = all_config.Optimization

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

    # # Debug
    # # Overwrite all_y with random values
    # all_y = [np.random.randn(len(y)) for y in all_y]

    logging.info(f"Loaded {len(all_train_X)} samples")

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

    # # Initialial evaluation on train set
    # init_train_eval_res = eval_model_on_train(reg, regressor_args, all_X, all_y)
    # logging.info(f"Initial evaluation on train set: {init_train_eval_res}")
    splitfn = ts_splitfn

    # Preprocess datasets
    train_datasets_collection = reg.get_preprocessed_datasets(all_train_X, all_train_y, splitfn, max_data_size=10000)
    test_datasets_collection = reg.get_preprocessed_datasets(all_test_X, all_test_y, splitfn, max_data_size=10000)
    
    # Setup optimization components
    optimizer = Adam(reg.model_.parameters(), lr=opt_config.lr)

    # Training loop
    train_dl = DataLoader(train_datasets_collection, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    test_dl = DataLoader(test_datasets_collection, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    
    for epoch in range(opt_config.n_epochs):
        running_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
        
        for iter, data_batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{opt_config.n_epochs}")):
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
            ) = data_batch  # shape: (batch_size, num_estimators, num_rows, num_cols)

            # Quick hack to fix different dtypes
            batch_y_test_raw = batch_y_test_raw.to(torch.float32)

            # # Debug
            # # assert no NaN or Inf
            # for i in range(len(X_trains_preprocessed)):
            #     assert not torch.isnan(X_trains_preprocessed[i]).any(), "NaN found in X_trains_preprocessed"
            #     assert not torch.isinf(X_trains_preprocessed[i]).any(), "Inf found in X_trains_preprocessed"
            #     assert not torch.isnan(X_tests_preprocessed[i]).any(), "NaN found in X_tests_preprocessed"
            #     assert not torch.isinf(X_tests_preprocessed[i]).any(), "Inf found in X_tests_preprocessed"
            #     assert not torch.isnan(y_trains_preprocessed[i]).any(), "NaN found in y_trains_preprocessed"
            #     assert not torch.isinf(y_trains_preprocessed[i]).any(), "Inf found in y_trains_preprocessed"
            # assert not torch.isnan(batch_y_test_raw).any()

            # Forward pass
            reg.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
            averaged_pred_logits, _, _ = reg.forward(X_tests_preprocessed)  # [BatchSize, N_test, NumBars]

            # Debug
            if torch.isnan(averaged_pred_logits).any():
                logging.warning("NaN values found in predictions")
                
            if torch.isinf(averaged_pred_logits).any():
                logging.warning("Infinite values found in predictions")

            # Calculate loss
            loss_fn = get_loss_fn(opt_config, reg, device)
            nll_loss_per_sample = loss_fn(averaged_pred_logits, batch_y_test_raw.to(device))
            loss = nll_loss_per_sample.mean()

            # TODO: Investigate and fix
            # Temporary hack to workaround occasional infinite loss values
            #   skipping this step for now
            if torch.isinf(loss).any():
                logging.warning("Infinite loss value encountered")
                continue
            
            # Normalize loss by accumulation steps
            loss = loss / opt_config.gradient_accumulation_steps
            running_loss += loss.item()
            
            # Backward pass with gradient accumulation
            loss.backward()
            
            # Step the optimizer every accumulation_steps iterations
            if (iter + 1) % opt_config.gradient_accumulation_steps == 0 or (iter + 1) == len(train_dl):
                optimizer.step()
                optimizer.zero_grad()
                print(f" Loss in EPOCH {epoch+1}, iter: {iter},"
                      f" accumulated loss: {running_loss * opt_config.gradient_accumulation_steps}")
                running_loss = 0.0

        # End of epoch evaluation
        logging.info("Evaluating on test set")
        test_loss = []
        inf_loss_count = 0
        with torch.no_grad():
            for test_data_batch in tqdm(test_dl, desc=f"Epoch {epoch+1}/{opt_config.n_epochs}"):
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
                ) = test_data_batch

                # Quick hack to fix different dtypes
                batch_y_test_raw = batch_y_test_raw.to(torch.float32)

                reg.fit_from_preprocessed(X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs)
                averaged_pred_logits, _, _ = reg.forward(X_tests_preprocessed)  # [BatchSize, N_test, NumBars]

                # Calculate loss
                loss_fn = get_loss_fn(opt_config, reg, device)
                nll_loss_per_sample = loss_fn(averaged_pred_logits, batch_y_test_raw.to(device))
                loss = nll_loss_per_sample.mean()

                if torch.isinf(loss).any():
                    logging.warning("Infinite loss value encountered")
                    inf_loss_count += 1
                    continue

                test_loss.append(loss.item())

        logging.info(f"Test loss: {np.mean(test_loss)}, "
                     f"infinite loss count: {inf_loss_count}")



if __name__ == "__main__":
    main()
