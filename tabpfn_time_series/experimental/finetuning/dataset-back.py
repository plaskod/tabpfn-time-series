import os
import time
from typing import Tuple, TypeAlias, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.experimental.features import FeatureTransformer
from tabpfn_time_series.experimental.pipeline.pipeline import FEATURE_MAP
from tabpfn_time_series.experimental.pipeline.time_series_preprocessing import (
    TimeSeriesPreprocessor,
)

logger = logging.getLogger(__name__)


XType: TypeAlias = pd.DataFrame
YType: TypeAlias = pd.Series
XTrainType: TypeAlias = XType
YTrainType: TypeAlias = YType
XTestType: TypeAlias = XType
YTestType: TypeAlias = YType


DEFAULT_FEATURE_CONFIG = {
    "RunningIndexFeature": {},
    "AdditionalCalendarFeature": {
        "additional_seasonal_features": {
            "second_of_minute": [60],
            "minute_of_hour": [60],
        }
    },
    "AutoSeasonalFeature": {
        "config": {"max_top_k": 5, "detrend_type": "linear", "zero_padding_factor": 2}
    },
}


class TabPFNTimeSeriesPretrainDataset(Dataset):
    def __init__(
        self,
        dataset_repo_name: str = "liamsbhoo/GiftEvalPretrainMini",
        dataset_names: list[str] = None,
        max_context_length: int = 4096,
        feature_config: dict = DEFAULT_FEATURE_CONFIG,
        hf_cache_dir: str = None,
    ):
        self.dataset_repo_name = dataset_repo_name
        self.dataset_names = dataset_names
        self.max_context_length = max_context_length
        self.feature_config = feature_config

        load_data_start_time = time.time()
        # If no dataset names provided, load the entire repository
        if self.dataset_names is None:
            logger.info(f"Loading all datasets from {dataset_repo_name}")
            self.datasets = [
                load_dataset(dataset_repo_name, split="train", cache_dir=hf_cache_dir)
            ]
            logger.info(f"Loaded {len(self.datasets)} samples")
        else:
            logger.info(f"Loading datasets: {dataset_names} from {dataset_repo_name}")

            # Validate dataset names
            # assert_valid_dataset_names(dataset_repo_name, dataset_names)

            # Load multiple datasets by name
            self.datasets = []
            for name in self.dataset_names:
                self.datasets.append(
                    load_dataset(
                        dataset_repo_name,
                        name,
                        split="train",
                        cache_dir=hf_cache_dir,
                    )
                )

            logger.info(f"Loaded {len(self.datasets)} datasets")
            for i, dataset in enumerate(self.datasets):
                dataset_name = (
                    self.dataset_names[i] if self.dataset_names else f"dataset_{i}"
                )
                logger.info(f"  - {dataset_name}: {len(dataset)} samples")
        load_data_end_time = time.time()
        logger.info(
            f"Time taken to load data: {int(load_data_end_time - load_data_start_time)} seconds"
        )

        self.ts_preprocessor = TimeSeriesPreprocessor(
            max_context_length=self.max_context_length,
        )
        self.feature_transformer = self._create_feature_transformer(self.feature_config)

        # Calculate total length across all datasets
        self.dataset_lengths = [len(ds) for ds in self.datasets]
        self.dataset_cumulative_lengths = [0]
        for length in self.dataset_lengths:
            self.dataset_cumulative_lengths.append(
                self.dataset_cumulative_lengths[-1] + length
            )

    def __len__(self):
        # Total length is the sum of all dataset lengths
        return self.dataset_cumulative_lengths[-1]

    def __getitem__(self, index: int) -> Tuple[XType, YType]:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} is out of bounds for dataset of size {len(self)}"
            )

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_length in enumerate(self.dataset_cumulative_lengths[1:], 1):
            if index < cum_length:
                dataset_idx = i - 1
                break

        # Calculate the local index within the dataset
        local_index = index - self.dataset_cumulative_lengths[dataset_idx]

        # Get the sample from the appropriate dataset
        sample = self.datasets[dataset_idx][local_index]

        if not self.is_univariate(sample["target"]):
            # TODO: support multivariate time series
            raise ValueError(
                f"Target is not univariate for dataset: {self.dataset_names[dataset_idx]}"
            )

        X, y = self.time_series_to_feat_tabular_dataset(
            start_timestamp=sample["start"],
            freq=sample["freq"],
            target=sample["target"],
        )

        return X, y

    def time_series_to_feat_tabular_dataset(
        self,
        start_timestamp: datetime,
        freq: str,
        target: list[float],
    ) -> Tuple[XType, YType]:
        """
        Convert a raw time series into train/test tabular datasets with features.

        Args:
            start_timestamp: Starting timestamp of the time series
            freq: Frequency of the time series (e.g., 'D' for daily)
            target: List of target values

        Returns:
            X: Feature matrix
            y: Target values
        """
        # Create timestamp index for the time series
        timestamp = pd.date_range(start=start_timestamp, periods=len(target), freq=freq)

        # Create TimeSeriesDataFrame and preprocess it
        tsdf = TimeSeriesDataFrame(
            pd.DataFrame(
                {"target": target, "timestamp": timestamp, "dummy_item_id": 0}
            ),
            timestamp_column="timestamp",
            id_column="dummy_item_id",
        )
        preprocessed_tsdf = self.ts_preprocessor.forward(tsdf)
        feat_tsdf, _ = self.feature_transformer.transform(preprocessed_tsdf)

        X = feat_tsdf.to_data_frame().drop(columns=["target"])
        y = feat_tsdf.to_data_frame()["target"]

        return X, y

    @staticmethod
    def _convert_tsdf_to_tabular(
        tsdf: TimeSeriesDataFrame, timestamp_column: str = "timestamp"
    ) -> pd.DataFrame:
        return (
            tsdf.to_data_frame()
            .reset_index()
            .drop(columns=["item_id"])
            .set_index(timestamp_column)
        )

    @staticmethod
    def _create_feature_transformer(feature_config: dict) -> FeatureTransformer:
        features = []
        for feature_name, config in feature_config.items():
            if feature_name in FEATURE_MAP:
                features.append(FEATURE_MAP[feature_name](**config))
            else:
                raise ValueError(f"Feature {feature_name} not found in FEATURE_MAP")
        return FeatureTransformer(features)

    @staticmethod
    def is_univariate(target: list[float] | list[list[float]] | np.ndarray) -> bool:
        return np.array(target).ndim == 1


def efficient_collate_fn(
    batch: list[Tuple[XType, YType]],
) -> Tuple[list[XType], list[YType]]:
    """
    Efficiently collate batches of (X, y) pairs.

    Args:
        batch: List of (X, y) tuples where X is a DataFrame and y is a Series

    Returns:
        Tuple of (X_list, y_list) where X_list contains all DataFrames and
        y_list contains all Series from the batch
    """
    # Unpack the batch of (X, y) tuples into separate lists of X and y
    X_list, y_list = [], []
    for X, y in batch:
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list


def load_all_ts_datasets(
    dataset: TabPFNTimeSeriesPretrainDataset,
    shuffle: bool = False,
    max_length: int = None,
    preprocess_fn: Callable[[XType, YType], Tuple[XType, YType]] = None,
    prefix: str = None,
) -> Tuple[list[XType], list[YType]]:
    """
    Load all time series datasets efficiently with minimal memory copying

    Args:
        dataset: The dataset to load
        shuffle: Whether to shuffle the dataset
        max_length: Maximum number of samples to load (None for all)
        preprocess_fn: Optional callable to preprocess/filter individual samples
                      Should take (X, y) and return (X, y) for each sample

    Returns:
        Tuple of (all_X, all_y) where all_X is a list of DataFrames and
        all_y is a list of Series
    """
    # Automatically set num_workers to the number of available CPU cores
    num_workers = max(os.cpu_count() - 1, 1)
    batch_size = 128
    prefetch_factor = 4

    # Adjust batch size and prefetch_factor if max_length is specified
    if max_length is not None:
        batch_size = min(128, max_length)
        prefetch_factor = 4 if max_length > 1000 else None

    # Pre-allocate lists with known capacity to avoid resizing
    all_X, all_y = [], []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=efficient_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    # Process each batch
    n_samples_loaded = 0
    n_samples_skipped = 0
    for batch_X, batch_y in tqdm(dataloader, desc="Loading time series datasets"):
        # Apply preprocessing to each sample in the batch if provided
        if preprocess_fn is not None:
            logger.debug(f"Applying preprocess_fn to {len(batch_X)} samples")
            # Process in a single list comprehension to avoid intermediate lists
            valid_pairs = [
                result
                for X, y in zip(batch_X, batch_y)
                if (result := preprocess_fn(X, y)) is not None
            ]
            n_samples_skipped += len(batch_X) - len(valid_pairs)
            logger.debug(f"Left with {len(valid_pairs)} samples")

            # Only unzip if we have valid samples
            if valid_pairs:
                batch_X, batch_y = zip(*valid_pairs)
            else:
                batch_X, batch_y = [], []

        # Check if we've reached max_length before extending
        if max_length is not None:
            remaining = max_length - n_samples_loaded
            if remaining <= 0:
                break

            # Only take what we need to reach max_length
            if len(batch_X) > remaining:
                batch_X = batch_X[:remaining]
                batch_y = batch_y[:remaining]

        # Add samples and update counter
        all_X.extend(batch_X)
        all_y.extend(batch_y)
        n_samples_loaded += len(batch_X)

    logger.info(f"Loaded {len(all_X)} total samples, skipped {n_samples_skipped}")

    # Optional debugging: Save dataset to CSV files if debug flag is set
    if os.environ.get("DEBUG_SAVE_RAW_DATA"):
        save_data_into_csvs(all_X, all_y, prefix=prefix)

    return all_X, all_y


def save_data_into_csvs(all_X, all_y, prefix: str = None):
    """
    Save dataset to CSV files for debugging purposes.

    Args:
        all_X: List of feature DataFrames
        all_y: List of target Series
    """
    import pandas as pd

    try:
        all_X_df, all_y_df = [], []
        for i, (X, y) in enumerate(zip(all_X, all_y)):
            # Create copies to avoid modifying originals
            X_copy = X.copy() if hasattr(X, "copy") else X
            y_copy = y.copy() if hasattr(y, "copy") else y

            X_copy["index"] = i
            y_copy = pd.DataFrame(y_copy)
            y_copy["index"] = i

            all_X_df.append(X_copy)
            all_y_df.append(y_copy)

        pd.concat(all_X_df).to_csv(f"{prefix}_debug_all_X.csv", index=False)
        logger.debug(f"Saved features to {prefix}_debug_all_X.csv")

        pd.concat(all_y_df).to_csv(f"{prefix}_debug_all_y.csv", index=False)
        logger.debug(f"Saved targets to {prefix}_debug_all_y.csv")

    except Exception as e:
        logger.warning(f"Failed to save debug CSV files: {e}")


def filter_constant_series(
    X: XType,
    y: YType,
    threshold: float = 1e-8,
) -> Optional[Tuple[XType, YType]]:
    """
    Filter out time series with nearly constant y values.

    Args:
        X: Feature DataFrame
        y: Target Series

    Returns:
        Tuple of (X, y) if the series passes the filters, None otherwise
    """
    # Skip if standard deviation is too small (nearly constant)
    if y.std() < threshold:
        logger.debug(f"Skipping constant series with std: {y.std()}")
        return None

    # Series passed all filters
    return X, y
