import os
import time
from typing import Tuple, TypeAlias
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import logging

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


@dataclass
class TimeSeriesPretrainConfig:
    feature_config: dict = field(default_factory=lambda: DEFAULT_FEATURE_CONFIG)
    max_context_length: int = 4096
    prediction_length: int = 7


class TabPFNTimeSeriesPretrainDataset(Dataset):
    def __init__(
        self,
        dataset_repo_name: str = "liamsbhoo/GiftEvalPretrainMini",
        dataset_names: list[str] = None,
        time_series_pretrain_config: TimeSeriesPretrainConfig = TimeSeriesPretrainConfig(),
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
                        dataset_repo_name, name, split="train",
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
        logger.info(f"Time taken to load data: {int(load_data_end_time - load_data_start_time)} seconds")

        self.ts_preprocessor = TimeSeriesPreprocessor(
            max_context_length=self.max_context_length,
        )
        self.feature_transformer = self._create_feature_transformer(
            self.feature_config
        )

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
    """Efficiently collate batches by using zip to unpack in one pass"""
    return zip(*batch)


def load_all_ts_datasets(
    dataset: TabPFNTimeSeriesPretrainDataset,
    shuffle: bool = False,
    loading_batch_size: int = 128,
    loading_num_workers: int = 1,
) -> Tuple[list[XType], list[YType]]:
    """Load all time series datasets efficiently with minimal memory copying"""
    dataloader = DataLoader(
        dataset,
        batch_size=loading_batch_size,
        shuffle=shuffle,
        collate_fn=efficient_collate_fn,
        pin_memory=True,
        num_workers=loading_num_workers,
    )

    # Pre-allocate a single list and extend it once per batch
    all_X, all_y = [], []
    for X, y in dataloader:
        all_X.extend(X)
        all_y.extend(y)

    return all_X, all_y
