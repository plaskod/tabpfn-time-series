import os
import time
import json
import pickle
import hashlib
import random
import logging
from typing import Tuple, TypeAlias, Optional, Callable, List, Dict, Any, Union
from datetime import datetime
import multiprocessing
import pandas as pd
import numpy as np
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
CACHE_KEY_TYPE: TypeAlias = str
CACHE_PATH_TYPE: TypeAlias = str
METADATA_TYPE: TypeAlias = Dict[CACHE_KEY_TYPE, CACHE_PATH_TYPE]

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


def try_read_pickle(
    path: str,
    fallback: Optional[Any] = None,
) -> Any:
    """
    Try to read a pickle file.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Failed to load pickle file {path}: {e}. Returning {fallback}.")
        return fallback


def try_write_pickle(path: str, data: Any, protocol: Optional[int] = None):
    """
    Try to write a pickle file.
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    except Exception as e:
        logger.warning(f"Failed to write pickle file {path}: {e}")
        raise e


class TimeSeriesDataManager:
    """
    Manager for time series datasets with caching and feature extraction.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        max_context_length: int = 4096,
        feature_config: Dict = DEFAULT_FEATURE_CONFIG,
    ):
        """
        Initialize the TimeSeriesDataManager.

        Args:
            cache_dir: Directory to store cached datasets
            hf_cache_dir: Directory for Hugging Face dataset cache
            max_context_length: Maximum context length for time series preprocessing
            feature_config: Configuration for feature extraction
        """

        self.cache_dir = cache_dir
        self.hf_cache_dir = hf_cache_dir
        self.max_context_length = max_context_length
        self.feature_config = feature_config

        # Caching
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # Create subdirectories for different cache levels
            for subdir in ["featurized", "metadata"]:
                os.makedirs(os.path.join(cache_dir, subdir), exist_ok=True)

        self._metadata: METADATA_TYPE = self._load_metadata()

        # Generate a hash for the feature configuration
        self.feature_hash = self._hash_config(feature_config)
        logger.debug(f"Feature configuration hash: {self.feature_hash}")

    def _get_metadata_path(self) -> str:
        """Get the path to the cache metadata file."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, "metadata", "cache_metadata.pkl")

    @staticmethod
    def _get_cache_key(
        dataset_repo_name: str,
        dataset_name: str,
        feature_hash: str,
        preprocess_fn: Optional[
            Callable[[XType, YType], Optional[Tuple[XType, YType]]]
        ] = None,
        max_samples_per_dataset: Optional[int] = None,
    ) -> CACHE_KEY_TYPE:
        """Get the cache key for a dataset."""
        name = (
            f"{dataset_repo_name.replace('/', '_')}_{dataset_name}_feat{feature_hash}"
        )
        if preprocess_fn is not None:
            name += f"_preprocess{preprocess_fn.__name__}"
        if max_samples_per_dataset is not None:
            name += f"_max{max_samples_per_dataset}"
        return name

    def _get_dataset_cache_path(self, cache_key: CACHE_KEY_TYPE) -> str:
        """Get the path to the dataset cache file."""
        return os.path.join(self.cache_dir, "featurized", f"{cache_key}.pkl")

    def _load_metadata(self) -> METADATA_TYPE:
        """Load the cache metadata from disk if it exists."""
        metadata_path = self._get_metadata_path()
        if metadata_path and os.path.exists(metadata_path):
            res = try_read_pickle(metadata_path, {})
            logger.info(f"Loaded cache metadata with {len(res)} entries")
            logger.debug(f"Cache metadata: {res}")
        else:
            logger.info("No cache metadata found. Creating new metadata.")
            res = {}

        return res

    def _save_metadata(self, metadata: METADATA_TYPE):
        """Save the cache metadata to disk."""
        metadata_path = self._get_metadata_path()
        try_write_pickle(metadata_path, metadata, protocol=4)
        logger.info(f"Saved cache metadata with {len(metadata)} entries")

    def _update_metadata(self, cache_key: CACHE_KEY_TYPE, cache_path: CACHE_PATH_TYPE):
        """Update the cache metadata."""
        self._metadata[cache_key] = cache_path
        self._save_metadata(self._metadata)

    def _remove_from_metadata(self, cache_key: CACHE_KEY_TYPE):
        """Remove a cache key from the metadata."""
        del self._metadata[cache_key]
        self._save_metadata(self._metadata)

    def _load_cache(self, cache_key: CACHE_KEY_TYPE) -> Tuple[List[XType], List[YType]]:
        """Load a featurized dataset from cache."""
        cache_path = self._metadata[cache_key]
        data = try_read_pickle(cache_path, fallback=None)
        if data is None:
            logger.warning(f"No data found for cache key {cache_key}")
            return None, None

        X, y = data
        return X, y

    def _save_cache(
        self,
        cache_path: CACHE_PATH_TYPE,
        X: List[XType],
        y: List[YType],
    ):
        """Save a featurized dataset to cache."""
        try_write_pickle(cache_path, (X, y), protocol=4)
        logger.info(f"Saved featurized dataset to cache at {cache_path}")

    def load_featurized_datasets(
        self,
        dataset_repo_name: str,
        dataset_names: Optional[List[str]],
        max_samples_per_dataset: Optional[int] = None,
        preprocess_fn: Optional[
            Callable[[XType, YType], Optional[Tuple[XType, YType]]]
        ] = None,
        force_recompute: bool = False,
    ) -> Tuple[List[XType], List[YType]]:
        """
        Load a featurized dataset from cache/scratch.
        This will load all the datasets into memory, so use with caution.
        """

        all_X, all_y = [], []
        for dataset_name in dataset_names:
            cache_key = self._get_cache_key(
                dataset_repo_name,
                dataset_name,
                self.feature_hash,
                preprocess_fn,
                max_samples_per_dataset,
            )
            if cache_key in self._metadata and not force_recompute:
                logger.info(f"Loading featurized dataset for {dataset_name} from CACHE")
                X, y = self._load_cache(cache_key)
                if X is None:
                    logger.warning(f"No data found for cache key {cache_key}")
                    self._remove_from_metadata(cache_key)
                    raise ValueError(f"No data found for cache key {cache_key}")

                if max_samples_per_dataset is not None:
                    # Just a sanity check
                    assert len(X) == max_samples_per_dataset
                    assert len(y) == max_samples_per_dataset

            else:
                logger.info(
                    f"Loading featurized dataset from SCRATCH for {dataset_name}"
                )

                # Load from scratch, preprocess and featurize data.
                X, y = self._load_dataset(
                    dataset_repo_name=dataset_repo_name,
                    dataset_name=dataset_name,
                    hf_cache_dir=self.hf_cache_dir,
                    max_samples_per_dataset=max_samples_per_dataset,
                )

                # Save to cache, update metadata
                cache_path = self._get_dataset_cache_path(cache_key)
                self._save_cache(cache_path, X, y)
                self._update_metadata(cache_key, cache_path)

                print(f"Before preprocess_fn: {len(X)}")

                if preprocess_fn is not None:
                    preprocessed_X, preprocessed_y = [], []
                    for X_i, y_i in zip(X, y):
                        X_i, y_i = preprocess_fn(X_i, y_i)
                        preprocessed_X.append(X_i)
                        preprocessed_y.append(y_i)
                    X, y = preprocessed_X, preprocessed_y

                print(f"After preprocess_fn: {len(X)}")

            all_X.extend(X)
            all_y.extend(y)

        return all_X, all_y

    def _load_dataset(
        self,
        dataset_repo_name: str,
        dataset_name: str,
        hf_cache_dir: Optional[str] = None,
        max_samples_per_dataset: Optional[int] = None,
    ) -> Tuple[List[XType], List[YType]]:
        """
        Load a single raw dataset from Hugging Face.

        Args:
            dataset_repo_name: Repository name on Hugging Face
            dataset_name: Dataset name to load or None for default dataset
            hf_cache_dir: Directory for Hugging Face to store downloaded datasets
            max_samples_per_dataset: Maximum number of samples to load from the dataset

        Returns:
            The loaded dataset as (X, y) tuple
        """
        # Load from Hugging Face using TabPFNTimeSeriesPretrainDataset
        logger.info(f"Loading raw dataset from {dataset_repo_name}/{dataset_name}")
        start_time = time.time()

        try:
            # Create dataset and dataloader
            dataset = TabPFNTimeSeriesPretrainDataset(
                dataset_repo_name=dataset_repo_name,
                dataset_names=[dataset_name],
                max_context_length=self.max_context_length,
                feature_config=self.feature_config,
                hf_cache_dir=hf_cache_dir,
            )

            # Automatically determine number of workers based on available CPU cores
            num_workers = max(0, multiprocessing.cpu_count() - 1)

            # Use DataLoader to efficiently load all data
            dataloader = DataLoader(
                dataset,
                batch_size=64,  # Adjust batch size as needed
                collate_fn=efficient_collate_fn,
                num_workers=num_workers,
            )

            logger.info(f"Using {num_workers} worker processes for data loading")

            # Collect all data
            all_X, all_y = [], []
            for batch_X, batch_y in tqdm(dataloader, desc=f"Loading {dataset_name}"):
                all_X.extend(batch_X)
                all_y.extend(batch_y)

                # Check if we've reached the maximum number of samples
                if (
                    max_samples_per_dataset is not None
                    and len(all_X) >= max_samples_per_dataset
                ):
                    all_X = all_X[:max_samples_per_dataset]
                    all_y = all_y[:max_samples_per_dataset]
                    break

            logger.info(f"Loaded dataset '{dataset_name}' with {len(all_X)} samples")

        except Exception as e:
            logger.error(
                f"Failed to load dataset from {dataset_repo_name}/{dataset_name}: {e}"
            )
            raise ValueError(
                f"Failed to load dataset from {dataset_repo_name}/{dataset_name}: {e}"
            )

        end_time = time.time()
        logger.info(
            f"Time taken to load raw data: {int(end_time - start_time)} seconds"
        )

        return all_X, all_y

    @staticmethod
    def _hash_config(config: Dict) -> str:
        """Create a hash of a configuration dictionary."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


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
    batch: List[Tuple[XType, YType]],
) -> Tuple[List[XType], List[YType]]:
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


def filter_constant_series(
    X: XType,
    y: YType,
    threshold: float = 1e-8,
) -> Optional[Tuple[XType, YType]]:
    """
    Filter out time series with nearly constant y values.
    """
    # Skip if standard deviation is too small (nearly constant)
    if y.std() < threshold:
        logger.debug(f"Skipping constant series with std: {y.std()}")
        return None

    # Series passed all filters
    return X, y


def load_all_ts_datasets(
    dataset: Union[TabPFNTimeSeriesPretrainDataset, Dict],
    shuffle: bool = False,
    max_length: Optional[int] = None,
    preprocess_fn: Optional[
        Callable[[XType, YType], Optional[Tuple[XType, YType]]]
    ] = None,
    prefix: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[List[XType], List[YType]]:
    """
    Load all time series datasets efficiently.

    This function is maintained for backward compatibility.
    For new code, use TimeSeriesDataManager directly.

    Args:
        dataset: Dataset object or configuration dictionary
        shuffle: Whether to shuffle the dataset
        max_length: Maximum number of samples to load
        preprocess_fn: Optional function to filter/transform samples
        prefix: Optional prefix for debug file names
        cache_dir: Directory to store cached datasets
        force_recompute: If True, recompute even if cached

    Returns:
        Tuple of (all_X, all_y) containing processed features and targets
    """
    # If dataset is already a TabPFNTimeSeriesPretrainDataset, use its data
    if isinstance(dataset, TabPFNTimeSeriesPretrainDataset):
        all_X, all_y = dataset.all_X, dataset.all_y

        # Apply additional preprocessing if needed
        if preprocess_fn and preprocess_fn != dataset.preprocess_fn:
            logger.info(f"Applying additional preprocessing: {preprocess_fn.__name__}")
            filtered_data = []
            for X, y in zip(all_X, all_y):
                result = preprocess_fn(X, y)
                if result is not None:
                    filtered_data.append(result)

            if filtered_data:
                all_X, all_y = zip(*filtered_data)
            else:
                all_X, all_y = [], []

        # Apply max_length if specified
        if max_length is not None and len(all_X) > max_length:
            all_X = all_X[:max_length]
            all_y = all_y[:max_length]

        # Apply shuffle if requested
        if shuffle:
            indices = list(range(len(all_X)))
            random.shuffle(indices)
            all_X = [all_X[i] for i in indices]
            all_y = [all_y[i] for i in indices]

        return all_X, all_y

    # If dataset is a dictionary, create a new TabPFNTimeSeriesPretrainDataset
    elif isinstance(dataset, dict):
        dataset_obj = TabPFNTimeSeriesPretrainDataset(
            dataset_repo_name=dataset.get(
                "dataset_repo_name", "liamsbhoo/GiftEvalPretrainMini"
            ),
            dataset_names=dataset.get("dataset_names"),
            max_context_length=dataset.get("max_context_length", 4096),
            feature_config=dataset.get("feature_config", DEFAULT_FEATURE_CONFIG),
            hf_cache_dir=dataset.get("hf_cache_dir"),
            cache_dir=cache_dir,
            preprocess_fn=preprocess_fn,
            max_samples=max_length,
            force_recompute=force_recompute,
        )

        return dataset_obj.all_X, dataset_obj.all_y

    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def save_data_into_csvs(all_X, all_y, prefix: Optional[str] = None):
    """
    Save dataset to CSV files for debugging purposes.

    Args:
        all_X: List of feature DataFrames
        all_y: List of target Series
        prefix: Optional prefix for file names
    """
    prefix = prefix or "debug"

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
