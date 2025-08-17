#!/usr/bin/env python3
"""
Fixed TabPFN-TS wrapper with proper multi-GPU support
"""

import os
import torch
import logging
from typing import Iterator, Tuple
import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast, Forecast
from gluonts.itertools import batcher

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import (
    TabPFNTimeSeriesPredictor,
    FeatureTransformer,
    TabPFNMode,
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

logger = logging.getLogger(__name__)


class TabPFNTSPredictor:
    """
    TabPFN-TS Predictor with automatic multi-GPU support using LOCAL mode
    
    When using TabPFNMode.LOCAL, the predictor automatically:
    1. Detects the number of available GPUs
    2. Sets num_workers = torch.cuda.device_count()
    3. Distributes time series across GPUs for parallel processing
    """
    DEFAULT_FEATURES = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]

    def __init__(
        self,
        ds_prediction_length: int,
        ds_freq: str,
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,  # Use LOCAL for multi-GPU
        context_length: int = 4096,
        debug: bool = False,
        num_workers: int = None,  # For compatibility, but not used with LOCAL mode
        gpu_ids: list = None,      # Specify which GPUs to use
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.context_length = context_length
        self.debug = debug
        
        # Configure GPU visibility BEFORE creating the predictor
        if gpu_ids is not None:
            # Set CUDA_VISIBLE_DEVICES to use specific GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            logger.info(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Check GPU availability
        if tabpfn_mode == TabPFNMode.LOCAL and not torch.cuda.is_available():
            raise ValueError("GPU is required for LOCAL TabPFN inference")
        
        # Log GPU configuration
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"CUDA Available: True")
            logger.info(f"Number of visible GPUs: {num_gpus}")
            logger.info(f"TabPFN will use {num_gpus} workers for parallel processing")
            
            for i in range(num_gpus):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info(f"CUDA Available: False - using CLIENT mode")
        
        # Initialize TabPFN predictor
        # When using LOCAL mode, it automatically sets num_workers = torch.cuda.device_count()
        # Do NOT pass num_workers as it's not a valid parameter
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
        )
        
        # Verify the configuration
        if hasattr(self.tabpfn_predictor, 'worker'):
            if hasattr(self.tabpfn_predictor.worker, 'num_workers'):
                actual_workers = self.tabpfn_predictor.worker.num_workers
                logger.info(f"TabPFN worker configured with {actual_workers} workers")
        
        self.feature_transformer = FeatureTransformer(self.DEFAULT_FEATURES)

    def predict(self, test_data_input) -> Iterator[Forecast]:
        logger.debug(f"Processing {len(test_data_input)} time series")
        
        # Use larger batch size for multi-GPU processing
        # This allows more time series to be distributed across GPUs
        num_gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        batch_size = min(1024 * num_gpus, len(test_data_input))
        
        logger.debug(f"Using batch size: {batch_size} for {num_gpus} GPUs")
        
        forecasts = []
        for batch_idx, batch in enumerate(batcher(test_data_input, batch_size=batch_size)):
            logger.debug(f"Processing batch {batch_idx + 1}")
            forecasts.extend(self._predict_batch(batch))

        return forecasts

    def _predict_batch(self, test_data_input):
        batch_size = len(test_data_input)
        logger.debug(f"Processing batch of size: {batch_size}")
        
        # Log expected GPU distribution
        if self.debug and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            per_gpu = batch_size // num_gpus if batch_size >= num_gpus else batch_size
            logger.debug(f"Expected distribution: ~{per_gpu} time series per GPU across {num_gpus} GPUs")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # Generate predictions
        # The LocalTabPFN worker will automatically:
        # 1. Shuffle and split time series into chunks
        # 2. Assign each chunk to a different GPU
        # 3. Process chunks in parallel using joblib
        logger.debug(f"Starting parallel prediction on {len(train_tsdf.item_ids)} time series")
        
        pred: TimeSeriesDataFrame = self.tabpfn_predictor.predict(train_tsdf, test_tsdf)
        pred = pred.drop(columns=["target"])

        # Pre-allocate forecasts list and get forecast quantile keys
        forecasts = [None] * len(pred.item_ids)
        forecast_keys = list(map(str, TABPFN_TS_DEFAULT_QUANTILE_CONFIG))

        # Generate QuantileForecast objects for each time series
        for i, (_, item_data) in enumerate(pred.groupby(level="item_id")):
            forecast_start_timestamp = item_data.index.get_level_values(1)[0]
            forecasts[i] = QuantileForecast(
                forecast_arrays=item_data.values.T,
                forecast_keys=forecast_keys,
                start_date=forecast_start_timestamp.to_period(self.ds_freq),
            )

        logger.debug(f"Generated {len(forecasts)} forecasts")
        return forecasts

    def _preprocess_test_data(
        self, test_data_input
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Preprocess includes:
        - Turn the test_data_input into a TimeSeriesDataFrame
        - Handle NaN values in "target" column
        - If context_length is set, slice the train_tsdf to the last context_length timesteps
        - Generate test data and apply feature transformations
        """
        # Convert input to TimeSeriesDataFrame
        train_tsdf = self.convert_to_timeseries_dataframe(test_data_input)

        # Handle NaN values
        train_tsdf = self.handle_nan_values(train_tsdf)

        # Assert no more NaN in train_tsdf target
        assert not train_tsdf.target.isnull().any()

        # Slice if needed
        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )
        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        return train_tsdf, test_tsdf

    @staticmethod
    def handle_nan_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """Handle NaN values in the TimeSeriesDataFrame"""
        processed_series = []
        ts_with_0_or_1_valid_value = []
        ts_with_nan = []

        # Process each time series individually
        for item_id, item_data in tsdf.groupby(level="item_id"):
            target = item_data.target.values
            timestamps = item_data.index.get_level_values("timestamp")

            # If there are 0 or 1 valid values, fill NaNs with 0
            valid_value_count = np.count_nonzero(~np.isnan(target))
            if valid_value_count <= 1:
                ts_with_0_or_1_valid_value.append(item_id)
                target = np.where(np.isnan(target), 0, target)
                processed_df = pd.DataFrame(
                    {"target": target},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps], names=["item_id", "timestamp"]
                    ),
                )
                processed_series.append(processed_df)

            # Else drop NaN values
            elif np.isnan(target).any():
                ts_with_nan.append(item_id)
                valid_indices = ~np.isnan(target)
                processed_df = pd.DataFrame(
                    {"target": target[valid_indices]},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps[valid_indices]],
                        names=["item_id", "timestamp"],
                    ),
                )
                processed_series.append(processed_df)

            # No NaNs, keep as is
            else:
                processed_series.append(item_data)

        # Log warnings about NaN handling
        if ts_with_0_or_1_valid_value:
            logger.warning(
                f"Found time-series with 0 or 1 valid values, item_ids: {ts_with_0_or_1_valid_value}"
            )

        if ts_with_nan:
            logger.warning(
                f"Found time-series with NaN targets, item_ids: {ts_with_nan}"
            )

        # Combine processed series
        return TimeSeriesDataFrame(pd.concat(processed_series))

    @staticmethod
    def convert_to_timeseries_dataframe(test_data_input, use_covariates: bool = False):
        """Convert test_data_input to TimeSeriesDataFrame"""
        # Pre-allocate list with known size
        time_series = [None] * len(test_data_input)

        for i, item in enumerate(test_data_input):
            target = item["target"]

            # Create timestamp index
            timestamp = pd.date_range(
                start=item["start"].to_timestamp(),
                periods=len(target),
                freq=item["freq"],
            )

            # Create DataFrame with target
            df = pd.DataFrame({"target": target}, index=timestamp)

            # Create MultiIndex DataFrame
            time_series[i] = df.set_index(
                pd.MultiIndex.from_product(
                    [[i], df.index], names=["item_id", "timestamp"]
                )
            )

        #