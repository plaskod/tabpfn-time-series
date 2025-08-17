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


class TabPFNTSPredictor:  # Keep the same class name your evaluate_multigpu.py expects
    """
    TabPFN-TS Predictor with automatic multi-GPU support using LOCAL mode
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
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,
        context_length: int = 4096,
        debug: bool = False,
        num_workers: int = None,  # Ignored - for compatibility only
        gpu_ids: list = None,      # Specify which GPUs to use
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.context_length = context_length
        self.debug = debug
        
        # Configure GPU usage (only if not already set)
        if gpu_ids is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            logger.info(f"Set CUDA_VISIBLE_DEVICES to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Check GPU availability for LOCAL mode
        if tabpfn_mode == TabPFNMode.LOCAL:
            if not torch.cuda.is_available():
                raise ValueError("GPU is required for LOCAL TabPFN inference")
            
            num_gpus = torch.cuda.device_count()
            logger.info(f"Auto-detected {num_gpus} GPUs for parallel processing")
            logger.info(f"Available GPUs: {num_gpus}")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Initialize TabPFN predictor - do NOT pass num_workers
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
        )
        
        self.feature_transformer = FeatureTransformer(self.DEFAULT_FEATURES)
        
        # Log configuration
        if hasattr(self.tabpfn_predictor, 'worker'):
            if hasattr(self.tabpfn_predictor.worker, 'num_workers'):
                logger.info(f"TabPFN configured with {self.tabpfn_predictor.worker.num_workers} workers")

    def predict(self, test_data_input) -> Iterator[Forecast]:
        logger.debug(f"len(test_data_input): {len(test_data_input)}")
        
        # Process in larger batches to better utilize multiple GPUs
        batch_size = min(1024 * max(1, torch.cuda.device_count()), len(test_data_input))
        
        forecasts = []
        for batch in batcher(test_data_input, batch_size=batch_size):
            forecasts.extend(self._predict_batch(batch))

        return forecasts

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")
        
        # Log which GPUs will be used
        if self.debug and torch.cuda.is_available():
            logger.debug(f"Batch will be distributed across {torch.cuda.device_count()} GPUs")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # Generate predictions - this will use multiple GPUs internally
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
        """
        Handle NaN values in the TimeSeriesDataFrame
        """
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
        """
        Convert test_data_input to TimeSeriesDataFrame - FIXED VERSION
        """
        time_series = []

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
            df_with_multiindex = df.copy()
            df_with_multiindex.index = pd.MultiIndex.from_product(
                [[i], df.index], names=["item_id", "timestamp"]
            )
            
            time_series.append(df_with_multiindex)

        # Concat and return as TimeSeriesDataFrame
        combined_df = pd.concat(time_series)
        return TimeSeriesDataFrame(combined_df)