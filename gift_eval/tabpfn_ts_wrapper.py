from typing import Iterator, Tuple
import logging

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
        batch_size: int = 1024,
        debug: bool = False,
        few_shot_k: int = 0,
        few_shot_len: int = 0,
        few_shot_seed: int = 42,
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
        )
        self.context_length = context_length
        self.debug = debug
        self.batch_size = batch_size
        self.few_shot_k = max(0, few_shot_k)
        self.few_shot_len = max(0, few_shot_len)
        self.few_shot_seed = few_shot_seed

        self.feature_transformer = FeatureTransformer(self.DEFAULT_FEATURES)

    def predict(self, test_data_input) -> Iterator[Forecast]:
        logger.debug(f"len(test_data_input): {len(test_data_input)}, batch size: {self.batch_size}")

        forecasts = []
        for batch in batcher(test_data_input, batch_size=self.batch_size):
            forecasts.extend(self._predict_batch(batch))

        return forecasts

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # TODO: for fixed context length retrieve chunks from 'larger train' here
        # TODO 1: check last and first timestamp 
        # overlap of 'train' vs 'local test' that gets split into context and test query

        # Generate predictions
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
        train_tsdf_full = self.convert_to_timeseries_dataframe(test_data_input)

        # Handle NaN values
        train_tsdf_full = self.handle_nan_values(train_tsdf_full)

        # Assert no more NaN in train_tsdf target
        assert not train_tsdf_full.target.isnull().any()

        # Slice if needed
        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf_full.slice_by_timestep(-self.context_length, None)
        else:
            train_tsdf = train_tsdf_full

        # Few-shot augmentation: sample random subsequences from the earlier (larger) training split
        if self.few_shot_k > 0 and self.few_shot_len > 0:
            rng = np.random.default_rng(self.few_shot_seed)
            support_segments = []

            for item_id, full_item_df in train_tsdf_full.groupby(level="item_id", sort=False):
                # Determine range to sample from: exclude the most recent context region
                full_item_df = full_item_df.copy()
                full_len = len(full_item_df)
                exclude_recent = max(0, self.context_length)
                max_start = full_len - exclude_recent - self.few_shot_len
                if max_start <= 0:
                    continue

                num_samples = min(self.few_shot_k, max_start)
                starts = rng.choice(max_start, size=num_samples, replace=False)
                for s in starts:
                    window_df = full_item_df.iloc[s : s + self.few_shot_len]
                    support_segments.append(window_df)

            if support_segments:
                support_tsdf = TimeSeriesDataFrame(pd.concat(support_segments))
                train_tsdf = TimeSeriesDataFrame(pd.concat([train_tsdf, support_tsdf]).sort_index())

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length, freq=self.ds_freq
        )
        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        return train_tsdf, test_tsdf

    @staticmethod
    def handle_nan_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """
        Handle NaN values in the TimeSeriesDataFrame:
        - If time series has 0 or 1 valid value, fill with 0s
        - Else, drop the NaN values within the time series

        Args:
            tsdf: TimeSeriesDataFrame containing time series data

        Returns:
            TimeSeriesDataFrame: Processed data with NaN values handled
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
        Convert test_data_input to TimeSeriesDataFrame.

        Args:
            test_data_input: List of dictionaries containing time series data
            use_covariates: Whether to include covariates in the output

        Returns:
            TimeSeriesDataFrame: Converted data
        """
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

        # Concat pre-allocated list
        return TimeSeriesDataFrame(pd.concat(time_series))
