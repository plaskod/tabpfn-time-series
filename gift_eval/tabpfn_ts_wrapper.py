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
    MAX_CONTEXT_ROWS = 10000

    def __init__(
        self,
        ds_prediction_length: int,
        ds_freq: str,
        ds_season_length: int = None,
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,
        context_length: int = 4096,
        batch_size: int = 1024,
        debug: bool = False,
        retrieval_augmentation: str = "none",
        num_retrieved_subsequences: int = 1, # top_k
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.ds_season_length = ds_season_length
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
            retrieval_augmentation=retrieval_augmentation,
            num_retrieved_subsequences=num_retrieved_subsequences
        )
        self.context_length = min(context_length, self.MAX_CONTEXT_ROWS)
        self.debug = debug
        self.batch_size = batch_size
        self.retrieval_augmentation = retrieval_augmentation
        self.num_retrieved_subsequences = num_retrieved_subsequences

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
        train_tsdf = self.convert_to_timeseries_dataframe(test_data_input)

        # Handle NaN values
        train_tsdf = self.handle_nan_values(train_tsdf)

        # Assert no more NaN in train_tsdf target
        assert not train_tsdf.target.isnull().any()



        # Generate test data
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )

        if self.retrieval_augmentation != "none":
            train_tsdf = self._apply_retrieval_augmentation(train_tsdf, test_tsdf)
        else:
            if self.context_length > 0:
                train_tsdf = self._limit_context_per_item(train_tsdf, self.context_length)

        # Apply feature transformations
        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        train_tsdf = self._enforce_max_context_limit(train_tsdf)
        # Slice if needed
        # if self.context_length > 0:
        #     logger.info(
        #         f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
        #     )
        #     train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        return train_tsdf, test_tsdf
    

    def _apply_retrieval_augmentation(
            self,
            train_tsdf: TimeSeriesDataFrame,
            test_tsdf: TimeSeriesDataFrame
    ) -> TimeSeriesDataFrame:
        augmented_dfs = []

        for item_id in train_tsdf.item_ids:
            item_train = train_tsdf.loc[item_id]
            item_test = test_tsdf.loc[item_id]

            base_context_length = self.ds_season_length # or self.ds_prediction_length
            base_context = item_train.slice_by_timestep(-base_context_length, None)

            if self.retrieval_augmentation == "random":
                augmented_item = self._random_retrieval_augmentation(
                    full_train=item_train,
                    base_context=base_context,
                    test_tsdf=test_tsdf,
                    item_id=item_id,
                )
            else:
                raise ValueError(f"Uknown retrieval augmentation: {self.retrieval_augmentation}")
            
            augmented_dfs.append(augmented_item)

        return TimeSeriesDataFrame(pd.concat(augmented_dfs))
    
    def _random_retrieval_augmentation(
            self,
            full_train: TimeSeriesDataFrame,
            base_context: TimeSeriesDataFrame,
            test_tsdf: TimeSeriesDataFrame,
            item_id: str,
    ) -> TimeSeriesDataFrame:
        subsequence_length = self.ds_season_length + self.ds_prediction_length
        available_history_length = len(full_train) - len(base_context)
        if available_history_length < subsequence_length:
            return base_context

        retrieved_subsequences = []

        remaining_budget = self.context_length - len(base_context)
        max_subsequences = min(
            self.num_retrieved_subsequences,
            remaining_budget // subsequence_length
        )
        if self.debug: print(f"Remaining budget {remaining_budget}, max_subsequences: {max_subsequences}")
        if max_subsequences > 0:
            history_for_sampling = full_train.slice_by_timestep(0, -len(base_context))
            for _ in range(max_subsequences):
                max_start = len(history_for_sampling) - subsequence_length
                if max_start <= 0:
                    break
                start_idx = np.random.randint(0, max_start+1) # does not set random seed select the same start_indx?

                # Extract subsequence (context + horizon)
                context_part = history_for_sampling.slice_by_timestep(start_idx, start_idx + self.ds_season_length)
                horizon_part = history_for_sampling.slice_by_timestep(
                    start_idx + self.ds_season_length,
                    start_idx + subsequence_length
                )

                context_with_features, _ = self.feature_transformer.transform(
                    context_part, 
                    horizon_part
                )

                retrieved_subsequences.append(context_with_features)

        if retrieved_subsequences:
            base_with_features, _ = self.feature_transformer.transform(
                base_context, 
                test_tsdf
            )
            all_parts = retrieved_subsequences + [base_with_features]
                        # Concatenate along the time axis for this item
            combined_dfs = []
            for part in all_parts:
                # Ensure consistent item_id
                part_df = pd.DataFrame(part)
                part_df['item_id'] = item_id
                combined_dfs.append(part_df)
            
            combined = pd.concat(combined_dfs, ignore_index=True)
            combined = combined.set_index(['item_id', combined.index])
            combined.index.names = ['item_id', 'timestamp']
            
            return TimeSeriesDataFrame(combined)
        else:
            return base_context

    def _limit_context_per_item(
        self, 
        train_tsdf: TimeSeriesDataFrame, 
        max_length: int
    ) -> TimeSeriesDataFrame:
        """
        Limit each time series to max_length observations.
        """
        limited_dfs = []
        for item_id in train_tsdf.item_ids:
            item_data = train_tsdf.loc[item_id]
            if len(item_data) > max_length:
                item_data = item_data.slice_by_timestep(-max_length, None)
            limited_dfs.append(item_data)
        
        return TimeSeriesDataFrame(pd.concat(limited_dfs))
    
    def _enforce_max_context_limit(
        self, 
        train_tsdf: TimeSeriesDataFrame
    ) -> TimeSeriesDataFrame:
        """
        Final safety check to ensure total context doesn't exceed MAX_CONTEXT_ROWS.
        This is applied after all augmentation and feature engineering.
        """
        total_rows = len(train_tsdf)
        if total_rows > self.MAX_CONTEXT_ROWS:
            logger.warning(
                f"Total context rows ({total_rows}) exceeds limit ({self.MAX_CONTEXT_ROWS}). "
                f"Truncating each time series proportionally."
            )
            
            # Calculate how much to keep from each time series
            num_items = len(train_tsdf.item_ids)
            max_per_item = self.MAX_CONTEXT_ROWS // num_items
            
            truncated_dfs = []
            for item_id in train_tsdf.item_ids:
                item_data = train_tsdf.loc[item_id]
                if len(item_data) > max_per_item:
                    # Keep the most recent observations
                    item_data = item_data.slice_by_timestep(-max_per_item, None)
                truncated_dfs.append(item_data)
            
            train_tsdf = TimeSeriesDataFrame(pd.concat(truncated_dfs))
            logger.info(f"After truncation: {len(train_tsdf)} total rows")
        
        return train_tsdf
    
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
