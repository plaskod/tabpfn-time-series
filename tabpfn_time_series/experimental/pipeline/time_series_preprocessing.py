import pandas as pd
import numpy as np
import logging

from autogluon.timeseries import TimeSeriesDataFrame

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    def __init__(
        self,
        max_context_length: int = 4096,
        handle_nan_values: bool = True,
    ):
        self.max_context_length = max_context_length
        if not ((max_context_length > 0) and isinstance(max_context_length, int)):
            raise ValueError(
                f"max_context_length must be a positive integer, got {max_context_length}"
            )

        self.should_handle_nan_values = handle_nan_values

    def forward(
        self,
        tsdf: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        # Handle NaN values
        if self.should_handle_nan_values:
            tsdf = self.handle_nan_values(tsdf)

        # Slice to max_context_length
        logger.debug(
            f"Slicing tsdf from {tsdf.shape[0]} to {self.max_context_length} timesteps"
        )
        tsdf = tsdf.slice_by_timestep(-self.max_context_length, None)

        return tsdf

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

            # Count NaN values
            nan_count = np.count_nonzero(np.isnan(target))
            valid_value_count = len(target) - nan_count

            # If there are 0 or 1 valid values, fill NaNs with 0
            if valid_value_count <= 1:
                ts_with_0_or_1_valid_value.append((item_id, nan_count))
                target = np.where(np.isnan(target), 0, target)
                processed_df = pd.DataFrame(
                    {"target": target},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps], names=["item_id", "timestamp"]
                    ),
                )
                processed_series.append(processed_df)

            # Else drop NaN values
            elif nan_count > 0:
                ts_with_nan.append((item_id, nan_count))
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
                f"Found time-series with 0 or 1 valid values, (item_ids, nan_count): {ts_with_0_or_1_valid_value}"
            )

        if ts_with_nan:
            logger.warning(
                f"Found time-series with NaN targets, (item_ids, nan_count): {ts_with_nan}"
            )

        # Combine processed series
        return TimeSeriesDataFrame(pd.concat(processed_series))
