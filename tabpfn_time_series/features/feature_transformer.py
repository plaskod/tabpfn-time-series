from typing import List, Tuple

import pandas as pd

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


class FeatureTransformer:
    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both train and test data with the configured feature generators"""

        self._validate_input(train_tsdf, test_tsdf, target_column)

        # Mark rows to preserve correct train/test split after groupby-apply
        train_tsdf = train_tsdf.copy()
        test_tsdf = test_tsdf.copy()
        train_tsdf["__is_test"] = 0
        test_tsdf["__is_test"] = 1

        # Concatenate and ensure deterministic time order within each item/segment
        tsdf = pd.concat([train_tsdf, test_tsdf])
        tsdf = tsdf.sort_index(level=["item_id", "timestamp"])  # stable time order

        # Apply all feature generators
        # If segment_id is present, isolate features per (item_id, segment_id)
        for generator in self.feature_generators:
            if "segment_id" in tsdf.columns:
                tsdf = (
                    tsdf.groupby([pd.Grouper(level="item_id"), "segment_id"], group_keys=False)
                    .apply(generator)
                )
            else:
                tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(generator)

        # Split train and test tsdf using the marker to avoid order-related leakage
        train_tsdf = tsdf[tsdf["__is_test"] == 0].drop(columns=["__is_test"])  # type: ignore
        test_tsdf = tsdf[tsdf["__is_test"] == 1].drop(columns=["__is_test"])  # type: ignore

        assert not train_tsdf[target_column].isna().any(), (
            "All target values in train_tsdf should be non-NaN"
        )
        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf

    @staticmethod
    def _validate_input(
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str,
    ):
        if target_column not in train_tsdf.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        if not test_tsdf[target_column].isna().all():
            raise ValueError("Test data should not contain target values")
