from typing import List, Tuple

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series.experimental.features.feature_generator_base import (
    FeatureGenerator,
)


class FeatureTransformer:
    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame = None,
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both train and test data with the configured feature generators"""

        # Validate input for train data
        if target_column not in train_tsdf.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )
        
        # If test_tsdf is None, only process train data
        if test_tsdf is None:
            # Apply all feature generators to train data only
            transformed_train = train_tsdf.copy()
            for generator in self.feature_generators:
                transformed_train = transformed_train.groupby(level="item_id", group_keys=False).apply(generator)
            
            assert (
                not transformed_train[target_column].isna().any()
            ), "All target values in train_tsdf should be non-NaN"
            
            return transformed_train, None
        
        # Validate test data if provided
        self._validate_input(train_tsdf, test_tsdf, target_column)
        
        # Process both train and test data together
        tsdf = pd.concat([train_tsdf, test_tsdf])
        
        # Apply all feature generators
        for generator in self.feature_generators:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(generator)
        
        # Split back into train and test
        transformed_train = tsdf.iloc[: len(train_tsdf)]
        transformed_test = tsdf.iloc[len(train_tsdf) :]
        
        assert (
            not transformed_train[target_column].isna().any()
        ), "All target values in train_tsdf should be non-NaN"
        assert transformed_test[target_column].isna().all()
        
        return transformed_train, transformed_test

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
