from copy import deepcopy
from typing import Any, Dict, Type, TypeAlias, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


InferenceConfig: TypeAlias = Dict[str, Any]
"""
Configuration dictionary for model adapters.

Expected structure:
{
    "fit": {
        "param1": value1,
        "param2": value2,
        ...
    },
    "predict": {
        "param1": value1,
        "param2": value2,
        ...
    }
}
"""


class ModelAdapter:
    """Base model adapter for scikit-learn compatible models."""

    def __init__(
        self,
        model_class: Type[RegressorMixin],
        model_config: Dict[str, Any],
        inference_config: InferenceConfig = None,
    ) -> None:
        """
        Initialize the base model adapter.

        Args:
            model_class: Scikit-learn compatible regressor class
            model_config: Configuration parameters for model initialization
            inference_config: Configuration for fit and predict methods
        """
        self.model_class = model_class
        self.model_config = deepcopy(model_config)
        self.inference_config = deepcopy(inference_config or {})

    def predict(
        self,
        train_X: Union[np.ndarray, pd.DataFrame],
        train_y: Union[np.ndarray, pd.Series],
        test_X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Train model and make predictions.

        Args:
            train_X: Training features
            train_y: Training targets
            test_X: Test features

        Returns:
            Predictions as numpy array
        """
        model = self.model_class(**self.model_config)

        fit_kwargs = self.inference_config.get("fit", {})
        predict_kwargs = self.inference_config.get("predict", {})

        # Convert dataframe to numpy array
        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.values
        if isinstance(train_y, pd.Series):
            train_y = train_y.values
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values

        model.fit(train_X, train_y, **fit_kwargs)
        pred_output = model.predict(test_X, **predict_kwargs)

        return pred_output
