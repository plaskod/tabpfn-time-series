from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


class BaseInferenceEngine:
    def __init__(
        self,
        model_class: RegressorMixin,
        model_config: dict,
        inference_kwargs: dict = {},
    ):
        self.model_class = model_class
        self.model_config = deepcopy(model_config)
        self.inference_kwargs = deepcopy(inference_kwargs)

    def predict(
        self,
        train_X: np.ndarray | pd.DataFrame,
        train_y: np.ndarray | pd.Series,
        test_X: np.ndarray | pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        model = self.model_class(**self.model_config)

        fit_kwargs = self.inference_kwargs.get("fit", {})
        predict_kwargs = self.inference_kwargs.get("predict", {})

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
