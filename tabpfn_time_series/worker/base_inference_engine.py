from copy import deepcopy

import numpy as np
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
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        model = self.model_class(**self.model_config)

        fit_kwargs = self.inference_kwargs.get("fit", {})
        predict_kwargs = self.inference_kwargs.get("predict", {})

        model.fit(train_X, train_y, **fit_kwargs)
        pred_output = model.predict(test_X, **predict_kwargs)

        return pred_output
