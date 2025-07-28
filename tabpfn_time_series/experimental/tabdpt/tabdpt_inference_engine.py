from tabpfn_time_series.worker.base_inference_engine import BaseInferenceEngine
from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG

import numpy as np

from tabdpt import TabDPTRegressor


class TabDPTInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        inference_kwargs: dict = {},
    ):
        super().__init__(
            model_class=TabDPTRegressor,
            model_config={
                "use_flash": False,
                "compile": False,
                "device": "cuda",
            },
            inference_kwargs=inference_kwargs,
        )

    def predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = DEFAULT_QUANTILE_CONFIG,
    ):
        pred_output = super().predict(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
        )

        # TabDPT doesn't return uncertainty estimates
        # so workaround, we will return the estimated target instead
        # Therefore, we ignore the uncertainty estimates.
        result = {"target": pred_output}
        result.update({q: pred_output for q in quantiles})

        return result
