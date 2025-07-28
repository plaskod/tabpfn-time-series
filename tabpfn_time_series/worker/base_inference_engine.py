from abc import ABC, abstractmethod

import numpy as np

from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG


class InferenceEngine(ABC):
    @abstractmethod
    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    ) -> dict[str, np.ndarray]:
        pass
