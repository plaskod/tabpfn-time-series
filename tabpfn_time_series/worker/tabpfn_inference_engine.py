import numpy as np

from tabpfn import TabPFNRegressor
from tabpfn_client import init, TabPFNRegressor as TabPFNClientRegressor

from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG
from tabpfn_time_series.worker.base_inference_engine import InferenceEngine


def process_tabpfn_pred_output(
    pred_output: dict,
    output_selection: str,
    quantiles: list[float | str],
) -> dict[str, np.ndarray]:
    result = {"target": pred_output[output_selection]}

    result.update({q: q_pred for q, q_pred in zip(quantiles, pred_output["quantiles"])})

    return result


class TabPFNClientInferenceEngine(InferenceEngine):
    def __init__(self, tabpfn_config: dict):
        init()

        self.tabpfn_config = tabpfn_config.copy()
        self.tabpfn_config["tabpfn_internal"]["model_path"] = self._parse_model_name(
            self.tabpfn_config["tabpfn_internal"]["model_path"]
        )

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    ) -> dict[str, np.ndarray]:
        model = TabPFNClientRegressor(**self.tabpfn_config["tabpfn_internal"])

        model.fit(train_X, train_y)
        pred_output = model.predict(test_X, output_type="main")

        return process_tabpfn_pred_output(
            pred_output,
            self.tabpfn_config["tabpfn_output_selection"],
            quantiles,
        )

    def _parse_model_name(self, model_name: str) -> str:
        available_models = TabPFNClientRegressor.list_available_models()

        for m in available_models:
            if m in model_name:
                return m
        raise ValueError(
            f"Model {model_name} not found. Available models: {available_models}."
        )


class LocalTabPFNInferenceEngine(InferenceEngine):
    def __init__(self, tabpfn_config: dict):
        self._download_model(tabpfn_config["tabpfn_internal"]["model_path"])
        self.tabpfn_config = tabpfn_config.copy()

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    ) -> dict[str, np.ndarray]:
        model = TabPFNRegressor(**self.tabpfn_config["tabpfn_internal"])

        model.fit(train_X, train_y)
        pred_output = model.predict(test_X, output_type="main")

        return process_tabpfn_pred_output(
            pred_output,
            self.tabpfn_config["tabpfn_output_selection"],
            quantiles,
        )

    @staticmethod
    def _download_model(model_name: str):
        from tabpfn.model.loading import resolve_model_path, download_model

        # Resolve the model path
        # If the model path is not specified, this resolves to the default model path
        model_path, _, model_name, which = resolve_model_path(
            model_name,
            which="regressor",
        )

        if not model_path.exists():
            download_model(
                to=model_path,
                which=which,
                version="v2",
                model_name=model_name,
            )


class MockTabPFNInferenceEngine(InferenceEngine):
    def __init__(self, tabpfn_config: dict):
        pass

    def run(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
    ) -> dict[str, np.ndarray]:
        return {
            "target": np.zeros(len(test_X)),
            "quantiles": np.zeros((len(test_X), len(quantiles))),
        }
