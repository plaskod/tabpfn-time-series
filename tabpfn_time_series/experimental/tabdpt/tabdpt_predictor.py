from tabpfn_time_series.predictor import TimeSeriesPredictor
from tabpfn_time_series.worker.parallel import GPUParallelWorker

from tabpfn_time_series.experimental.tabdpt.tabdpt_inference_engine import (
    TabDPTInferenceEngine,
)


class TabDPTTimeSeriesPredictor(TimeSeriesPredictor):
    DEFAULT_INFERENCE_KWARGS = {
        "predict": {
            "n_ensembles": 8,
            "context_size": 10000,
            "seed": 42,
        }
    }

    def __init__(
        self,
        inference_kwargs: dict = DEFAULT_INFERENCE_KWARGS,
    ):
        super().__init__(
            inference_engine=TabDPTInferenceEngine(
                inference_kwargs=inference_kwargs,
            ),
            worker_class=GPUParallelWorker,
        )
