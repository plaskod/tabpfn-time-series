from tabpfn_time_series.predictor import TimeSeriesPredictor
from tabpfn_time_series.worker.parallel import GPUParallelWorker
from tabpfn_time_series.experimental.other_tfm.tabdpt.tabdpt_model_adapter import (
    TabDPTModelAdapter,
)


class TabDPTTimeSeriesPredictor(TimeSeriesPredictor):
    """
    A TabDPT-based time series predictor.
    """

    def __init__(
        self,
        tabdpt_config: dict = {},
    ) -> None:
        model_adapter = TabDPTModelAdapter(
            adapter_kwargs=tabdpt_config,
        )

        super().__init__(
            model_adapter=model_adapter,
            worker_class=GPUParallelWorker,
        )
