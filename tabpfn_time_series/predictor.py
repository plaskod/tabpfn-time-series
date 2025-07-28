import logging
from enum import Enum

from typing import Callable, Type

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_CONFIG
from tabpfn_time_series.worker.parallel import (
    ParallelWorker,
    CPUParallelWorker,
    GPUParallelWorker,
)
from tabpfn_time_series.worker.tabpfn_inference_engine import (
    TabPFNClientInferenceEngine,
    LocalTabPFNInferenceEngine,
    MockTabPFNInferenceEngine,
)


logger = logging.getLogger(__name__)


class TabPFNMode(Enum):
    LOCAL = "tabpfn-local"
    CLIENT = "tabpfn-client"
    MOCK = "tabpfn-mock"


class TimeSeriesPredictor:
    """
    Given a TimeSeriesDataFrame (multiple time series), perform prediction on each time series individually.
    """

    def __init__(
        self,
        inference_routine: Callable,
        worker_class: Type[ParallelWorker],
        worker_kwargs: dict = {},
    ):
        self.inference_routine = inference_routine
        self.worker = worker_class(inference_routine, **worker_kwargs)

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        """
        Predict on each time series individually (local forecasting).
        """

        logger.info(f"Predicting {len(train_tsdf.item_ids)} time series...")

        return self.worker.predict(train_tsdf, test_tsdf)


class TabPFNTimeSeriesPredictor(TimeSeriesPredictor):
    """
    Given a TimeSeriesDataFrame (multiple time series), perform prediction on each time series individually.
    """

    def __init__(
        self,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        config: dict = TABPFN_TS_DEFAULT_CONFIG,
    ) -> None:
        inference_engine_mapping = {
            TabPFNMode.CLIENT: lambda: TabPFNClientInferenceEngine(config),
            TabPFNMode.LOCAL: lambda: LocalTabPFNInferenceEngine(config),
            TabPFNMode.MOCK: lambda: MockTabPFNInferenceEngine(config),
        }

        worker_mapping = {
            TabPFNMode.CLIENT: CPUParallelWorker,
            TabPFNMode.LOCAL: GPUParallelWorker,
            TabPFNMode.MOCK: CPUParallelWorker,
        }

        self.inference_engine = inference_engine_mapping[tabpfn_mode]()
        self.worker = worker_mapping[tabpfn_mode](
            inference_routine=self.inference_engine.run,
        )
