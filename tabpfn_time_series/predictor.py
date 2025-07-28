import logging
from enum import Enum
from typing import Type

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_CONFIG
from tabpfn_time_series.worker.parallel import (
    ParallelWorker,
    CPUParallelWorker,
    GPUParallelWorker,
)
from tabpfn_time_series.worker.tabpfn_inference_engine import (
    BaseInferenceEngine,
    TabPFNClientInferenceEngine,
    LocalTabPFNInferenceEngine,
)


logger = logging.getLogger(__name__)


class TabPFNMode(Enum):
    LOCAL = "tabpfn-local"
    CLIENT = "tabpfn-client"


class TimeSeriesPredictor:
    """
    Given a TimeSeriesDataFrame (multiple time series), perform prediction on each time series individually.
    """

    def __init__(
        self,
        inference_engine: Type[BaseInferenceEngine],
        worker_class: Type[ParallelWorker],
        worker_kwargs: dict = {},
    ):
        self.worker = worker_class(inference_engine.predict, **worker_kwargs)

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
        tabpfn_config: dict = TABPFN_TS_DEFAULT_CONFIG,
        tabpfn_output_selection: str = "median",  # mean or median
    ) -> None:
        inference_engine_mapping = {
            TabPFNMode.CLIENT: TabPFNClientInferenceEngine,
            TabPFNMode.LOCAL: LocalTabPFNInferenceEngine,
        }

        worker_mapping = {
            TabPFNMode.CLIENT: CPUParallelWorker,
            TabPFNMode.LOCAL: GPUParallelWorker,
        }

        inference_engine = inference_engine_mapping[tabpfn_mode](
            tabpfn_config,
            tabpfn_output_selection,
        )

        super().__init__(
            inference_engine=inference_engine,
            worker_class=worker_mapping[tabpfn_mode],
        )
