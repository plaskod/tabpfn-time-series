from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Any
import json
from enum import Enum


class OptimizationSpace(Enum):
    RAW = "raw"
    PREPROCESSED = "preprocessed"


@dataclass
class FinetuneConfig:
    model: "FinetuneConfig.Model" = field(
        default_factory=lambda: FinetuneConfig.Model()
    )
    optimization: "FinetuneConfig.Optimization" = field(
        default_factory=lambda: FinetuneConfig.Optimization()
    )
    preprocessing: "FinetuneConfig.TabPFNPreprocessing" = field(
        default_factory=lambda: FinetuneConfig.TabPFNPreprocessing()
    )
    train_dataset: "FinetuneConfig.TrainDataset" = field(
        default_factory=lambda: FinetuneConfig.TrainDataset()
    )
    test_dataset: "FinetuneConfig.TestDataset" = field(
        default_factory=lambda: FinetuneConfig.TestDataset()
    )

    @dataclass
    class Model:
        ignore_pretraining_limits: bool = True
        n_estimators: int = 8
        random_state: int = 0
        differentiable_input: bool = False
        inference_precision: str = "float32"

    @dataclass
    class Optimization:
        n_epochs: int = 5
        lr: float = 1e-5
        space: OptimizationSpace = OptimizationSpace.PREPROCESSED
        gradient_accumulation_steps: int = 16

    @dataclass
    class TabPFNPreprocessing:
        max_data_size: int = 10000

    @dataclass
    class TrainDataset:
        dataset_repo_name: str = "liamsbhoo/GiftEvalPretrain"
        dataset_names: Tuple[str, ...] = (
            "bdg-2_panther",
            "bdg-2_fox",
            "bdg-2_rat",
        )
        max_context_length: int = 1000
        prediction_length: int = 400

    @dataclass
    class TestDataset:
        dataset_repo_name: str = "liamsbhoo/GiftEvalPretrain"
        dataset_names: Tuple[str, ...] = ("bdg-2_bear",)
        max_context_length: int = 1000
        prediction_length: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        return {
            "model": asdict(self.model),
            "optimization": asdict(self.optimization),
            "train_dataset": asdict(self.train_dataset),
            "test_dataset": asdict(self.test_dataset),
        }

    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
