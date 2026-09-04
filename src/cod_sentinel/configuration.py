"""Deterministic scaffold configuration.

Economic, simulator, model, and policy settings belong to their implementation
milestones. This module only defines cross-cutting reproducibility settings.
"""

from dataclasses import dataclass
from pathlib import Path

from cod_sentinel.versioning import CONFIG_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class SplitSizes:
    train: int = 13_000
    calibration: int = 2_000
    validation: int = 2_000
    test: int = 3_000

    @property
    def total(self) -> int:
        return self.train + self.calibration + self.validation + self.test

    def validate(self, expected_total: int) -> None:
        values = (self.train, self.calibration, self.validation, self.test)
        if any(value <= 0 for value in values):
            raise ValueError("Every temporal split must contain at least one row.")
        if self.total != expected_total:
            raise ValueError(
                f"Split total {self.total} does not match dataset size "
                f"{expected_total}."
            )


@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int = 42
    dataset_size: int = 20_000
    splits: SplitSizes = SplitSizes()
    config_version: str = CONFIG_VERSION

    def validate(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive.")
        self.splits.validate(self.dataset_size)


DEFAULT_CONFIG = ProjectConfig()
DEFAULT_CONFIG.validate()
