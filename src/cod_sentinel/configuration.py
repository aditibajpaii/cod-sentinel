"""Deterministic scaffold configuration.

Economic, simulator, model, and policy settings belong to their implementation
milestones. This module only defines cross-cutting reproducibility settings.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from cod_sentinel.schemas import Action, MerchantEconomics
from cod_sentinel.versioning import CONFIG_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


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


@dataclass(frozen=True)
class PolicySettings:
    merchant_economics: MerchantEconomics
    tie_break_order: tuple[Action, ...]
    config_version: str


def load_policy_settings(
    path: Path = DEFAULT_POLICY_CONFIG_PATH,
) -> PolicySettings:
    """Load and validate merchant economics and deterministic policy settings."""

    if not path.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("merchant_economics")
    if not isinstance(values, dict):
        raise ValueError("Config must contain a merchant_economics object.")
    raw_order = payload.get("tie_break_order")
    if not isinstance(raw_order, list):
        raise ValueError("Config must contain a tie_break_order list.")
    tie_break_order = tuple(Action(value) for value in raw_order)
    if len(tie_break_order) != len(Action) or set(tie_break_order) != set(Action):
        raise ValueError("tie_break_order must contain each economic action once.")
    config_version = payload.get("config_version")
    if not isinstance(config_version, str) or not config_version:
        raise ValueError("config_version must be a non-empty string.")
    return PolicySettings(
        merchant_economics=MerchantEconomics(**values),
        tie_break_order=tie_break_order,
        config_version=config_version,
    )


def load_merchant_economics(
    path: Path = DEFAULT_POLICY_CONFIG_PATH,
) -> MerchantEconomics:
    """Load validated merchant assumptions from a versioned JSON config."""

    return load_policy_settings(path).merchant_economics
