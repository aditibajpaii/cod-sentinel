import json

import pandas as pd
import pytest

import app
from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.versioning import DATASET_VERSION, MODEL_VERSION, POLICY_VERSION


def test_app_module_imports_without_starting_server() -> None:
    assert callable(app.main)


def test_break_even_curve_clips_negative_region_for_display() -> None:
    merchant = load_policy_settings().merchant_economics
    curve = app.break_even_curve(merchant, min_value=200, max_value=1_000, step=200)
    floor = app.minimum_profitable_cod_value(merchant)

    assert 250 < floor < 350
    assert curve["break_even_display"].between(0.0, 1.0).all()
    assert bool(curve.loc[curve["order_value"] == 200, "cod_never_pays"].iloc[0])
    assert not bool(curve.loc[curve["order_value"] == 1_000, "cod_never_pays"].iloc[0])
    assert curve.loc[curve["order_value"] == 1_000, "break_even_raw"].iloc[0] > 0.4


def test_metrics_loader_reads_generated_json(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "evidence_scope": "synthetic",
                "risk_model": {},
                "policies": {},
                "sensitivity": {},
                "versions": {
                    "dataset": DATASET_VERSION,
                    "model": MODEL_VERSION,
                    "policy": POLICY_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )

    assert app.load_metrics(path)["evidence_scope"] == "synthetic"


def test_metrics_loader_reports_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation metrics not found"):
        app.load_metrics(tmp_path / "missing.json")


def test_observable_loader_parses_timestamp(tmp_path, synthetic_world) -> None:
    path = tmp_path / "observable.csv"
    orders = synthetic_world[0].iloc[[0]].copy()
    orders["split"] = "test"
    orders.to_csv(path, index=False)

    loaded = app.load_observable_orders(path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["ordered_at"])
