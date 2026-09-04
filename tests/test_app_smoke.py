import json

import pandas as pd
import pytest

import app


def test_app_module_imports_without_starting_server() -> None:
    assert callable(app.main)


def test_metrics_loader_reads_generated_json(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"evidence_scope": "synthetic"}), encoding="utf-8")

    assert app.load_metrics(path)["evidence_scope"] == "synthetic"


def test_metrics_loader_reports_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation metrics not found"):
        app.load_metrics(tmp_path / "missing.json")


def test_observable_loader_parses_timestamp(tmp_path) -> None:
    path = tmp_path / "observable.csv"
    pd.DataFrame(
        {
            "order_id": ["ORDER-1"],
            "ordered_at": ["2025-01-01T00:00:00"],
        }
    ).to_csv(path, index=False)

    loaded = app.load_observable_orders(path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["ordered_at"])
