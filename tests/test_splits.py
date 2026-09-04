import json

import pandas as pd
import pytest

from cod_sentinel.configuration import DEFAULT_CONFIG
from cod_sentinel.splits import SPLIT_ORDER, assign_temporal_splits, persist_split_manifest


def test_default_split_counts_and_chronology(synthetic_world) -> None:
    observable, _, _ = synthetic_world

    assert observable["split"].value_counts().to_dict() == {
        "train": 13_000,
        "calibration": 2_000,
        "validation": 2_000,
        "test": 3_000,
    }
    previous_end = None
    for name in SPLIT_ORDER:
        rows = observable.loc[observable["split"] == name]
        start = pd.Timestamp(rows["ordered_at"].min())
        end = pd.Timestamp(rows["ordered_at"].max())
        if previous_end is not None:
            assert start > previous_end
        previous_end = end


def test_split_rejects_wrong_row_count() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["ORDER-1"],
            "ordered_at": [pd.Timestamp("2025-01-01")],
        }
    )

    with pytest.raises(ValueError, match="Expected 20000"):
        assign_temporal_splits(orders, DEFAULT_CONFIG)


def test_split_manifest_persists_complete_boundaries(
    synthetic_world,
    tmp_path,
) -> None:
    observable, _, _ = synthetic_world
    path = tmp_path / "splits.json"

    persist_split_manifest(observable, path)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    assert manifest["total_rows"] == 20_000
    assert sum(
        split["count"] for split in manifest["splits"].values()
    ) == 20_000
    assert list(manifest["splits"]) == list(SPLIT_ORDER)
