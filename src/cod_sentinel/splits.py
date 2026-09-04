"""Chronological dataset split assignment and persistence."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cod_sentinel.configuration import ProjectConfig

SPLIT_ORDER = ("train", "calibration", "validation", "test")


def assign_temporal_splits(
    orders: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    """Return timestamp-sorted orders with one complete split assignment."""

    config.validate()
    required = {"order_id", "ordered_at"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Cannot split orders; missing columns: {sorted(missing)}")
    if len(orders) != config.dataset_size:
        raise ValueError(
            f"Expected {config.dataset_size} orders, received {len(orders)}."
        )
    if orders["order_id"].duplicated().any():
        raise ValueError("order_id must be unique before splitting.")

    result = orders.sort_values(
        ["ordered_at", "order_id"],
        kind="stable",
    ).reset_index(drop=True)
    counts = asdict(config.splits)
    labels: list[str] = []
    for name in SPLIT_ORDER:
        labels.extend([name] * counts[name])
    if len(labels) != len(result):
        raise ValueError("Temporal split labels do not cover every order.")
    result["split"] = labels
    return result


def persist_split_manifest(orders: pd.DataFrame, path: Path) -> None:
    """Persist split counts and closed timestamp boundaries as JSON."""

    required = {"order_id", "ordered_at", "split"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Cannot persist split manifest: {sorted(missing)}")

    manifest: dict[str, object] = {"total_rows": int(len(orders)), "splits": {}}
    split_manifest: dict[str, object] = {}
    for name in SPLIT_ORDER:
        rows = orders.loc[orders["split"] == name]
        if rows.empty:
            raise ValueError(f"Split {name!r} is empty.")
        split_manifest[name] = {
            "count": int(len(rows)),
            "first_order_id": str(rows.iloc[0]["order_id"]),
            "last_order_id": str(rows.iloc[-1]["order_id"]),
            "start": pd.Timestamp(rows["ordered_at"].min()).isoformat(),
            "end": pd.Timestamp(rows["ordered_at"].max()).isoformat(),
        }
    manifest["splits"] = split_manifest

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
