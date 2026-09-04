"""Artifact hashing and immutable split-contract validation."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from cod_sentinel.configuration import ProjectConfig
from cod_sentinel.splits import SPLIT_ORDER


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_sidecar_path(path: Path | str) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".sha256")


def write_hash_sidecar(path: Path | str) -> Path:
    path = Path(path)
    sidecar = hash_sidecar_path(path)
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="utf-8")
    return sidecar


def verify_hash_sidecar(path: Path | str) -> None:
    path = Path(path)
    sidecar = hash_sidecar_path(path)
    if not sidecar.exists():
        raise FileNotFoundError(f"Artifact hash sidecar not found: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")


def validate_split_manifest(
    observable: pd.DataFrame,
    manifest_path: Path,
    config: ProjectConfig,
) -> None:
    """Bind editable split labels to persisted counts, IDs, and chronology."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("total_rows", -1)) != config.dataset_size:
        raise ValueError("Split manifest total does not match project config.")
    if len(observable) != config.dataset_size:
        raise ValueError("Observable row count does not match project config.")
    if observable["order_id"].duplicated().any():
        raise ValueError("Observable order IDs must be unique.")
    ordered = observable.sort_values(
        ["ordered_at", "order_id"],
        kind="stable",
    ).reset_index(drop=True)
    if not ordered["order_id"].equals(observable.reset_index(drop=True)["order_id"]):
        raise ValueError("Observable orders are not in chronological order.")

    expected_counts = {
        "train": config.splits.train,
        "calibration": config.splits.calibration,
        "validation": config.splits.validation,
        "test": config.splits.test,
    }
    previous_end: pd.Timestamp | None = None
    for name in SPLIT_ORDER:
        rows = observable.loc[observable["split"] == name]
        entry = manifest["splits"].get(name, {})
        if len(rows) != expected_counts[name] or int(entry.get("count", -1)) != len(
            rows
        ):
            raise ValueError(f"Split count mismatch for {name}.")
        if str(rows.iloc[0]["order_id"]) != entry.get("first_order_id"):
            raise ValueError(f"Split first order mismatch for {name}.")
        if str(rows.iloc[-1]["order_id"]) != entry.get("last_order_id"):
            raise ValueError(f"Split last order mismatch for {name}.")
        start = pd.Timestamp(rows["ordered_at"].min())
        end = pd.Timestamp(rows["ordered_at"].max())
        if start.isoformat() != entry.get("start") or end.isoformat() != entry.get(
            "end"
        ):
            raise ValueError(f"Split timestamp boundary mismatch for {name}.")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Temporal splits overlap or are out of order.")
        previous_end = end
