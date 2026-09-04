import pytest

from cod_sentinel.configuration import DEFAULT_CONFIG
from cod_sentinel.integrity import (
    validate_split_manifest,
    verify_hash_sidecar,
    write_hash_sidecar,
)
from cod_sentinel.splits import persist_split_manifest


def test_split_manifest_rejects_relabelled_test_rows(
    synthetic_world,
    tmp_path,
) -> None:
    observable, _, _ = synthetic_world
    manifest_path = tmp_path / "splits.json"
    persist_split_manifest(observable, manifest_path)
    corrupted = observable.copy()
    corrupted.loc[corrupted["split"] == "test", "split"] = "validation"

    with pytest.raises(ValueError, match="Split count mismatch"):
        validate_split_manifest(corrupted, manifest_path, DEFAULT_CONFIG)


def test_hash_sidecar_detects_modified_artifact(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"trusted")
    write_hash_sidecar(artifact)
    verify_hash_sidecar(artifact)

    artifact.write_bytes(b"modified")

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hash_sidecar(artifact)
