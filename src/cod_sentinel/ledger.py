"""Append-only decision ledger with hash chaining and replay verification.

This module is a sink, not a dependency. No runtime decision reads it, and
`DecisionEngine.decide` never writes to it, so the policy stays pure and
deterministic. Records carry only the runtime feature allowlist, so oracle
columns cannot reach the log.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from cod_sentinel.configuration import ARTIFACTS_DIR, load_policy_settings
from cod_sentinel.features import RUNTIME_FEATURES
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.policy import Decision, DecisionEngine

LEDGER_PATH = ARTIFACTS_DIR / "decision_ledger.jsonl"
SAMPLE_LEDGER_PATH = ARTIFACTS_DIR / "decision_ledger_sample.jsonl"
RECORD_SCHEMA_VERSION = "decision-record-v1"
GENESIS_HASH = "0" * 64


def _json_safe(value: object) -> object:
    """Convert a pandas/numpy scalar into a JSON-serializable value."""

    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _canonical(record: Mapping[str, object]) -> str:
    """Serialize a record deterministically so hashes are stable."""

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def record_hash(record: Mapping[str, object]) -> str:
    """Return the sha256 of one complete record, including its chain link."""

    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def record_from_decision(
    decision: Decision,
    order: Mapping[str, object],
    recorded_at: str | None = None,
) -> dict[str, object]:
    """Build an audit record from a decision and the order that produced it."""

    payload = decision.to_dict()
    timestamp = recorded_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "recorded_at": timestamp,
        "decision_id": payload["decision_id"],
        "order_id": payload["order_id"],
        "selected_action": payload["selected_action"],
        "reason_codes": list(payload["reason_codes"]),
        "probabilities": payload["probabilities"],
        "expected_contributions": payload["expected_contributions"],
        "versions": payload["versions"],
        "config_version": payload["config_version"],
        "inputs": {
            name: _json_safe(order[name])
            for name in RUNTIME_FEATURES
            if name in order
        },
    }


def read_ledger(path: Path = LEDGER_PATH) -> list[dict[str, object]]:
    """Read every record from a ledger file, oldest first."""

    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_decision(
    record: Mapping[str, object],
    path: Path = LEDGER_PATH,
) -> dict[str, object]:
    """Append one record, linking it to the hash of the previous record."""

    existing = read_ledger(path)
    previous_hash = record_hash(existing[-1]) if existing else GENESIS_HASH
    linked = {**dict(record), "previous_hash": previous_hash}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(linked) + "\n")
    return linked


def verify_chain(records: list[dict[str, object]]) -> int | None:
    """Return the index of the first broken link, or None if intact.

    Editing any historical record changes its hash, so every later link fails.
    """

    expected = GENESIS_HASH
    for index, record in enumerate(records):
        if record.get("previous_hash") != expected:
            return index
        expected = record_hash(record)
    return None


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of re-deriving a recorded decision from its logged inputs."""

    matches: bool
    status: str
    detail: str
    recorded_decision_id: str
    replayed_decision_id: str


def replay(record: Mapping[str, object], engine: DecisionEngine) -> Decision:
    """Re-run the policy over the inputs stored in an audit record."""

    row = {**dict(record["inputs"]), "order_id": record["order_id"]}
    return engine.decide(row)


def verify_replay(
    record: Mapping[str, object],
    engine: DecisionEngine,
) -> ReplayResult:
    """Replay a record and explain any mismatch as drift or corruption.

    A decision id is a hash over the inputs, economics, and versions, so a
    changed model or config deliberately produces a different id. Separating
    that from an input mismatch is the difference between expected drift and
    a corrupted log.
    """

    decision = replay(record, engine)
    recorded_id = str(record["decision_id"])
    if decision.decision_id == recorded_id:
        return ReplayResult(
            matches=True,
            status="match",
            detail="Replayed decision is identical to the recorded decision.",
            recorded_decision_id=recorded_id,
            replayed_decision_id=decision.decision_id,
        )

    changed = sorted(
        name
        for name, value in decision.versions.items()
        if dict(record["versions"]).get(name) != value
    )
    if changed:
        status = "version_drift"
        detail = f"Artifact versions changed since recording: {changed}."
    elif record["config_version"] != decision.config_version:
        status = "config_drift"
        detail = (
            f"Merchant config changed from {record['config_version']!r} to "
            f"{decision.config_version!r}."
        )
    else:
        status = "input_mismatch"
        detail = (
            "Versions and config are unchanged, so the recorded inputs no "
            "longer reproduce the recorded decision."
        )
    return ReplayResult(
        matches=False,
        status=status,
        detail=detail,
        recorded_decision_id=recorded_id,
        replayed_decision_id=decision.decision_id,
    )


def write_sample_ledger(
    limit: int = 25,
    path: Path = SAMPLE_LEDGER_PATH,
    observable_path: Path = OBSERVABLE_PATH,
    bundle_path: Path = MODEL_BUNDLE_PATH,
) -> Path:
    """Write a small committed ledger over held-out orders as evidence.

    Each record is stamped with the order's own timestamp rather than the
    wall clock, so regenerating the sample is deterministic.
    """

    if not observable_path.exists():
        raise FileNotFoundError(
            f"Observable orders not found: {observable_path}. Run `make generate`."
        )
    orders = pd.read_csv(observable_path, parse_dates=["ordered_at"])
    test_orders = orders.loc[orders["split"] == "test"].head(limit)
    if test_orders.empty:
        raise ValueError("Observable artifact contains no test orders.")

    engine = DecisionEngine(
        bundle=ModelBundle.load(bundle_path),
        settings=load_policy_settings(),
    )
    if path.exists():
        path.unlink()
    for row in test_orders.to_dict(orient="records"):
        decision = engine.decide(row)
        record = record_from_decision(
            decision,
            row,
            recorded_at=pd.Timestamp(row["ordered_at"]).isoformat(),
        )
        append_decision(record, path)
    return path


def main() -> None:
    path = write_sample_ledger()
    records = read_ledger(path)
    broken = verify_chain(records)
    print(f"Wrote {len(records)} decision records to {path}")
    if broken is None:
        print("Hash chain verified intact.")
    else:
        raise SystemExit(f"Hash chain broken at record {broken}.")


if __name__ == "__main__":
    main()
