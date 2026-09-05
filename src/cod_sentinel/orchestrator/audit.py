"""Append-only agent audit ledger (sink, not a runtime dependency)."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from cod_sentinel.configuration import ARTIFACTS_DIR

AGENT_AUDIT_PATH = ARTIFACTS_DIR / "agent_audit.jsonl"
RECORD_SCHEMA_VERSION = "agent-audit-v1"
GENESIS_HASH = "0" * 64


def _canonical(record: Mapping[str, object]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def record_hash(record: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


def read_audit(path: Path = AGENT_AUDIT_PATH) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_audit_record(
    record: Mapping[str, object],
    path: Path = AGENT_AUDIT_PATH,
) -> dict[str, object]:
    existing = read_audit(path)
    previous_hash = record_hash(existing[-1]) if existing else GENESIS_HASH
    linked = {
        **dict(record),
        "schema_version": RECORD_SCHEMA_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "previous_hash": previous_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(linked) + "\n")
    return linked


def append_step(
    *,
    order_id: str,
    step: int,
    tool: str,
    model: str | None,
    input_summary: str,
    output_summary: str,
    reason_codes: tuple[str, ...] = (),
    path: Path = AGENT_AUDIT_PATH,
) -> dict[str, object]:
    payload = {
        "order_id": order_id,
        "step": step,
        "tool": tool,
        "model": model,
        "input_hash": hashlib.sha256(input_summary.encode("utf-8")).hexdigest()[:16],
        "input_summary": input_summary,
        "output_summary": output_summary,
        "reason_codes": list(reason_codes),
    }
    return append_audit_record(payload, path)
