import pytest

from cod_sentinel.configuration import PolicySettings
from cod_sentinel.ledger import (
    GENESIS_HASH,
    append_decision,
    read_ledger,
    record_from_decision,
    replay,
    verify_chain,
    verify_replay,
)
from cod_sentinel.policy import REVIEW, DecisionEngine
from cod_sentinel.schemas import Action, MerchantEconomics


@pytest.fixture
def settings() -> PolicySettings:
    return PolicySettings(
        merchant_economics=MerchantEconomics(
            gross_margin_rate=0.30,
            forward_shipping_cost=0.0,
            reverse_shipping_cost=0.0,
            packaging_cost=0.0,
        ),
        tie_break_order=(Action.COD, Action.OTP, Action.PREPAID),
        config_version="test-v1",
    )


class _StubBundle:
    """Deterministic stand-in for a trained bundle."""

    def predict_all(self, features):
        return {
            "cod_rto": [0.80],
            "otp_completion": [0.90],
            "otp_rto": [0.10],
            "prepaid_conversion": [0.50],
            "prepaid_failure": [0.05],
        }


class _BrokenBundle:
    def predict_all(self, features):
        raise ValueError("corrupt artifact")


def _order(synthetic_world) -> dict:
    observable, _, _ = synthetic_world
    return observable.iloc[0].to_dict()


def test_recorded_decision_replays_to_identical_id(
    tmp_path,
    settings: PolicySettings,
    synthetic_world,
) -> None:
    engine = DecisionEngine(bundle=_StubBundle(), settings=settings)
    order = _order(synthetic_world)
    decision = engine.decide(order)
    path = tmp_path / "ledger.jsonl"

    append_decision(record_from_decision(decision, order), path)
    records = read_ledger(path)

    assert len(records) == 1
    assert records[0]["previous_hash"] == GENESIS_HASH
    replayed = replay(records[0], engine)
    assert replayed.decision_id == decision.decision_id
    assert replayed.selected_action == decision.selected_action

    result = verify_replay(records[0], engine)
    assert result.matches
    assert result.status == "match"


def test_ledger_records_only_runtime_features(
    tmp_path,
    settings: PolicySettings,
    synthetic_world,
) -> None:
    engine = DecisionEngine(bundle=_StubBundle(), settings=settings)
    order = _order(synthetic_world)
    order["cod_rto"] = True
    order["archetype"] = "serial_abuse"

    record = record_from_decision(engine.decide(order), order)

    assert "cod_rto" not in record["inputs"]
    assert "archetype" not in record["inputs"]


def test_editing_a_record_breaks_the_hash_chain(
    tmp_path,
    settings: PolicySettings,
    synthetic_world,
) -> None:
    engine = DecisionEngine(bundle=_StubBundle(), settings=settings)
    observable, _, _ = synthetic_world
    path = tmp_path / "ledger.jsonl"
    for index in range(3):
        order = observable.iloc[index].to_dict()
        append_decision(record_from_decision(engine.decide(order), order), path)

    records = read_ledger(path)
    assert verify_chain(records) is None

    records[0]["selected_action"] = "PREPAID"

    assert verify_chain(records) == 1


def test_review_decision_round_trips(
    tmp_path,
    settings: PolicySettings,
    synthetic_world,
) -> None:
    engine = DecisionEngine(bundle=_BrokenBundle(), settings=settings)
    order = _order(synthetic_world)
    decision = engine.decide(order)
    assert decision.selected_action == REVIEW

    path = tmp_path / "ledger.jsonl"
    append_decision(record_from_decision(decision, order), path)
    record = read_ledger(path)[0]

    assert record["probabilities"] is None
    assert record["expected_contributions"] is None
    assert verify_replay(record, engine).matches


def test_replay_reports_version_drift_not_corruption(
    tmp_path,
    settings: PolicySettings,
    synthetic_world,
) -> None:
    engine = DecisionEngine(bundle=_StubBundle(), settings=settings)
    order = _order(synthetic_world)
    record = record_from_decision(engine.decide(order), order)
    record["versions"] = {**record["versions"], "model": "outcome-models-v0"}
    record["decision_id"] = "0" * 20

    result = verify_replay(record, engine)

    assert not result.matches
    assert result.status == "version_drift"
    assert "model" in result.detail
    assert result.recorded_decision_id != result.replayed_decision_id
