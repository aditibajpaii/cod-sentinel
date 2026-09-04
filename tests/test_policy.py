from dataclasses import replace

import pytest

from cod_sentinel.configuration import (
    PolicySettings,
    load_policy_settings,
)
from cod_sentinel.policy import (
    REVIEW,
    DecisionEngine,
    decide_from_probabilities,
)
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


def _probabilities() -> dict[str, float]:
    return {
        "cod_rto": 0.80,
        "otp_completion": 0.90,
        "otp_rto": 0.10,
        "prepaid_conversion": 0.50,
        "prepaid_failure": 0.05,
    }


def test_policy_selects_highest_expected_contribution(
    settings: PolicySettings,
) -> None:
    decision = decide_from_probabilities(
        order_id="ORDER-1",
        order_value=1_000.0,
        probability_values=_probabilities(),
        settings=settings,
    )

    assert decision.selected_action == "OTP"
    assert decision.requires_review is False
    assert decision.expected_contributions["OTP"] == pytest.approx(243.0)
    assert "OTP_HIGHEST_EXPECTED_CONTRIBUTION" in decision.reason_codes


def test_policy_ties_prefer_lower_friction_action(
    settings: PolicySettings,
) -> None:
    tied_settings = replace(
        settings,
        merchant_economics=replace(
            settings.merchant_economics,
            gross_margin_rate=1.0,
        ),
    )
    probabilities = {
        "cod_rto": 0.0,
        "otp_completion": 1.0,
        "otp_rto": 0.0,
        "prepaid_conversion": 1.0,
        "prepaid_failure": 0.0,
    }

    decision = decide_from_probabilities(
        order_id="ORDER-TIE",
        order_value=500.0,
        probability_values=probabilities,
        settings=tied_settings,
    )

    assert decision.selected_action == "COD"


@pytest.mark.parametrize(
    ("probabilities", "reason"),
    [
        ({"cod_rto": 0.2}, "MISSING_REQUIRED_PROBABILITY"),
        (
            {
                **_probabilities(),
                "cod_rto": float("nan"),
            },
            "SCHEMA_VALIDATION_FAILED",
        ),
        (
            {
                **_probabilities(),
                "otp_completion": 1.2,
            },
            "SCHEMA_VALIDATION_FAILED",
        ),
    ],
)
def test_invalid_probabilities_fall_back_to_review(
    probabilities: dict[str, float],
    reason: str,
    settings: PolicySettings,
) -> None:
    decision = decide_from_probabilities(
        order_id="ORDER-BAD",
        order_value=1_000.0,
        probability_values=probabilities,
        settings=settings,
    )

    assert decision.selected_action == REVIEW
    assert decision.reason_codes == (reason,)
    assert decision.expected_contributions is None


def test_decision_is_deterministic(settings: PolicySettings) -> None:
    arguments = {
        "order_id": "ORDER-STABLE",
        "order_value": 1_250.0,
        "probability_values": _probabilities(),
        "settings": settings,
    }

    first = decide_from_probabilities(**arguments)
    second = decide_from_probabilities(**arguments)

    assert first == second
    assert len(first.decision_id) == 20


def test_missing_model_artifact_falls_back_to_review(
    settings: PolicySettings,
) -> None:
    decision = DecisionEngine(bundle=None, settings=settings).decide(
        {"order_id": "ORDER-NO-MODEL"}
    )

    assert decision.selected_action == REVIEW
    assert decision.reason_codes == ("MODEL_ARTIFACT_UNAVAILABLE",)


class _BrokenBundle:
    def predict_all(self, features):
        raise ValueError("corrupt artifact")


def test_corrupt_model_artifact_falls_back_to_review(
    settings: PolicySettings,
    synthetic_world,
) -> None:
    observable, _, _ = synthetic_world
    row = observable.iloc[0].to_dict()

    decision = DecisionEngine(
        bundle=_BrokenBundle(),
        settings=settings,
    ).decide(row)

    assert decision.selected_action == REVIEW
    assert decision.reason_codes == ("MODEL_INFERENCE_FAILED",)


def test_default_policy_config_loads_complete_tie_break() -> None:
    loaded = load_policy_settings()

    assert loaded.tie_break_order == (Action.COD, Action.OTP, Action.PREPAID)
    assert loaded.config_version == "merchant-economics-v1"
