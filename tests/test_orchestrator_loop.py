"""Tests for orchestrator sequencing with mocked external APIs."""

from unittest.mock import MagicMock

import pytest

httpx = pytest.importorskip("httpx")

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.orchestrator.credentials import AgentCredentials
from cod_sentinel.orchestrator.runner import OrchestratorRunner
from cod_sentinel.orchestrator.schemas import CheckoutEvent
from cod_sentinel.policy import DecisionEngine, REVIEW


@pytest.fixture
def credentials() -> AgentCredentials:
    return AgentCredentials(
        anthropic_api_key="test-key",
        google_maps_api_key="maps-key",
        twilio_account_sid="sid",
        twilio_auth_token="token",
        twilio_whatsapp_from="whatsapp:+14155238886",
        razorpay_key_id="rzp_test",
        razorpay_key_secret="secret",
        cod_rto_threshold=0.50,
    )


@pytest.fixture
def engine() -> DecisionEngine:
    bundle = ModelBundle.load(MODEL_BUNDLE_PATH)
    return DecisionEngine(bundle=bundle, settings=load_policy_settings())


@pytest.fixture
def sample_event(engine: DecisionEngine) -> CheckoutEvent:
    import pandas as pd

    from cod_sentinel.features import RUNTIME_FEATURES
    from cod_sentinel.generator import OBSERVABLE_PATH

    orders = pd.read_csv(OBSERVABLE_PATH)
    row = orders.loc[orders["split"] == "test"].iloc[0]
    features = {name: row[name] for name in RUNTIME_FEATURES if name in row}
    return CheckoutEvent(
        order_id=str(row["order_id"]),
        payment_method="COD",
        raw_address="Flat 12, Karol Bagh, New Delhi 110005",
        buyer_phone="+919876543210",
        order_features=features,
    )


def _mock_http() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if "geocode" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "formatted_address": "Karol Bagh, Delhi",
                            "geometry": {"location": {"lat": 28.6, "lng": 77.2}},
                            "partial_match": False,
                        }
                    ],
                },
            )
        if "twilio.com" in str(request.url):
            return httpx.Response(200, json={"sid": "SM123", "status": "queued"})
        if "razorpay.com" in str(request.url):
            return httpx.Response(
                200,
                json={"id": "plink_123", "short_url": "https://rzp.io/i/test"},
            )
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_runner_returns_review_when_credentials_missing(
    engine: DecisionEngine,
    sample_event: CheckoutEvent,
) -> None:
    creds = AgentCredentials(
        anthropic_api_key="",
        google_maps_api_key="",
        twilio_account_sid="",
        twilio_auth_token="",
        twilio_whatsapp_from="",
        razorpay_key_id="",
        razorpay_key_secret="",
        cod_rto_threshold=0.50,
    )
    runner = OrchestratorRunner(engine=engine, credentials=creds)
    result = runner.run(sample_event)
    assert result.final_status == "REVIEW"
    assert result.selected_action == REVIEW
    assert "AGENT_CREDENTIALS_UNAVAILABLE" in result.reason_codes


def test_runner_fallback_when_buyer_declines(
    engine: DecisionEngine,
    sample_event: CheckoutEvent,
    credentials: AgentCredentials,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_economics(self, order):  # noqa: ANN001
        decision = engine.decide(order)
        return {
            "decision": decision.to_dict(),
            "cod_rto": 0.75,
            "unit_economics": {
                "order_value": float(order["order_value"]),
                "expected_margin": 300.0,
                "forward_shipping_cost": 60.0,
                "reverse_shipping_cost": 80.0,
                "cod_handling_cost": 20.0,
                "expected_contributions": decision.expected_contributions,
            },
            "max_prepaid_discount_rate": 0.05,
            "fallback_action": decision.selected_action,
        }

    monkeypatch.setattr(
        "cod_sentinel.orchestrator.tools.economics.EconomicsTool.run",
        fake_economics,
    )
    monkeypatch.setattr(
        "cod_sentinel.orchestrator.tools.call_e_negotiator.CallENegotiatorTool.start",
        lambda self, **kwargs: {"status": "DECLINED", "agreed_discount_rate": None},
    )
    runner = OrchestratorRunner(
        engine=engine,
        credentials=credentials,
        http_client=_mock_http(),
        anthropic_client=MagicMock(),
    )
    result = runner.run(sample_event)
    assert result.final_status == "FALLBACK_ACTION"
    assert result.steps[0].tool == "economics_tool"


def test_checkout_event_rejects_oracle_fields(engine: DecisionEngine) -> None:
    with pytest.raises(ValueError, match="Oracle field"):
        CheckoutEvent(
            order_id="ORD-1",
            payment_method="COD",
            raw_address="Flat 1, Delhi 110001",
            buyer_phone="+919876543210",
            order_features={"latent_address_quality": 0.5},
        )
