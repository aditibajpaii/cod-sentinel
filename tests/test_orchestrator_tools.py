"""Tests for orchestrator tool helpers."""

from unittest.mock import MagicMock

import pytest

httpx = pytest.importorskip("httpx")

from cod_sentinel.orchestrator.credentials import AgentCredentials
from cod_sentinel.orchestrator.tools.address_detective import (
    AddressDetectiveTool,
    address_needs_structuring,
)
from cod_sentinel.orchestrator.tools.dynamic_dealmaker import DynamicDealmakerTool


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


@pytest.mark.parametrize(
    ("address", "expected"),
    [
        ("near metro station karol bagh", True),
        ("Flat 12, 110005 Karol Bagh New Delhi", False),
        ("Plot 5, Sector 18, Noida, Uttar Pradesh 201301", False),
        ("some vague landmark only", True),
    ],
)
def test_address_needs_structuring_heuristics(address: str, expected: bool) -> None:
    assert address_needs_structuring(address) is expected


def test_dynamic_dealmaker_rejects_over_cap_discount(credentials: AgentCredentials) -> None:
    tool = DynamicDealmakerTool(credentials, client=MagicMock())
    result = tool.run(
        order_id="ORD-1",
        order_value=1_000.0,
        agreed_discount_rate=0.10,
        max_prepaid_discount_rate=0.02,
        buyer_phone="+919876543210",
    )
    assert result["status"] == "REJECTED"
    assert result["reason_code"] == "DISCOUNT_EXCEEDS_MARGIN"


def test_address_detective_geocode_failure_marks_unverified(
    credentials: AgentCredentials,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ZERO_RESULTS", "results": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = AddressDetectiveTool(credentials, client=client, anthropic_client=None)
    result = tool.run("Flat 12, Karol Bagh, Delhi 110005")
    assert result["verified"] is False
    assert result["reason_code"] == "ADDRESS_UNVERIFIED"


def test_load_credentials_missing_returns_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_WHATSAPP_FROM", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)

    from cod_sentinel.orchestrator.credentials import load_credentials

    creds = load_credentials()
    assert creds.complete is False
    assert "ANTHROPIC_API_KEY" in creds.missing
