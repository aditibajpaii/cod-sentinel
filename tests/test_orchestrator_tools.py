"""Tests for orchestrator tool helpers."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

httpx = pytest.importorskip("httpx")

from cod_sentinel.orchestrator.credentials import AgentCredentials
from cod_sentinel.orchestrator.tools.address_detective import (
    AddressDetectiveTool,
    _deterministic_structure,
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
    """A failing live geocode must mark the address unverified."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ZERO_RESULTS", "results": []})

    live = replace(credentials, live_geocode=True)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = AddressDetectiveTool(live, client=client, anthropic_client=None)
    result = tool.run("Flat 12, Karol Bagh, Delhi 110005")
    assert result["verified"] is False
    assert result["reason_code"] == "ADDRESS_UNVERIFIED"


def test_simulated_geocode_still_rejects_a_bogus_address(
    credentials: AgentCredentials,
) -> None:
    """Simulated mode must discriminate, not rubber-stamp every address."""

    tool = AddressDetectiveTool(credentials, client=MagicMock(), anthropic_client=None)

    good = tool.run("Flat 12, Karol Bagh, New Delhi 110005")
    bogus = tool.run("near the big tree, gali no 4")

    assert good["verified"] is True
    assert good["geocode"]["simulated"] is True
    assert bogus["verified"] is False
    assert bogus["reason_code"] == "ADDRESS_UNVERIFIED"


def test_deliverability_score_separates_complete_from_landmark_only(
    credentials: AgentCredentials,
) -> None:
    """A landmark-only address must score far below a complete one."""

    tool = AddressDetectiveTool(credentials, client=MagicMock(), anthropic_client=None)

    complete = tool.run("Flat 12, Karol Bagh, New Delhi 110005")
    landmark = tool.run("Opposite Hanuman Temple, near electric pole, Sharma Colony")

    assert complete["deliverability_score"] > 0.6
    assert complete["issues"] == []
    assert landmark["deliverability_score"] < 0.2
    assert "missing_pincode" in landmark["issues"]
    assert "no_house_or_plot_number" in landmark["issues"]


def test_deterministic_parser_extracts_city_and_pincode() -> None:
    """The fallback parser must not report every address as city-less."""

    parsed = _deterministic_structure("Plot 5, Sector 18, Noida, Uttar Pradesh 201301")

    assert parsed["pincode"] == "201301"
    assert parsed["city"] == "Noida"
    assert parsed["state"] == "Uttar Pradesh"
    assert parsed["issues"] == []


def test_issue_list_has_no_duplicates(credentials: AgentCredentials) -> None:
    tool = AddressDetectiveTool(credentials, client=MagicMock(), anthropic_client=None)

    issues = tool.run("just a vague landmark")["issues"]

    assert len(issues) == len(set(issues))


def test_dealmaker_ignores_a_hallucinated_amount(
    credentials: AgentCredentials,
) -> None:
    """A model-supplied amount must never reach the payment link."""

    anthropic = MagicMock()
    anthropic.messages.create.return_value.content = [
        MagicMock(text='{"amount_paise": 1, "description": "Pay now"}')
    ]
    tool = DynamicDealmakerTool(
        credentials,
        client=MagicMock(),
        anthropic_client=anthropic,
    )

    result = tool.run(
        order_id="ORD-1",
        order_value=1_000.0,
        agreed_discount_rate=0.02,
        max_prepaid_discount_rate=0.05,
        buyer_phone="+919876543210",
    )

    assert result["status"] == "CREATED"
    assert result["amount_paise"] == 98_000  # 1000 * (1 - 0.02) * 100, not 1


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
