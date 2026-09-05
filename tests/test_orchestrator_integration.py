"""Opt-in live integration smoke tests for the agent orchestrator."""

import os

import pytest

pytestmark = pytest.mark.integration


def _credentials_present() -> bool:
    required = (
        "ANTHROPIC_API_KEY",
        "GOOGLE_MAPS_API_KEY",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_WHATSAPP_FROM",
        "RAZORPAY_KEY_ID",
        "RAZORPAY_KEY_SECRET",
    )
    return all(os.getenv(name, "").strip() for name in required)


@pytest.mark.skipif(not _credentials_present(), reason="Live agent credentials not configured.")
def test_live_credentials_load() -> None:
    from cod_sentinel.orchestrator.credentials import load_credentials

    creds = load_credentials()
    assert creds.complete
