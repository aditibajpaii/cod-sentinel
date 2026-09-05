"""Environment credential validation for live agent integrations."""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True, slots=True)
class AgentCredentials:
    anthropic_api_key: str
    google_maps_api_key: str
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_whatsapp_from: str
    razorpay_key_id: str
    razorpay_key_secret: str
    cod_rto_threshold: float

    @property
    def complete(self) -> bool:
        return all(
            (
                self.anthropic_api_key,
                self.google_maps_api_key,
                self.twilio_account_sid,
                self.twilio_auth_token,
                self.twilio_whatsapp_from,
                self.razorpay_key_id,
                self.razorpay_key_secret,
            )
        )

    @property
    def missing(self) -> list[str]:
        names = []
        if not self.anthropic_api_key:
            names.append("ANTHROPIC_API_KEY")
        if not self.google_maps_api_key:
            names.append("GOOGLE_MAPS_API_KEY")
        if not self.twilio_account_sid:
            names.append("TWILIO_ACCOUNT_SID")
        if not self.twilio_auth_token:
            names.append("TWILIO_AUTH_TOKEN")
        if not self.twilio_whatsapp_from:
            names.append("TWILIO_WHATSAPP_FROM")
        if not self.razorpay_key_id:
            names.append("RAZORPAY_KEY_ID")
        if not self.razorpay_key_secret:
            names.append("RAZORPAY_KEY_SECRET")
        return names


def load_credentials() -> AgentCredentials:
    threshold_raw = os.getenv("AGENT_COD_RTO_THRESHOLD", "0.50")
    try:
        threshold = float(threshold_raw)
    except ValueError as error:
        raise ValueError("AGENT_COD_RTO_THRESHOLD must be numeric.") from error
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("AGENT_COD_RTO_THRESHOLD must be between 0 and 1.")
    return AgentCredentials(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
        twilio_whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM", "").strip(),
        razorpay_key_id=os.getenv("RAZORPAY_KEY_ID", "").strip(),
        razorpay_key_secret=os.getenv("RAZORPAY_KEY_SECRET", "").strip(),
        cod_rto_threshold=threshold,
    )
