"""Environment credential validation and per-service live/simulated routing.

Only ``ANTHROPIC_API_KEY`` is mandatory. The three external services default to
simulated backends so a clean clone runs end to end with one credential; each
switches to live calls via its own environment flag, with no code change.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

SIMULATED = "SIMULATED"
LIVE = "LIVE"

GEOCODE = "geocode"
WHATSAPP = "whatsapp"
RAZORPAY = "razorpay"
SERVICES = (GEOCODE, WHATSAPP, RAZORPAY)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUTHY


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
    live_geocode: bool = False
    live_whatsapp: bool = False
    live_razorpay: bool = False

    def is_live(self, service: str) -> bool:
        if service == GEOCODE:
            return self.live_geocode
        if service == WHATSAPP:
            return self.live_whatsapp
        if service == RAZORPAY:
            return self.live_razorpay
        raise ValueError(f"Unknown service {service!r}.")

    def mode(self, service: str) -> str:
        return LIVE if self.is_live(service) else SIMULATED

    @property
    def modes(self) -> dict[str, str]:
        return {service: self.mode(service) for service in SERVICES}

    @property
    def missing(self) -> list[str]:
        """Credentials required by the current mode, and absent.

        A service running simulated is not missing anything.
        """

        names: list[str] = []
        if not self.anthropic_api_key:
            names.append("ANTHROPIC_API_KEY")
        if self.live_geocode and not self.google_maps_api_key:
            names.append("GOOGLE_MAPS_API_KEY")
        if self.live_whatsapp:
            if not self.twilio_account_sid:
                names.append("TWILIO_ACCOUNT_SID")
            if not self.twilio_auth_token:
                names.append("TWILIO_AUTH_TOKEN")
            if not self.twilio_whatsapp_from:
                names.append("TWILIO_WHATSAPP_FROM")
        if self.live_razorpay:
            if not self.razorpay_key_id:
                names.append("RAZORPAY_KEY_ID")
            if not self.razorpay_key_secret:
                names.append("RAZORPAY_KEY_SECRET")
        return names

    @property
    def complete(self) -> bool:
        """True when every credential the current mode actually needs is present."""

        return not self.missing


def load_credentials() -> AgentCredentials:
    threshold_raw = os.getenv("AGENT_COD_RTO_THRESHOLD", "0.50")
    try:
        threshold = float(threshold_raw)
    except ValueError as error:
        raise ValueError("AGENT_COD_RTO_THRESHOLD must be numeric.") from error
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("AGENT_COD_RTO_THRESHOLD must be between 0 and 1.")

    live_all = _flag("LIVE_MODE")
    return AgentCredentials(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", "").strip(),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
        twilio_whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM", "").strip(),
        razorpay_key_id=os.getenv("RAZORPAY_KEY_ID", "").strip(),
        razorpay_key_secret=os.getenv("RAZORPAY_KEY_SECRET", "").strip(),
        cod_rto_threshold=threshold,
        live_geocode=_flag("LIVE_GEOCODE", default=live_all),
        live_whatsapp=_flag("LIVE_WHATSAPP", default=live_all),
        live_razorpay=_flag("LIVE_RAZORPAY", default=live_all),
    )
