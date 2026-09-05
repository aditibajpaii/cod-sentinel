"""Address structuring and geocoding validation."""

import json
import re
from typing import Any

import httpx

from cod_sentinel.orchestrator.credentials import GEOCODE, AgentCredentials
from cod_sentinel.orchestrator.models import ADDRESS_MODEL
from cod_sentinel.orchestrator.prompts import ADDRESS_DETECTIVE_SYSTEM_PROMPT

LANDMARK_KEYWORDS = (
    "near",
    "opposite",
    "landmark",
    "gali",
    "lane",
    "behind",
    "next to",
    "saamne",
    "ke paas",
)
HOUSE_PATTERN = re.compile(r"\b(\d+[a-zA-Z]?|plot|flat|house|h\.?\s*no)\b", re.I)
PINCODE_PATTERN = re.compile(r"\b(\d{6})\b")


def address_needs_structuring(raw_address: str) -> bool:
    """Cheap heuristic pre-check before calling the address model."""

    text = raw_address.strip().lower()
    if not text:
        return True
    if not PINCODE_PATTERN.search(text):
        return True
    if any(keyword in text for keyword in LANDMARK_KEYWORDS):
        return True
    if not HOUSE_PATTERN.search(text):
        return True
    return False


def _deterministic_structure(raw_address: str) -> dict[str, Any]:
    """Parse an address without a model, as the fallback and the stub path.

    Indian addresses are usually comma-separated with the pincode trailing the
    state, so the segment holding the pincode gives the state and the segment
    before it gives the city.
    """

    parts = [part.strip() for part in raw_address.split(",") if part.strip()]
    pincode_match = PINCODE_PATTERN.search(raw_address)
    pincode = pincode_match.group(1) if pincode_match else ""

    state = ""
    city = ""
    if pincode:
        for index, part in enumerate(parts):
            if pincode in part:
                state = part.replace(pincode, "").strip(" ,-")
                if index > 0:
                    city = parts[index - 1]
                break
    if not city and len(parts) > 1:
        city = parts[-2] if pincode else parts[-1]

    issues: list[str] = []
    if not pincode:
        issues.append("missing_pincode")
    if not city:
        issues.append("missing_city")

    confidence = 0.2
    if pincode:
        confidence = 0.7 if city else 0.6

    return {
        "line1": parts[0] if parts else raw_address.strip(),
        "line2": parts[1] if len(parts) > 2 else None,
        "city": city or "Unknown",
        "state": state or "Unknown",
        "pincode": pincode,
        "confidence": confidence,
        "issues": issues,
    }


def deliverability_score(
    structured: dict[str, Any],
    geocoded: dict[str, Any],
) -> tuple[float, list[str]]:
    """Score how deliverable a parsed address is, and say what is wrong.

    Starts from the parser's own confidence and deducts for each missing or
    unresolvable component, so a landmark-only address scores far below a
    complete one even when both parse without error.
    """

    issues: list[str] = [str(issue) for issue in structured.get("issues") or []]
    try:
        score = float(structured.get("confidence", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = min(max(score, 0.0), 1.0)

    pincode = str(structured.get("pincode") or "")
    if not PINCODE_PATTERN.fullmatch(pincode):
        score -= 0.35
        if "missing_pincode" not in issues:
            issues.append("missing_or_invalid_pincode")

    line1 = str(structured.get("line1") or "")
    if not HOUSE_PATTERN.search(line1):
        score -= 0.20
        issues.append("no_house_or_plot_number")

    if not structured.get("city") or str(structured.get("city")).lower() == "unknown":
        score -= 0.10
        if "missing_city" not in issues:
            issues.append("missing_city")

    status = geocoded.get("status")
    if status != "OK":
        score -= 0.25
        issues.append(f"geocode_{str(status or 'failed').lower()}")
    elif geocoded.get("partial_match"):
        score -= 0.15
        issues.append("geocode_partial_match")

    return round(min(max(score, 0.0), 1.0), 3), list(dict.fromkeys(issues))


class AddressDetectiveTool:
    """Extract structured address and validate via Google Geocoding."""

    def __init__(
        self,
        credentials: AgentCredentials,
        *,
        client: httpx.Client | None = None,
        anthropic_client: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._http = client or httpx.Client(timeout=30.0)
        self._anthropic = anthropic_client

    def run(self, raw_address: str) -> dict[str, Any]:
        structured = self._extract_structure(raw_address)
        geocoded = self._geocode(structured)
        verified = (
            geocoded.get("status") == "OK"
            and structured.get("confidence", 0.0) >= 0.5
            and not geocoded.get("partial_match", True)
        )
        score, issues = deliverability_score(structured, geocoded)
        return {
            "structured": structured,
            "geocode": geocoded,
            "verified": verified,
            "deliverability_score": score,
            "issues": issues,
            "reason_code": None if verified else "ADDRESS_UNVERIFIED",
        }

    def _extract_structure(self, raw_address: str) -> dict[str, Any]:
        if self._anthropic is not None:
            try:
                response = self._anthropic.messages.create(
                    model=ADDRESS_MODEL,
                    max_tokens=512,
                    system=ADDRESS_DETECTIVE_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": raw_address},
                    ],
                )
                parsed = json.loads(response.content[0].text.strip())
            except (
                json.JSONDecodeError,
                AttributeError,
                IndexError,
                KeyError,
                TypeError,
            ):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("pincode"):
                parsed.setdefault("confidence", 0.5)
                return parsed
            # Malformed or incomplete model output falls through to the
            # deterministic parse rather than failing the whole address check.

        return _deterministic_structure(raw_address)

    def _geocode(self, structured: dict[str, Any]) -> dict[str, Any]:
        query = ", ".join(
            part
            for part in (
                structured.get("line1"),
                structured.get("line2"),
                structured.get("city"),
                structured.get("state"),
                structured.get("pincode"),
            )
            if part
        )
        if not self._credentials.is_live(GEOCODE):
            return self._simulated_geocode(structured, query)
        response = self._http.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={
                "address": query,
                "key": self._credentials.google_maps_api_key,
                "region": "in",
            },
        )
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status", "UNKNOWN")
        if status != "OK" or not payload.get("results"):
            return {"status": status, "partial_match": True}
        result = payload["results"][0]
        location = result["geometry"]["location"]
        return {
            "status": status,
            "formatted_address": result.get("formatted_address"),
            "lat": location.get("lat"),
            "lng": location.get("lng"),
            "partial_match": bool(result.get("partial_match", False)),
        }

    @staticmethod
    def _simulated_geocode(structured: dict[str, Any], query: str) -> dict[str, Any]:
        """Deterministic stand-in for the Geocoding API.

        Resolution is driven by the same signals the live path depends on — a
        valid six-digit pincode and a house/plot token — so a bogus address
        still fails verification when no credential is present.
        """

        pincode = str(structured.get("pincode") or "")
        has_pincode = bool(PINCODE_PATTERN.fullmatch(pincode))
        has_house = bool(HOUSE_PATTERN.search(query))
        if not has_pincode:
            return {"status": "ZERO_RESULTS", "partial_match": True, "simulated": True}
        return {
            "status": "OK",
            "formatted_address": query,
            "lat": None,
            "lng": None,
            "partial_match": not has_house,
            "simulated": True,
        }
