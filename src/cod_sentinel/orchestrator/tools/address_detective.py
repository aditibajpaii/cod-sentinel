"""Address structuring and geocoding validation."""

import json
import re
from typing import Any

import httpx

from cod_sentinel.orchestrator.credentials import AgentCredentials
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
        return {
            "structured": structured,
            "geocode": geocoded,
            "verified": verified,
            "reason_code": None if verified else "ADDRESS_UNVERIFIED",
        }

    def _extract_structure(self, raw_address: str) -> dict[str, Any]:
        if self._anthropic is not None:
            response = self._anthropic.messages.create(
                model=ADDRESS_MODEL,
                max_tokens=512,
                system=ADDRESS_DETECTIVE_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": raw_address},
                ],
            )
            text = response.content[0].text.strip()
            return json.loads(text)

        pincode_match = PINCODE_PATTERN.search(raw_address)
        return {
            "line1": raw_address.split(",")[0].strip(),
            "line2": None,
            "city": "Unknown",
            "state": "Unknown",
            "pincode": pincode_match.group(1) if pincode_match else "",
            "confidence": 0.6 if pincode_match else 0.2,
            "issues": [] if pincode_match else ["missing_pincode"],
        }

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
