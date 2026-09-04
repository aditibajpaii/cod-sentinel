"""Deterministic risk-to-action economic policy."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

import pandas as pd

from cod_sentinel.configuration import PolicySettings
from cod_sentinel.economics import expected_contribution
from cod_sentinel.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from cod_sentinel.models import ModelBundle
from cod_sentinel.schemas import Action, ActionProbabilities, OrderEconomics
from cod_sentinel.versioning import (
    CALIBRATION_VERSION,
    ECONOMICS_VERSION,
    MODEL_VERSION,
    POLICY_VERSION,
)

REVIEW = "REVIEW"
REQUIRED_PROBABILITIES = (
    "cod_rto",
    "otp_completion",
    "otp_rto",
    "prepaid_conversion",
    "prepaid_failure",
)


@dataclass(frozen=True)
class Decision:
    decision_id: str
    order_id: str
    selected_action: str
    expected_contributions: dict[str, float] | None
    probabilities: dict[str, float] | None
    reason_codes: tuple[str, ...]
    versions: dict[str, str]
    config_version: str

    @property
    def requires_review(self) -> bool:
        return self.selected_action == REVIEW

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _versions() -> dict[str, str]:
    return {
        "model": MODEL_VERSION,
        "calibration": CALIBRATION_VERSION,
        "economics": ECONOMICS_VERSION,
        "policy": POLICY_VERSION,
    }


def _stable_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _valid_order_id(value: object) -> bool:
    if not pd.api.types.is_scalar(value):
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        return False
    return bool(str(value).strip())


def _runtime_row_is_valid(order: Mapping[str, object]) -> bool:
    for name in CATEGORICAL_FEATURES:
        value = order.get(name)
        if not pd.api.types.is_scalar(value) or not isinstance(value, str):
            return False
        if not value.strip():
            return False
    for name in NUMERIC_FEATURES:
        value = order.get(name)
        if not pd.api.types.is_scalar(value) or isinstance(value, (str, bytes)):
            return False
        try:
            if not isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def review_decision(
    order_id: object,
    reason_code: str,
    settings: PolicySettings,
) -> Decision:
    """Return a deterministic non-economic REVIEW fallback."""

    normalized_order_id = str(order_id) if _valid_order_id(order_id) else "UNKNOWN"
    payload = {
        "order_id": normalized_order_id,
        "selected_action": REVIEW,
        "reason_code": reason_code,
        "versions": _versions(),
        "config_version": settings.config_version,
    }
    return Decision(
        decision_id=_stable_id(payload),
        order_id=normalized_order_id,
        selected_action=REVIEW,
        expected_contributions=None,
        probabilities=None,
        reason_codes=(reason_code,),
        versions=_versions(),
        config_version=settings.config_version,
    )


def decide_from_probabilities(
    *,
    order_id: object,
    order_value: object,
    probability_values: Mapping[str, object],
    settings: PolicySettings,
) -> Decision:
    """Choose the highest-EV economic action from validated probabilities."""

    if not _valid_order_id(order_id):
        return review_decision(order_id, "MISSING_ORDER_ID", settings)
    try:
        probabilities = {
            name: float(probability_values[name])
            for name in REQUIRED_PROBABILITIES
        }
    except KeyError:
        return review_decision(
            order_id,
            "MISSING_REQUIRED_PROBABILITY",
            settings,
        )
    except (TypeError, ValueError):
        return review_decision(order_id, "INVALID_PROBABILITY", settings)

    try:
        order = OrderEconomics(order_value=float(order_value))
        action_probabilities = {
            Action.COD: ActionProbabilities.for_cod(probabilities["cod_rto"]),
            Action.OTP: ActionProbabilities(
                conversion_probability=probabilities["otp_completion"],
                delivery_probability=1.0 - probabilities["otp_rto"],
            ),
            Action.PREPAID: ActionProbabilities(
                conversion_probability=probabilities["prepaid_conversion"],
                delivery_probability=1.0 - probabilities["prepaid_failure"],
            ),
        }
        contributions = {
            action: expected_contribution(
                action,
                order,
                settings.merchant_economics,
                action_probabilities[action],
            )
            for action in Action
        }
    except (TypeError, ValueError):
        return review_decision(order_id, "SCHEMA_VALIDATION_FAILED", settings)

    rank = {action: index for index, action in enumerate(settings.tie_break_order)}
    selected = max(
        Action,
        key=lambda action: (contributions[action], -rank[action]),
    )
    reasons = [f"{selected.value}_HIGHEST_EXPECTED_CONTRIBUTION"]
    if probabilities["cod_rto"] >= 0.50:
        reasons.append("HIGH_PREDICTED_COD_RTO_RISK")
    if probabilities["otp_completion"] < 0.60:
        reasons.append("OTP_COMPLETION_FRICTION_HIGH")
    if probabilities["prepaid_conversion"] < 0.60:
        reasons.append("PREPAID_CONVERSION_FRICTION_HIGH")

    serialized_contributions = {
        action.value: float(contributions[action]) for action in Action
    }
    payload = {
        "order_id": str(order_id),
        "order_value": order.order_value,
        "probabilities": probabilities,
        "economics": asdict(settings.merchant_economics),
        "contributions": serialized_contributions,
        "selected_action": selected.value,
        "versions": _versions(),
        "config_version": settings.config_version,
    }
    return Decision(
        decision_id=_stable_id(payload),
        order_id=str(order_id),
        selected_action=selected.value,
        expected_contributions=serialized_contributions,
        probabilities=probabilities,
        reason_codes=tuple(reasons),
        versions=_versions(),
        config_version=settings.config_version,
    )


@dataclass
class DecisionEngine:
    bundle: ModelBundle | None
    settings: PolicySettings

    def decide(self, order: Mapping[str, object]) -> Decision:
        order_id = order.get("order_id")
        if not _valid_order_id(order_id):
            return review_decision(
                order_id,
                "MISSING_ORDER_ID",
                self.settings,
            )
        if self.bundle is None:
            return review_decision(
                order_id,
                "MODEL_ARTIFACT_UNAVAILABLE",
                self.settings,
            )
        if not _runtime_row_is_valid(order):
            return review_decision(
                order_id,
                "SCHEMA_VALIDATION_FAILED",
                self.settings,
            )
        try:
            features = pd.DataFrame([dict(order)])
            predictions = {
                name: float(values[0])
                for name, values in self.bundle.predict_all(features).items()
            }
        except Exception:
            return review_decision(
                order_id,
                "MODEL_INFERENCE_FAILED",
                self.settings,
            )
        return decide_from_probabilities(
            order_id=order_id,
            order_value=order.get("order_value"),
            probability_values=predictions,
            settings=self.settings,
        )
