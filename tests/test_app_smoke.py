import json

import pandas as pd
import pytest

import app
from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.versioning import DATASET_VERSION, MODEL_VERSION, POLICY_VERSION


def test_app_module_imports_without_starting_server() -> None:
    assert callable(app.main)


def test_break_even_curve_clips_negative_region_for_display() -> None:
    merchant = load_policy_settings().merchant_economics
    curve = app.break_even_curve(merchant, min_value=200, max_value=1_000, step=200)
    floor = app.minimum_profitable_cod_value(merchant)

    assert 250 < floor < 350
    assert curve["break_even_display"].between(0.0, 1.0).all()
    assert bool(curve.loc[curve["order_value"] == 200, "cod_never_pays"].iloc[0])
    assert not bool(curve.loc[curve["order_value"] == 1_000, "cod_never_pays"].iloc[0])
    assert curve.loc[curve["order_value"] == 1_000, "break_even_raw"].iloc[0] > 0.4


def test_metrics_loader_reads_generated_json(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "evidence_scope": "synthetic",
                "risk_model": {},
                "policies": {},
                "sensitivity": {},
                "versions": {
                    "dataset": DATASET_VERSION,
                    "model": MODEL_VERSION,
                    "policy": POLICY_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )

    assert app.load_metrics(path)["evidence_scope"] == "synthetic"


def test_metrics_loader_reports_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation metrics not found"):
        app.load_metrics(tmp_path / "missing.json")


def test_observable_loader_parses_timestamp(tmp_path, synthetic_world) -> None:
    path = tmp_path / "observable.csv"
    orders = synthetic_world[0].iloc[[0]].copy()
    orders["split"] = "test"
    orders.to_csv(path, index=False)

    loaded = app.load_observable_orders(path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["ordered_at"])


def test_status_pill_and_stat_callout_are_reusable_html() -> None:
    pill = app._status_pill("OTP", "verify")
    stat = app._stat_callout("0.411", "Precision · frozen test")

    assert 'class="pill pill-verify"' in pill
    assert "OTP" in pill
    assert 'class="stat-value"' in stat
    assert "0.411" in stat


def test_hero_strip_uses_frozen_policy_numbers() -> None:
    metrics = {
        "policies": {
            "always_cod": {"realized_contribution_per_order": 152.78262059200006},
            "always_otp": {"realized_contribution_per_order": 242.3051690046667},
            "cod_sentinel": {
                "realized_contribution_per_order": 218.49186688000003,
                "improvement_vs_best_simple_baseline_per_order": -23.813302124666677,
            },
        }
    }

    html = app._hero_strip(metrics)

    assert "₹152.78" in html
    assert "₹218.49" in html
    assert "₹242.31" in html
    assert "₹23.81" in html
    assert "trails" in html
    assert "hero-proof-value" in html


def test_outcome_cards_use_frozen_policy_numbers() -> None:
    metrics = {
        "policies": {
            "always_otp": {"realized_contribution_per_order": 242.3051690046667},
            "cod_sentinel": {
                "realized_contribution_per_order": 218.49186688000003,
                "improvement_vs_best_simple_baseline_per_order": -23.813302124666677,
            },
        }
    }

    html = app._outcome_cards(metrics)

    assert "₹218.49" in html
    assert "₹242.31" in html
    assert "outcome-card" in html
    assert "best" in html


def _agent_credentials(**overrides):
    from cod_sentinel.orchestrator.credentials import AgentCredentials

    values = {
        "anthropic_api_key": "key",
        "google_maps_api_key": "",
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_whatsapp_from": "",
        "razorpay_key_id": "",
        "razorpay_key_secret": "",
        "cod_rto_threshold": 0.50,
    }
    values.update(overrides)
    return AgentCredentials(**values)


def test_agent_credential_strip_marks_a_missing_required_key() -> None:
    html = app._agent_credential_strip(_agent_credentials(anthropic_api_key=""))

    assert "Anthropic" in html
    assert "pill-fail" in html


def test_agent_credential_strip_reports_simulated_services_as_healthy() -> None:
    """Missing service keys are a mode, not a failure."""

    html = app._agent_credential_strip(_agent_credentials())

    assert "pill-fail" not in html
    assert "Razorpay · Simulated" in html
    assert "WhatsApp · Simulated" in html


def test_agent_credential_strip_shows_live_services() -> None:
    html = app._agent_credential_strip(
        _agent_credentials(razorpay_key_id="rzp", razorpay_key_secret="s", live_razorpay=True)
    )

    assert "Razorpay · Live" in html


def test_agent_timeline_renders_steps() -> None:
    html = app._agent_timeline(
        [{"step": 1, "tool": "economics_tool", "model": None, "output_summary": "COD"}]
    )
    assert "economics_tool" in html
    assert "console-table" in html


def test_leakage_pills_read_report_without_recomputing(tmp_path) -> None:
    path = tmp_path / "leakage_report.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "gates": {
                    "feature_allowlist_and_provenance": "passed",
                    "oracle_physical_separation": "passed",
                    "prior_history_recomputation": "passed",
                    "shuffled_label_sanity": "passed",
                },
            }
        ),
        encoding="utf-8",
    )

    html = app._leakage_pills(path)

    assert "feature allowlist" in html
    assert "pill-pass" in html
    assert "unavailable" not in html
