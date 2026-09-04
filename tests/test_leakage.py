import pytest

import cod_sentinel.leakage as leakage_module
from cod_sentinel.leakage import (
    shuffled_label_auc,
    validate_feature_contract,
    validate_physical_separation,
    validate_prior_history,
)


def test_clean_generated_world_passes_structural_gates(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world

    validate_feature_contract(observable)
    validate_physical_separation(observable, oracle)
    validate_prior_history(observable, oracle)


def test_feature_contract_rejects_injected_oracle_column(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world
    contaminated = observable.copy()
    contaminated["latent_intent"] = oracle["latent_intent"]

    with pytest.raises(ValueError, match="Oracle columns"):
        validate_feature_contract(contaminated)


def test_physical_separation_rejects_shared_split_column(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world
    contaminated = oracle.copy()
    contaminated["split"] = observable["split"]

    with pytest.raises(ValueError, match="forbidden columns"):
        validate_physical_separation(observable, contaminated)


def test_prior_history_gate_rejects_corrupted_feature(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world
    corrupted = observable.copy()
    corrupted.loc[100, "customer_prior_rto_rate"] = 0.987654

    with pytest.raises(ValueError, match="do not match strictly prior"):
        validate_prior_history(corrupted, oracle)


def test_provenance_must_cover_exact_runtime_allowlist(
    synthetic_world,
    monkeypatch,
) -> None:
    observable, _, _ = synthetic_world
    incomplete = dict(leakage_module.FEATURE_PROVENANCE)
    incomplete.pop("order_value")
    monkeypatch.setattr(leakage_module, "FEATURE_PROVENANCE", incomplete)

    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_feature_contract(observable)


def test_shuffled_labels_lose_predictive_ability(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world

    auc = shuffled_label_auc(observable, oracle, random_seed=42)

    assert 0.40 <= auc < 0.60
