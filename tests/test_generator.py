import pandas as pd
from pandas.testing import assert_frame_equal

from cod_sentinel.configuration import ProjectConfig, SplitSizes
from cod_sentinel.features import ORACLE_DENYLIST
from cod_sentinel.generator import ARCHETYPES, generate_synthetic_world


def test_default_world_has_exact_size_and_separate_oracle(synthetic_world) -> None:
    observable, oracle, metadata = synthetic_world

    assert len(observable) == 20_000
    assert len(oracle) == 20_000
    assert metadata["dataset_size"] == 20_000
    assert not ORACLE_DENYLIST.intersection(observable.columns)
    assert {"cod_rto", "otp_completed", "prepaid_converted"}.issubset(
        oracle.columns
    )


def test_world_contains_repeated_customers_and_all_archetypes(
    synthetic_world,
) -> None:
    observable, oracle, _ = synthetic_world

    assert observable["customer_id"].nunique() < len(observable)
    assert set(oracle["archetype"]) == set(ARCHETYPES)


def test_orders_are_chronological_and_unique(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world

    ordered_at = pd.to_datetime(observable["ordered_at"])
    assert ordered_at.is_monotonic_increasing
    assert observable["order_id"].is_unique
    assert oracle["order_id"].is_unique
    assert observable["order_id"].tolist() == oracle["order_id"].tolist()


def test_outcomes_are_observed_after_order_time(synthetic_world) -> None:
    observable, oracle, _ = synthetic_world
    timing = observable[["order_id", "ordered_at"]].merge(
        oracle[["order_id", "outcome_observed_at"]],
        on="order_id",
        validate="one_to_one",
    )

    assert (
        pd.to_datetime(timing["outcome_observed_at"])
        >= pd.to_datetime(timing["ordered_at"])
    ).all()


def test_potential_outcomes_and_probabilities_are_valid(synthetic_world) -> None:
    _, oracle, _ = synthetic_world

    binary_columns = (
        "cod_rto",
        "otp_completed",
        "otp_rto_if_completed",
        "prepaid_converted",
        "prepaid_failed_delivery",
    )
    probability_columns = tuple(
        column for column in oracle.columns if column.startswith("oracle_p_")
    )
    for column in binary_columns:
        assert set(oracle[column]).issubset({0, 1})
    for column in probability_columns:
        assert oracle[column].between(0.0, 1.0).all()


def test_generator_is_deterministic_for_same_seed() -> None:
    config = ProjectConfig(
        random_seed=7,
        dataset_size=100,
        splits=SplitSizes(train=60, calibration=15, validation=10, test=15),
    )

    first = generate_synthetic_world(config)
    second = generate_synthetic_world(config)

    assert_frame_equal(first[0], second[0])
    assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]
