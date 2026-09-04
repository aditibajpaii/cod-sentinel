import cod_sentinel

from cod_sentinel.configuration import DEFAULT_CONFIG, ProjectConfig, SplitSizes


def test_package_exposes_project_version() -> None:
    assert cod_sentinel.__version__ == "0.1.0"


def test_default_split_accounts_for_every_order() -> None:
    assert DEFAULT_CONFIG.dataset_size == 20_000
    assert DEFAULT_CONFIG.splits == SplitSizes(
        train=13_000,
        calibration=2_000,
        validation=2_000,
        test=3_000,
    )
    assert DEFAULT_CONFIG.splits.total == DEFAULT_CONFIG.dataset_size


def test_config_rejects_incomplete_split() -> None:
    config = ProjectConfig(
        dataset_size=20_000,
        splits=SplitSizes(
            train=12_000,
            calibration=2_000,
            validation=2_000,
            test=3_000,
        ),
    )

    try:
        config.validate()
    except ValueError as error:
        assert "does not match dataset size" in str(error)
    else:
        raise AssertionError("Expected incomplete temporal split to be rejected.")
