import pytest

from cod_sentinel.generator import generate_synthetic_world


@pytest.fixture(scope="session")
def synthetic_world():
    return generate_synthetic_world()
