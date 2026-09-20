import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import make_two_populations
from src.mortality.lee_carter import LeeCarter


@pytest.fixture(scope="session")
def populations():
    return make_two_populations(seed=3)


@pytest.fixture(scope="session")
def lee_carter(populations):
    return LeeCarter().fit(populations[0])


@pytest.fixture
def rng():
    return np.random.default_rng(123)
