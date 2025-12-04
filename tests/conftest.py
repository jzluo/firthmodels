import numpy as np
import pytest
from scipy.special import expit


def make_separation_data(seed=42, n=100):
    """Generate data with quasi-complete separation for testing."""
    rng = np.random.default_rng(seed)

    separator = rng.choice([0, 1], n, p=[0.8, 0.2])  # causes separation
    x1 = rng.standard_normal(n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.exponential(scale=2.0, size=n)

    # Generate outcome from logistic model (excluding separator)
    logit = -0.5 + 0.8 * x1 - 0.6 * x2 + 0.3 * x3
    prob = expit(logit)
    y = rng.binomial(1, prob)

    # enforce quasi-complete separation
    y[separator == 1] = 1

    X = np.column_stack([separator, x1, x2, x3])
    return X, y


@pytest.fixture
def separation_data():
    return make_separation_data()
