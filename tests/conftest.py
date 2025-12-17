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


@pytest.fixture
def separation_data_df():
    """Separation data as DataFrame with named columns."""
    import pandas as pd

    X, y = make_separation_data()
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4"])
    return df, y


def make_cox_separation_data(seed=42, n=100):
    """Generate survival data with monotone likelihood for testing."""
    rng = np.random.default_rng(seed)

    separator = rng.choice([0, 1], n, p=[0.35, 0.65])  # only affects censoring
    x1 = rng.standard_normal(n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.choice([0, 1], n, p=[0.6, 0.4])

    X = np.column_stack([separator, x1, x2, x3])

    # Generate survival times from Cox model
    beta = np.array([0.0, 0.5, -0.3, 0.7])
    eta = X @ beta

    baseline_hazard = 0.1
    survival_time = rng.exponential(1 / (baseline_hazard * np.exp(eta)))

    # Random censoring
    censor_time = rng.exponential(scale=15.0, size=n)

    time = np.minimum(survival_time, censor_time)
    event = survival_time <= censor_time

    # All events in separator=0 group become censored
    event[separator == 0] = False

    return X, time, event.astype(bool)


@pytest.fixture
def cox_separation_data():
    """Fixture providing survival data with monotone likelihood."""
    return make_cox_separation_data()
