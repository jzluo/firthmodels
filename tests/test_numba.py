import numpy as np
import pytest
from scipy.special import expit

from firthmodels import NUMBA_AVAILABLE

pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="numba not available")

if NUMBA_AVAILABLE:
    from firthmodels._numba._utils import (
        _expit,
    )


@pytest.mark.parametrize(
    "x", [-np.inf, -710, -100, -10, -1, 5, 0, 1, 5, 10, 100, 710, np.inf]
)
def test_expit(x):
    assert _expit(x) == pytest.approx(expit(x), rel=1e-14, abs=1e-300)
