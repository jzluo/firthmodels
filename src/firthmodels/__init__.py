try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from firthmodels.cox import FirthCoxPH
from firthmodels.logistic import FirthLogisticRegression

__all__ = [
    "FirthCoxPH",
    "FirthLogisticRegression",
]
