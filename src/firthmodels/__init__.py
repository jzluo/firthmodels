try:
    import numba

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from firthmodels.cox import FirthCoxPH
from firthmodels.logistic import FirthLogisticRegression
from firthmodels.separation import SeparationResult, detect_separation

__all__ = [
    "FirthCoxPH",
    "FirthLogisticRegression",
    "SeparationResult",
    "detect_separation",
]
