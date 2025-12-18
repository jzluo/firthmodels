import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray

from firthmodels import FirthLogisticRegression


class FirthLogit:
    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike,
        *,
        offset: ArrayLike | None = None,
        check_rank: bool = False,
        **kwargs,
    ):
        self.endog = np.asarray(endog)
        self.exog = np.asarray(exog)
        self.offset = np.asarray(offset) if offset is not None else None

        missing = kwargs.pop("missing", "none")

        if kwargs:
            raise TypeError(
                f"__init__() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        if missing == "drop":
            raise NotImplementedError("missing='drop' is not supported")
        elif missing == "raise":
            if np.isnan(self.endog).any() or np.isnan(self.exog).any():
                raise ValueError("Input contains NaN values")

        if hasattr(exog, "columns"):
            self.exog_names = list(exog.columns)
        else:
            self.exog_names = [f"x{i + 1}" for i in range(self.exog.shape[1])]
