from typing import Literal

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from firthmodels import FirthLogisticRegression
from firthmodels.logistic import compute_logistic_quantities


class FirthLogit:
    def __init__(
        self,
        endog: ArrayLike,
        exog: ArrayLike,
        *,
        offset: ArrayLike | None = None,
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

    @property
    def nobs(self) -> int:
        return self.exog.shape[0]

    def __repr__(self) -> str:
        return f"<FirthLogit: nobs={self.nobs}, k={self.exog.shape[1]}>"

    def fit(
        self,
        start_params: ArrayLike | None = None,
        method: Literal["newton"] = "newton",
        maxiter: int = 25,
        pl: bool = True,
        **kwargs,  # gtol, xtol
    ):
        if start_params is not None:
            raise NotImplementedError("start_params is not currently supported.")
        if method != "newton":
            raise ValueError("Only 'newton' method is currently supported.")

        gtol = kwargs.pop("gtol", 1e-4)
        xtol = kwargs.pop("xtol", 1e-4)
        if kwargs:
            raise TypeError(
                f"fit() got unexpected keyword arguments: {list(kwargs.keys())}"
            )

        estimator = FirthLogisticRegression(
            fit_intercept=False,
            max_iter=maxiter,
            gtol=gtol,
            xtol=xtol,
        )
        estimator.fit(self.exog, self.endog, offset=self.offset)
        if pl:
            estimator.lrt()
        return FirthLogitResults(self, estimator, pl=pl)


class FirthLogitResults:
    def __init__(
        self, model: FirthLogit, estimator: FirthLogisticRegression, pl: bool = True
    ):
        self.model = model
        self.estimator = estimator
        self._pl = pl

    @property
    def params(self) -> NDArray[np.float64]:
        return self.estimator.coef_

    @property
    def bse(self) -> NDArray[np.float64]:
        return self.estimator.lrt_bse_ if self._pl else self.estimator.bse_

    @property
    def tvalues(self) -> NDArray[np.float64]:
        return self.params / self.bse

    @property
    def pvalues(self) -> NDArray[np.float64]:
        return self.estimator.lrt_pvalues_ if self._pl else self.estimator.pvalues_

    @property
    def llf(self) -> float:
        return self.estimator.loglik_

    @property
    def converged(self) -> bool:
        return self.estimator.converged_

    @property
    def nobs(self) -> int:
        return self.model.exog.shape[0]

    @property
    def df_model(self) -> int:
        return len(self.params) - 1

    @property
    def df_resid(self) -> int:
        return self.nobs - len(self.params)

    @property
    def mle_retvals(self) -> dict:
        return {"converged": self.converged, "iterations": self.estimator.n_iter_}

    @property
    def fittedvalues(self) -> NDArray[np.float64]:
        return self.predict()

    def __repr__(self) -> str:
        return f"<FirthLogitResults: nobs={self.nobs}, converged={self.converged}>"

    def predict(
        self,
        exog: ArrayLike | None = None,
        offset: ArrayLike | None = None,
        **kwargs,
    ) -> NDArray[np.float64]:
        if exog is None:
            exog = self.model.exog
            offset = self.model.offset

        if offset is None:
            offset = np.zeros(exog.shape[0])

        linear_pred = exog @ self.params + offset

        if kwargs.get("linear", False):
            return linear_pred

        return expit(linear_pred)

    def conf_int(
        self,
        alpha=0.05,
        method: Literal["wald", "pl"] | None = None,
        **kwargs,  # maxiter, tol
    ) -> NDArray[np.float64]:
        if method is None:
            method = "pl" if self._pl else "wald"

        max_iter = kwargs.pop("maxiter", 25)
        tol = kwargs.pop("tol", 1e-4)
        if kwargs:
            raise TypeError(
                f"conf_int() got unexpected keyword arguments: {list(kwargs.keys())}"
            )
        return self.estimator.conf_int(
            alpha=alpha, method=method, max_iter=max_iter, tol=tol
        )

    def cov_params(self) -> NDArray[np.float64]:
        X, y, sample_weight, offset = self.estimator._fit_data
        q = compute_logistic_quantities(X, y, self.params, sample_weight, offset)
        return np.linalg.inv(q.fisher_info)

    def summary(self, alpha: float = 0.05) -> "FirthSummary":
        """Generate a summary of the regression results."""
        ci = self.conf_int(alpha=alpha)
        ci_lower = alpha / 2
        ci_upper = 1 - alpha / 2

        # Width and formatting
        width = 78

        def fmtval(x: float, width: int = 9) -> str:
            """Format a number: scientific notation for extreme values."""
            if np.isnan(x):
                return f"{'NaN':>{width}}"
            if x == 0:
                return f"{0.0:{width}.4f}"
            if abs(x) < 0.0001 or abs(x) >= 1e6:
                return f"{x:{width}.3e}"
            return f"{x:{width}.4f}"

        lines: list[str] = []

        # Title
        title = "Firth Logistic Regression Results"
        lines.append(title.center(width))
        lines.append("=" * width)

        # Model info - two column layout
        # Left: optimization/fitting, Right: data/model structure
        solver_str = self.estimator.solver.title()
        _, y, _, _ = self.estimator._fit_data
        n_events = int(y.sum())

        info_left = [
            ("Solver:", solver_str),
            ("Converged:", str(self.converged)),
            ("No. Iterations:", str(self.mle_retvals["iterations"])),
            ("Log-Likelihood:", f"{self.llf:.3f}"),
        ]
        info_right = [
            ("No. Observations:", str(self.nobs)),
            ("No. Events:", str(n_events)),
            ("Df Model:", str(self.df_model)),
            ("Df Residual:", str(self.df_resid)),
        ]

        for (l_lbl, l_val), (r_lbl, r_val) in zip(info_left, info_right):
            left = f"{l_lbl:<18} {l_val:<20}"
            right = f"{r_lbl:<18} {r_val:>10}"
            lines.append(left + right)

        lines.append("=" * width)

        # Coefficient table header
        ci_lo_hdr = f"[{ci_lower:.3g}"
        ci_hi_hdr = f"{ci_upper:.3g}]"
        hdr = f"{'':>12} {'coef':>9} {'std err':>9} {'z':>9} {'P>|z|':>9} {ci_lo_hdr:>9} {ci_hi_hdr:>9}"
        lines.append(hdr)
        lines.append("-" * width)

        # Coefficient rows
        for i, name in enumerate(self.model.exog_names):
            name_trunc = name[:12] if len(name) > 12 else name
            row = (
                f"{name_trunc:>12} "
                f"{fmtval(self.params[i])} "
                f"{fmtval(self.bse[i])} "
                f"{fmtval(self.tvalues[i])} "
                f"{fmtval(self.pvalues[i])} "
                f"{fmtval(ci[i, 0])} "
                f"{fmtval(ci[i, 1])}"
            )
            lines.append(row)

        lines.append("=" * width)

        # Footer
        if self._pl:
            lines.append(
                "P-values: Penalized likelihood ratio test | CIs: Profile penalized likelihood"
            )
        else:
            lines.append("P-values: Wald test | CIs: Wald")

        return FirthSummary("\n".join(lines))

    def summary_frame(self, alpha: float = 0.05):
        """Return summary as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for summary_frame()") from e

        ci = self.conf_int(alpha=alpha)
        ci_lower = alpha / 2
        ci_upper = 1 - alpha / 2

        return pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                "z": self.tvalues,
                "P>|z|": self.pvalues,
                f"[{ci_lower:.3g}": ci[:, 0],
                f"{ci_upper:.3g}]": ci[:, 1],
            },
            index=self.model.exog_names,
        )


class FirthSummary:
    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        return self._text

    def as_text(self) -> str:
        return self._text

    def as_html(self) -> str:
        raise NotImplementedError("HTML summary is not supported.")

    def as_latex(self) -> str:
        raise NotImplementedError("LaTeX summary is not supported.")
