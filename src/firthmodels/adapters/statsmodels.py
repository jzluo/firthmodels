"""
Statsmodels-style API adapter for Firth logistic regression.

This module provides `FirthLogit`, a statsmodels-compatible wrapper
around `firthmodels.FirthLogisticRegression` that uses Firth's
penalized likelihood method.

Example
-------
>>> import numpy as np
>>> import statsmodels.api as sm
>>> from firthmodels.adapters.statsmodels import FirthLogit
>>>
>>> # Separated data: standard logit would not converge
>>> X = np.array([[1], [2], [3], [4], [5], [6]])
>>> y = np.array([0, 0, 0, 1, 1, 1])  # perfect separation
>>> X = sm.add_constant(X)
>>>
>>> result = FirthLogit(y, X).fit()
>>> print(result.summary())
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from firthmodels import FirthLogisticRegression
from firthmodels.logistic import compute_logistic_quantities


class FirthLogit:
    """
    Firth penalized logistic regression model with statsmodels-style API.

    This class provides a statsmodels-compatible interface to Firth's
    bias-reduced penalized likelihood estimation. Unlike standard maximum
    likelihood, Firth regression produces finite coefficient estimates even
    with perfect or quasi-complete separation.

    Parameters
    ----------
    endog : array_like
        Binary response variable (0/1 or boolean).
    exog : array_like
        Design matrix of explanatory variables. Unlike sklearn, this class
        does NOT automatically add an intercept. Use `sm.add_constant(X)`
        to add a constant column if desired.
    offset : array_like, optional
        Offset to be added to the linear predictor.
    missing : {'none', 'raise'}, default 'none'
        How to handle missing values. 'none' does no checking (NaN will
        cause errors during fitting). 'raise' checks inputs and raises
        ValueError if NaN is present. 'drop' is not supported.

    Attributes
    ----------
    endog : ndarray
        The response variable as a numpy array.
    exog : ndarray
        The design matrix as a numpy array.
    exog_names : list of str
        Names for the columns in exog. Taken from DataFrame column names
        if exog is a DataFrame, otherwise generated as ['x1', 'x2', ...].
    nobs : int
        Number of observations.

    See Also
    --------
    FirthLogisticRegression : The sklearn-compatible implementation.

    Examples
    --------
    >>> import numpy as np
    >>> import statsmodels.api as sm
    >>> from firthmodels.adapters.statsmodels import FirthLogit

    Basic usage with numpy arrays:

    >>> X = np.random.randn(100, 2)
    >>> y = (X[:, 0] + np.random.randn(100) > 0).astype(int)
    >>> X = sm.add_constant(X)  # add intercept column
    >>> result = FirthLogit(y, X).fit()
    >>> result.params
    array([...])

    With pandas DataFrame (column names are preserved):

    >>> import pandas as pd
    >>> df = pd.DataFrame({'age': [25, 30, 35], 'treatment': [0, 1, 1]})
    >>> X = sm.add_constant(df)
    >>> y = [0, 1, 1]
    >>> result = FirthLogit(y, X).fit()
    >>> result.summary_frame()  # shows 'const', 'age', 'treatment' as row labels
    """

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

        self._formula: str | None = None

    @property
    def nobs(self) -> int:
        return self.exog.shape[0]

    def __repr__(self) -> str:
        return f"<FirthLogit: nobs={self.nobs}, k={self.exog.shape[1]}>"

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        subset: ArrayLike | None = None,
        **kwargs,
    ) -> FirthLogit:
        """
        Create a FirthLogit model from a formula and DataFrame.

        Parameters
        ----------
        formula : str
            R-style formula, e.g., "y ~ x1 + x2 + C(group)".
            Intercept is included by default; use "y ~ 0 + x" to exclude.
        data : DataFrame
            Data containing the variables referenced in the formula.
        subset : array_like, optional
            Boolean mask or index for subsetting rows.
        **kwargs
            Additional arguments passed to FirthLogit (e.g., `missing`).

        Returns
        -------
        FirthLogit
            Model instance ready to fit.

        Raises
        ------
        ImportError
            If formulaic is not installed.

        Examples
        --------
        >>> result = FirthLogit.from_formula("outcome ~ age + treatment", df).fit()
        >>> print(result.summary())

        With categorical variables (automatic dummy encoding):

        >>> result = FirthLogit.from_formula("y ~ C(group) + age", df).fit()

        Exclude intercept:

        >>> result = FirthLogit.from_formula("y ~ 0 + x1 + x2", df).fit()
        """
        try:
            from formulaic import model_matrix
        except ImportError as e:
            raise ImportError(
                "formulaic is required for from_formula(). "
                "Install with: pip install formulaic"
            ) from e

        if subset is not None:
            data = data.loc[subset]

        endog, exog = model_matrix(formula, data)

        model = cls(endog.iloc[:, 0], exog, **kwargs)
        model._formula = formula
        return model

    def fit(
        self,
        start_params: ArrayLike | None = None,
        method: Literal["newton"] = "newton",
        maxiter: int = 25,
        pl: bool = True,
        **kwargs,
    ) -> "FirthLogitResults":
        """
        Fit the Firth logistic regression model.

        Parameters
        ----------
        start_params : array_like, optional
            Not supported. Raises NotImplementedError if provided.
        method : {'newton'}, default 'newton'
            Optimization method. Only 'newton' (Newton-Raphson) is currently supported.
        maxiter : int, default 25
            Maximum number of iterations.
        pl : bool, default True
            If True (recommended), use profile likelihood inference:
            p-values from penalized likelihood ratio tests and profile
            likelihood confidence intervals. If False, use Wald inference
            (faster but less accurate for small samples).
        gtol : float, default 1e-4
            Gradient tolerance for convergence.
        xtol : float, default 1e-4
            Parameter tolerance for convergence.

        Returns
        -------
        FirthLogitResults
            Results object containing fitted parameters and inference methods.

        Notes
        -----
        The `pl` parameter controls which inference method is used throughout
        the results object. Profile likelihood inference (`pl=True`) is
        recommended for small samples and separated data, as Wald inference
        can be unreliable in these settings. This matches the default behavior
        of R's logistf package.

        When `pl=True`, the likelihood ratio test is computed at fit time,
        which adds computational overhead proportional to the number of
        parameters.

        Examples
        --------
        >>> result = FirthLogit(y, X).fit()  # pl=True by default
        >>> result.pvalues  # LRT p-values
        >>> result.conf_int()  # profile likelihood CIs

        >>> result = FirthLogit(y, X).fit(pl=False)  # faster, Wald inference
        >>> result.pvalues  # Wald p-values
        >>> result.conf_int()  # Wald CIs
        """
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
    """
    Results from fitting a Firth logistic regression model.

    This class provides statsmodels-style access to fitted parameters,
    standard errors, p-values, confidence intervals, and model summaries.

    Parameters
    ----------
    model : FirthLogit
        The model that was fit.
    estimator : FirthLogisticRegression
        The underlying sklearn-style estimator.
    pl : bool
        Whether profile likelihood inference was used.

    Attributes
    ----------
    params : ndarray
        Fitted coefficients.
    bse : ndarray
        Wald standard errors from the inverse Fisher information matrix.
    tvalues : ndarray
        Wald z-statistics (params / bse). Note: when `pl=True`, these do
        not correspond to `pvalues` (which are from LRT).
    pvalues : ndarray
        Two-sided p-values. LRT if `pl=True`, Wald if `pl=False`.
    llf : float
        Penalized log-likelihood at the fitted parameters.
    converged : bool
        Whether the optimizer converged.
    nobs : int
        Number of observations.
    df_model : int
        Model degrees of freedom (number of non-constant predictors).
    df_resid : int
        Residual degrees of freedom (nobs - number of parameters).
    fittedvalues : ndarray
        Predicted probabilities on the training data.
    model : FirthLogit
        Reference to the model object.

    Notes
    -----
    The `pl` parameter (set at fit time) controls `pvalues` and the default
    method for `conf_int()`. Standard errors (`bse`) are always Wald-based,
    following R's logistf convention. This means `tvalues` (Wald z) may not
    correspond to `pvalues` (LRT) when `pl=True` - this is intentional, as
    it keeps `bse` interpretable as a true standard error estimate.
    """

    def __init__(
        self,
        model: FirthLogit,
        estimator: FirthLogisticRegression,
        pl: bool = True,
    ):
        self.model = model
        self.estimator = estimator
        self._pl = pl

    @property
    def params(self) -> NDArray[np.float64]:
        return self.estimator.coef_

    @property
    def bse(self) -> NDArray[np.float64]:
        return self.estimator.bse_

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
        n_constant = np.sum(np.ptp(self.model.exog, axis=0) == 0)
        return len(self.params) - int(n_constant)

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
        """
        Compute predicted probabilities.

        Parameters
        ----------
        exog : array_like, optional
            Design matrix for prediction. If not provided, uses the training
            data (returning in-sample fitted values).
        offset : array_like, optional
            Offset for prediction. If `exog` is None, the training offset
            is used. If `exog` is provided but `offset` is None, no offset
            is applied (equivalent to zeros).
        linear : bool, default False
            If True, return the linear predictor (X @ beta + offset) instead
            of probabilities.

        Returns
        -------
        ndarray
            Predicted probabilities P(y=1), or linear predictor if linear=True.

        Examples
        --------
        >>> result = FirthLogit(y, X).fit()
        >>> result.predict()  # in-sample probabilities
        >>> result.predict(X_new)  # out-of-sample probabilities
        >>> result.predict(X_new, linear=True)  # linear predictor
        """
        if exog is None:
            exog_arr = self.model.exog
            offset_arr = self.model.offset
        else:
            exog_arr = np.asarray(exog)
            offset_arr = np.asarray(offset) if offset is not None else None

        if offset_arr is None:
            offset_arr = np.zeros(exog_arr.shape[0])

        linear_pred = exog_arr @ self.params + offset_arr

        if kwargs.get("linear", False):
            return linear_pred

        return expit(linear_pred)

    def conf_int(
        self,
        alpha: float = 0.05,
        method: Literal["wald", "pl"] | None = None,
        **kwargs,
    ) -> NDArray[np.float64]:
        """
        Compute confidence intervals for the coefficients.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level. Default gives 95% confidence intervals.
        method : {'wald', 'pl'}, optional
            Method for computing confidence intervals. If not specified,
            uses 'pl' (profile likelihood) if the model was fit with pl=True,
            otherwise 'wald'. Profile likelihood CIs are more accurate for
            small samples but slower to compute.
        maxiter : int, default 25
            Maximum iterations for profile likelihood CI computation.
            Only used when method='pl'.
        tol : float, default 1e-4
            Convergence tolerance for profile likelihood CI computation.
            Only used when method='pl'.

        Returns
        -------
        ndarray, shape (k, 2)
            Lower and upper confidence bounds for each parameter.

        Examples
        --------
        >>> result = FirthLogit(y, X).fit()
        >>> result.conf_int()  # 95% profile likelihood CIs (default)
        >>> result.conf_int(alpha=0.1)  # 90% CIs
        >>> result.conf_int(method='wald')  # Wald CIs (faster)
        """
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
        """
        Compute the covariance matrix of the parameter estimates.

        Returns the inverse of the augmented Fisher information matrix,
        which gives the asymptotic covariance of the estimates.

        Returns
        -------
        ndarray, shape (k, k)
            Covariance matrix of the fitted parameters.

        Notes
        -----
        The diagonal elements are the squared Wald standard errors, i.e.,
        `sqrt(diag(cov_params()))` equals `self.bse`.
        """
        return self.estimator._cov

    def summary(self, alpha: float = 0.05) -> "FirthSummary":
        """
        Generate a text summary of the regression results.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for confidence intervals.

        Returns
        -------
        FirthSummary
            Summary object with `__str__` method for printing.

        Examples
        --------
        >>> result = FirthLogit(y, X).fit()
        >>> print(result.summary())
                              Firth Logistic Regression Results
        ==============================================================================
        Solver:            Newton-Raphson      No. Observations:         100
        ...
        """
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
        """
        Return summary statistics as a pandas DataFrame.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for confidence intervals.

        Returns
        -------
        DataFrame
            DataFrame with columns: coef, std err, z, P>|z|, and CI bounds.
            Index contains the variable names from `model.exog_names`.

        Raises
        ------
        ImportError
            If pandas is not installed.

        Examples
        --------
        >>> result = FirthLogit(y, X).fit()
        >>> df = result.summary_frame()
        >>> df[df['P>|z|'] < 0.05]  # significant coefficients
        """
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
    """
    Container for regression summary text.

    Wraps the formatted summary string and provides output methods
    compatible with statsmodels Summary objects.

    Parameters
    ----------
    text : str
        The formatted summary text.
    """

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
