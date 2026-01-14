"""
Separation detection for binary logistic regression.

Implements the linear programming approach from Konis (2007) to detect
complete and quasi-complete separation before model fitting.

References
----------
Konis, K. (2007). Linear Programming Algorithms for Detecting Separated
Data in Binary Logistic Regression Models. DPhil thesis, University of Oxford.

Kosmidis I, Schumacher D, Schwendinger F (2022). _detectseparation:
Detect and Check for Separation and Infinite Maximum Likelihood
Estimates_. doi:10.32614/CRAN.package.detectseparation
<https://doi.org/10.32614/CRAN.package.detectseparation>, R package
version 0.3, <https://CRAN.R-project.org/package=detectseparation>.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog
from sklearn.utils.validation import check_X_y


@dataclass
class SeparationResult:
    """Result of separation detection.

    Attributes
    ----------
    separation : bool
        True if any form of separation was detected.
    is_finite : ndarray of shape (n_features,)
        Boolean array indicating which coefficients have finite MLEs.
        True = finite, False = infinite.
    directions : ndarray of shape (n_features,)
        Direction of infinite coefficients: +1 for +inf, -1 for -inf, 0 for finite.
    feature_names : tuple of str or None
        Names of coefficients (including intercept if present) from dataframe column
        names, or auto-generated (x0, x1, ...) if `fit_intercept=True`. `None` if `X`
        is a NumPy array and `fit_intercept=False`.
    """

    separation: bool
    is_finite: NDArray[np.bool_]
    directions: NDArray[np.int8]
    feature_names: tuple[str, ...] | None = None

    def __repr__(self) -> str:
        n_infinite = np.sum(~self.is_finite)
        return (
            f"SeparationResult(separation={self.separation}, n_infinite={n_infinite})"
        )

    def summary(self) -> str:
        """Return a formatted summary of separation detection results.

        Returns
        -------
        str
            Summary showing separation status and which coefficients have infinite MLEs.

        Examples
        --------
        >>> print(result.summary())
        Separation: True
          treatment   +Inf
          age         finite
          intercept   -Inf
        """
        lines = [f"Separation: {self.separation}"]

        if len(self.directions) == 0:
            return lines[0]

        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = tuple(f"x{i}" for i in range(len(self.directions)))

        # right-pad all names to the same width for alignment
        max_len = max(len(name) for name in names)

        for name, direction in zip(names, self.directions):
            if direction == 0:
                status = "finite"
            elif direction == 1:
                status = "+Inf"
            else:
                status = "-Inf"
            lines.append(f"  {name:{max_len}}  {status}")

        return "\n".join(lines)


def detect_separation(
    X: ArrayLike,
    y: ArrayLike,
    *,
    fit_intercept: bool = True,
    tol: float = 1e-4,
) -> SeparationResult:
    """Detect separation in binary logistic regression data.

    Uses linear programming to determine if complete or quasi-complete
    separation exists in the data, which would cause maximum likelihood
    estimates to be infinite.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target labels.
    fit_intercept : bool, default=True
        Whether an intercept will be included in the model.
    tol : float, default=1e-4
        Tolerance for determining if a coefficient is non-zero (and thus
        infinite). Coefficients with |beta| > tol from the LP solution
        are considered infinite.

    Returns
    -------
    SeparationResult
        Object containing:
        - separation: bool indicating if separation exists
        - is_finite: boolean array for each coefficient
        - directions: +1/-1/0 indicating direction of infinity

    Notes
    -----
    The algorithm solves a linear program based on Konis (2007), particularly (4.23):

    ```
        maximize: sum(X_bar) @ beta
        subject to: X_bar @ beta >= 0  (element-wise)
                    -1 <= beta <= 1
    ```

    where X_bar[i] = (2*y[i] - 1) * X[i].

    If a coefficient has |beta| > tol in the solution, its MLE is infinite.
    The sign of beta indicates the direction (+inf or -inf).
    This implements the functionality of the detect_separation() function from the R
    package detectseparation (Kosmidis et al., 2022).

    References
    ----------
    Konis, K. (2007). Linear Programming Algorithms for Detecting Separated
    Data in Binary Logistic Regression Models. DPhil thesis, University of Oxford.

    Kosmidis I, Schumacher D, Schwendinger F (2022). _detectseparation:
    Detect and Check for Separation and Infinite Maximum Likelihood
    Estimates_. doi:10.32614/CRAN.package.detectseparation
    <https://doi.org/10.32614/CRAN.package.detectseparation>, R package
    version 0.3, <https://CRAN.R-project.org/package=detectseparation>.

    Examples
    --------
    >>> import numpy as np
    >>> from firthmodels import detect_separation
    >>> # Complete separation: X perfectly predicts y
    >>> X = np.array([[1], [2], [3], [4], [5], [6]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> result = detect_separation(X, y)
    >>> result.separation
    True
    >>> result.directions  # +1 means +inf, 0 means finite, -1 means -inf
    array([ 1, -1], dtype=int8)

    >>> # No separation
    >>> X = np.array([[1], [1], [2], [2], [3], [3]])
    >>> y = np.array([0, 1, 0, 1, 0, 1])
    >>> result = detect_separation(X, y)
    >>> result.separation
    False
    >>> result.directions
    array([0, 0], dtype=int8)

    >>> # Endometrial cancer data used in R detectseparation vignette(Heinze & Schemper, 2002)
    >>> # NV (neovasculization) causes quasi-complete separation
    >>> X = endometrial[["NV", "PI", "EH"]]
    >>> y = endometrial["HG"]
    >>> result = detect_separation(X, y)
    >>> print(result.summary())
    Separation: True
      NV         +Inf
      PI         finite
      EH         finite
      intercept  finite
    """
    feature_names: tuple[str, ...] | None = None
    if hasattr(X, "columns"):
        feature_names = tuple(str(c) for c in X.columns)

    X_arr, y_arr = check_X_y(X, y, dtype=np.float64, y_numeric=False, ensure_2d=True)
    n_samples, n_features = X_arr.shape

    unique_y = np.unique(y_arr)
    if len(unique_y) != 2:
        raise ValueError(f"y must be binary, got {len(unique_y)} unique values")

    # Map to 0/1 if needed
    if not (unique_y[0] == 0 and unique_y[1] == 1):
        y_arr = (y_arr == unique_y[1]).astype(np.float64)

    if fit_intercept:
        X_arr = np.column_stack([X_arr, np.ones(n_samples)])
        if feature_names is not None:
            feature_names = feature_names + ("intercept",)
        else:
            feature_names = tuple(f"x{i}" for i in range(n_features)) + ("intercept",)

    n_params = X_arr.shape[1]

    # Transform for LP: y_bar in {-1, +1}, X_bar = X_arr * y_bar
    y_bar = 2 * y_arr - 1
    X_bar = X_arr * y_bar[:, np.newaxis]

    # Solve bounded LP:
    # ```
    #     maximize: sum(X_bar) @ beta
    #     subject to: X_bar @ beta >= 0  (element-wise)
    #                 -1 <= beta <= 1
    # ```
    c = -np.sum(X_bar, axis=0)

    # Constraints: X_bar @ beta >= 0  <=>  -X_bar @ beta <= 0
    A_ub = -X_bar
    b_ub = np.zeros(n_samples)

    # Bounds: -1 <= beta <= 1
    bounds = [(-1.0, 1.0) for _ in range(n_params)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    beta = result.x

    # Determine directions: if |beta| > tol, coefficient is infinite
    is_zero = np.abs(beta) <= tol
    directions = np.zeros(n_params, dtype=np.int8)
    directions[~is_zero] = np.sign(beta[~is_zero]).astype(np.int8)

    is_finite = directions == 0
    separation = not np.all(is_finite)

    return SeparationResult(
        separation=separation,
        is_finite=is_finite,
        directions=directions,
        feature_names=feature_names,
    )
