# firthmodels

[![CI](https://github.com/jzluo/firthmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/jzluo/firthmodels/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/firthmodels)](https://pypi.org/project/firthmodels/)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/firthmodels)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjzluo%2Ffirthmodels%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub License](https://img.shields.io/github/license/jzluo/firthmodels)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17863280.svg)](https://doi.org/10.5281/zenodo.17863280)

Firth-penalized models in Python:

- `FirthLogisticRegression`: scikit-learn–compatible Firth logistic regression
- `FirthCoxPH`: scikit-survival-style Firth Cox proportional hazards

Firth penalization reduces small-sample bias and produces finite estimates even when
standard MLE fails due to (quasi-)complete separation or monotone likelihood.

See the issue trackers [#12](https://github.com/jzluo/firthmodels/issues/12) for `FirthLogisticRegression` benchmarking and [#32](https://github.com/jzluo/firthmodels/issues/32) for `FirthCoxPH` benchmarking.

## Why Firth penalization?

Standard maximum-likelihood logistic regression fails when your data has complete or
quasi-complete separation: when a predictor (or combination of predictors) perfectly
separates the outcome classes. In these cases, MLE produces infinite coefficient
estimates.

In Cox proportional hazards, an analogous failure mode is monotone likelihood, where the
partial likelihood becomes unbounded (often due to small samples, rare events, or
near-perfect risk separation).

Firth's method adds a penalty term that:
- Produces **finite, well-defined estimates** even with separated data
- **Reduces small-sample bias** in coefficient estimates

These problems are common in:
- Case-control studies with rare exposures
- Small clinical trials
- Genome-wide or Phenome-wide association studies (GWAS/PheWAS)
- Any dataset where events are rare relative to predictors

## Installation

```bash
pip install firthmodels
```

Requires Python 3.11+ and depends on NumPy, SciPy, and scikit-learn.

Optional dependencies:
- Numba acceleration: `pip install firthmodels[numba]`
- Formula interface for the statsmodels adapter: `pip install firthmodels[formula]`
(or simply install [formulaic](https://matthewwardrop.github.io/formulaic/latest/)).

## Quick start

### Firth logistic regression

```python
import numpy as np
from firthmodels import FirthLogisticRegression

# Separated data: x=1 perfectly predicts y=1
X = np.array([[0], [0], [0], [1], [1], [1]])
y = np.array([0, 0, 0, 1, 1, 1])

# Standard logistic regression would fail here
model = FirthLogisticRegression().fit(X, y)

print(model.coef_)        # array([3.89181893])
print(model.intercept_)   # -2.725...
print(model.pvalues_)     # Wald p-values
print(model.bse_)         # Standard errors
```

### Firth Cox proportional hazards

```python
import numpy as np
from firthmodels import FirthCoxPH

X = np.array([[1.0], [0.0]])
event = np.array([True, False])
time = np.array([1.0, 2.0])

model = FirthCoxPH().fit(X, (event, time))
print(model.coef_)          # log hazard ratios
print(model.pvalues_)       # Wald p-values

# Survival curves evaluated at the training event times
S = model.predict_survival_function(X)  # shape: (n_samples, n_event_times)
```

`FirthCoxPH` also accepts `y` as a structured array with boolean `event` and float `time`
fields (scikit-survival style).

Both estimators take a `backend` parameter that can be one of `'auto'` (default), `'numba'`, or `'numpy'`. If `'auto'`, firthmodels auto-detects Numba availability and uses
it if installed, otherwise numpy/scipy.

## Estimators

### scikit-learn compatible API

`FirthLogisticRegression` follows the scikit-learn estimator API
(`fit`, `predict`, `predict_proba`, `get_params`, `set_params`, etc.), and can be used
with pipelines, cross-validation, and other sklearn tools:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), FirthLogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)
```

`FirthCoxPH` also follows the sklearn estimator API (`fit`, `predict`, `score`, etc.).
It also has a scikit-survival-like interface:
  - methods: `fit(X, y)`, `predict(X)` (linear predictor), `score(X, y)` (C-index),
  `predict_survival_function(X, return_array=True)`,
  `predict_cumulative_hazard_function(X, return_array=True)`.
  - attributes: `unique_times_`, `cum_baseline_hazard_`, `baseline_survival_` (Breslow-style baseline).

## Inference

### Likelihood ratio tests (LRT)

Both estimators support penalized likelihood ratio tests for individual coefficients.
These are often more reliable than Wald p-values in small samples.

`lrt()` populates `lrt_pvalues_` and `lrt_bse_` (a back-corrected standard error such
that `(beta / lrt_bse_)**2` matches the 1-df chi-squared statistic), which can be
useful for meta-analysis weighting:

```python
model.fit(X, y).lrt()  # Compute LRT for all coefficients

model.lrt_pvalues_     # LRT p-values
model.lrt_bse_         # Back-corrected standard errors (separate from Wald bse_)
```

Each feature requires a separate constrained model fit, so you can test selectively to
avoid unnecessary computation:

```python
model.lrt(0)              # Single feature by index
model.lrt([0, 2])         # Multiple features
model.lrt(['snp', 'age']) # By name (if fitted with DataFrame)
model.lrt(['snp', 2])     # Mixed

```

### Confidence intervals

```python
model.conf_int()                    # 95% Wald CIs
model.conf_int(alpha=0.1)           # 90% CIs
model.conf_int(method='pl')         # Profile likelihood CIs (more accurate)
model.conf_int(method='pl', features=['snp', 'age'])  # can selectively compute as with LRT
```

### Sample weights and offsets

```python
# currently for FirthLogisticRegression only
model.fit(X, y, sample_weight=weights)
model.fit(X, y, offset=offset)
```

## Statsmodels adapter (`FirthLogit`)

The statsmodels adapter wraps `FirthLogisticRegression` behind a statsmodels-like API
and returns a results object with common statsmodels attributes/methods
(`params`, `bse`, `pvalues`, `summary()`, `cov_params()`, etc.).

Notes:

- Unlike sklearn, `FirthLogit` does not add an intercept automatically; use `sm.add_constant(X)`.
- `fit(pl=True)` (default) computes likelihood ratio p-values and uses profile
  likelihood confidence intervals by default, matching R logistf convention. Standard errors (`bse`) remain Wald standard errors.

```python
import numpy as np
import statsmodels.api as sm
from firthmodels.adapters.statsmodels import FirthLogit

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

X = sm.add_constant(X)
res = FirthLogit(y, X).fit()

print(res.params)
print(res.pvalues)      # LRT p-values when pl=True
print(res.conf_int())   # profile likelihood CIs when pl=True
print(res.summary())
```

### Formula interface

If you install `firthmodels[formula]` (or `pip install formulaic`), you can fit from
a formula and a pandas DataFrame:

```python
import pandas as pd
from firthmodels.adapters.statsmodels import FirthLogit

df = pd.DataFrame({"y": [0, 1, 0, 1], "age": [20, 30, 40, 50], "treatment": [0, 1, 0, 1]})
res = FirthLogit.from_formula("y ~ age + treatment", df).fit()
```

## API notes

### `FirthLogisticRegression` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `'auto'` | `'auto'`, `'numba'`, or `'numpy'`. `'auto'` uses numba if available. |
| `fit_intercept` | `True` | Whether to add an intercept term |
| `max_iter` | `25` | Maximum Newton-Raphson iterations |
| `gtol` | `1e-4` | Gradient convergence tolerance (converged when max\|gradient\| < gtol) |
| `xtol` | `1e-4` | Parameter convergence tolerance (converged when max\|delta\| < xtol) |
| `max_step` | `5.0` | Maximum step size per coefficient |
| `max_halfstep` | `25` | Maximum step-halvings per iteration |

### `FirthLogisticRegression` attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `coef_` | Coefficient estimates |
| `intercept_` | Intercept (0.0 if `fit_intercept=False`) |
| `bse_` | Wald standard errors; includes intercept if `fit_intercept=True` |
| `pvalues_` | Wald p-values; includes intercept if `fit_intercept=True` |
| `loglik_` | Penalized log-likelihood |
| `n_iter_` | Number of iterations |
| `converged_` | Whether the solver converged |
| `lrt_pvalues_` | LRT p-values (after calling `lrt()`); includes intercept if `fit_intercept=True` |
| `lrt_bse_` | Back-corrected SEs (after calling `lrt()`); includes intercept if `fit_intercept=True` |
| `classes_` | Class labels (shape `(2,)`) |
| `n_features_in_` | Number of features seen during fit |
| `feature_names_in_` | Feature names (if X had string column names) |


### `FirthCoxPH` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | `'auto'` | `'auto'`, `'numba'`, or `'numpy'`. `'auto'` uses numba if available. |
| `max_iter` | `50` | Maximum Newton-Raphson iterations |
| `gtol` | `1e-4` | Gradient convergence tolerance (converged when max\|gradient\| < gtol) |
| `xtol` | `1e-6` | Parameter convergence tolerance (converged when max\|delta\| < xtol) |
| `max_step` | `5.0` | Maximum step size per coefficient |
| `max_halfstep` | `5` | Maximum step-halvings per iteration |

### `FirthCoxPH` attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `coef_` | Coefficient estimates (log hazard ratios) |
| `bse_` | Wald standard errors |
| `pvalues_` | Wald p-values |
| `loglik_` | Penalized log partial likelihood |
| `n_iter_` | Number of iterations |
| `converged_` | Whether the solver converged |
| `lrt_pvalues_` | LRT p-values (after calling `lrt()`) |
| `lrt_bse_` | Back-corrected SEs (after calling `lrt()`) |
| `unique_times_` | Unique event times (ascending order) |
| `cum_baseline_hazard_` | Breslow cumulative baseline hazard at `unique_times_` |
| `baseline_survival_` | Baseline survival function at `unique_times_` |
| `n_features_in_` | Number of features seen during fit |
| `feature_names_in_` | Feature names (if X had string column names) |

`predict(X)` returns the linear predictor `X @ coef_` (log partial hazard).
`predict_cumulative_hazard_function(X)` and `predict_survival_function(X)` return arrays
evaluated at `unique_times_`.

## References

Firth D (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80, 27-38.

Heinze G, Schemper M (2001). A solution to the problem of monotone likelihood in Cox regression. *Biometrics* 57, 114-119.

Heinze G, Schemper M (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine* 21, 2409-2419.

Mbatchou J et al. (2021). Computationally efficient whole-genome regression for quantitative and binary traits. *Nature Genetics* 53, 1097-1103.

Venzon DJ, Moolgavkar SH (1988). A method for computing profile-likelihood-based confidence intervals. *Applied Statistics* 37, 87-94.

## License

MIT
