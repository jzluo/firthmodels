# firthmodels

[![CI](https://github.com/jzluo/firthmodels/actions/workflows/ci.yml/badge.svg)](https://github.com/jzluo/firthmodels/actions/workflows/ci.yml)
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjzluo%2Ffirthmodels%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
![GitHub License](https://img.shields.io/github/license/jzluo/firthmodels)


Firth-penalized logistic regression in Python.

## Why Firth penalization?

Standard maximum-likelihood logistic regression fails when your data has complete or quasi-complete separation: when a predictor (or combination of predictors) perfectly separates the outcome classes. In these cases, MLE produces infinite coefficient estimates.

Firth's method adds a penalty term that:
- Produces **finite, well-defined estimates** even with separated data
- **Reduces small-sample bias** in coefficient estimates
- Works as a drop-in replacement for standard logistic regression

This is common in:
- Case-control studies with rare exposures
- Small clinical trials
- Genome-wide or Phenome-wide association studies (GWAS/PheWAS)
- Any dataset where events are rare relative to predictors

## Installation

For now:
```bash
pip install git+https://github.com/jzluo/firthmodels.git
```

Requires Python 3.11+ and depends on NumPy, SciPy, and scikit-learn.

## Quick start

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

## Features

### scikit-learn compatible

`FirthLogisticRegression` follows the scikit-learn estimator API (`fit`, `predict`, `predict_proba`, `get_params`, `set_params`, etc.), and can be used with pipelines, cross-validation, and other sklearn tools:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), FirthLogisticRegression())
scores = cross_val_score(pipe, X, y, cv=5)
```

### Likelihood ratio tests

Compute LRT p-values for individual coefficients. These are more reliable than Wald p-values for small samples.

Standard errors are back-corrected from the LRT chi-squared statistic (as in regenie), ensuring that (beta/SE)² = chi². This is useful for meta-analysis where studies are weighted by 1/SE²:

```python
model.fit(X, y).lrt()  # Compute LRT for all features

model.lrt_pvalues_     # LRT p-values
model.lrt_bse_         # Back-corrected standard errors
```

Each feature requires a separate constrained model fit, so you can test selectively to avoid unnecessary computation:

```python
model.lrt(0)              # Single feature by index
model.lrt([0, 2])         # Multiple features
model.lrt(['snp', 'age']) # By name (if fitted with DataFrame)
```

### Confidence intervals

```python
model.conf_int()          # 95% Wald CIs
model.conf_int(alpha=0.1) # 90% CIs
```

### Sample weights and offsets

```python
model.fit(X, y, sample_weight=weights)
model.fit(X, y, offset=offset)
```

## API reference

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fit_intercept` | `True` | Whether to fit an intercept term |
| `max_iter` | `25` | Maximum Newton-Raphson iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `max_step` | `5.0` | Maximum step size per coefficient |
| `max_halfstep` | `25` | Maximum step-halvings per iteration |

### Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `coef_` | Coefficient estimates |
| `intercept_` | Intercept (0.0 if `fit_intercept=False`) |
| `bse_` | Wald standard errors |
| `intercept_bse_` | Wald standard error for intercept |
| `pvalues_` | Wald p-values |
| `intercept_pvalue_` | Wald p-value for intercept |
| `loglik_` | Penalized log-likelihood |
| `n_iter_` | Number of iterations |
| `converged_` | Whether the solver converged |
| `lrt_pvalues_` | LRT p-values (after calling `lrt()`) |
| `lrt_bse_` | Back-corrected SEs (after calling `lrt()`) |
| `intercept_lrt_pvalue_` | LRT p-value for intercept |
| `intercept_lrt_bse_` | Back-corrected SE for intercept |

### Methods

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit the model |
| `predict(X)` | Predict class labels |
| `predict_proba(X)` | Predict class probabilities |
| `predict_log_proba(X)` | Predict log class probabilities |
| `decision_function(X)` | Return linear predictor values |
| `lrt(features)` | Compute LRT p-values; `features` can be indices or column names. If `None`, tests all features. |
| `conf_int(alpha)` | Wald confidence intervals; `alpha` is the two-sided significance level (e.g. 0.05 for 95% CIs) |

## Roadmap

Current implementation uses a damped Newton–Raphson solver.

Add additional solvers (IRLS, L-BFGS, etc) and models (Cox proportional hazards).

## References

Firth D (1993). Bias reduction of maximum likelihood estimates. *Biometrika* 80, 27-38.

Heinze G, Schemper M (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine* 21, 2409-2419.

Mbatchou J et al. (2021). Computationally efficient whole-genome regression for
quantitative and binary traits. *Nature Genetics* 53, 1097-1103.

## License

MIT
