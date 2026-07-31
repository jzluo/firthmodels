# Benchmarks

Benchmarking of implementations of Firth-penalized logistic regression and Cox regression.

## Environment

| | |
|-----|-----|
| **OS** | Pop!_OS 24.04 LTS |
| **CPU** | AMD Ryzen 5 5600X 6-Core Processor |
| **Python** | 3.12.12 |
| **NumPy** | 2.3.5 |
| **SciPy** | 1.16.3 |
| **Numba** | 0.64.0 |
| **R** | 4.5.2 |

---

## Firth Logistic Regression

Comparison of [firthmodels](https://github.com/jzluo/firthmodels),
R [brglm2](https://cran.r-project.org/web/packages/brglm2/index.html),
and R [logistf](https://cran.r-project.org/web/packages/logistf/index.html)
for Firth-penalized logistic regression.

### Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels** | 0.8.0 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **brglm2** | 1.0.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |
| **logistf** | 1.26.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |

### Configuration

| Parameter | Value |
|-----------|-------|
| Observations (n) | 1,000 |
| Event rate | 20% |
| Features (k) | 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 |
| Runs per config | 20 |
| Solver max_iter | 50 |
| Solver tolerance | 1e-06 |

brglm2 runs with `check_aliasing=FALSE` since the benchmark data is guaranteed full rank.

All implementations agree within chosen tolerance (coefficients 1e-06, CIs 1e-06, p-values 1e-06).

### Results

![Logistic benchmark scaling plot](logistic_results.png)

#### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | brglm2<br>(AS-mean) | brglm2<br>(MPL_Jeffreys) | logistf |
|--:|------:|------:|------------:|-------------:|--------:|
|   5 | 0.5 | 1.4 | 4.2 | 4.3 | 2.1 |
|  10 | 1.1 | 1.8 | 7.5 | 7.4 | 4.2 |
|  15 | 1.2 | 2.0 | 8.4 | 8.5 | 6.8 |
|  20 | 1.4 | 2.2 | 9.6 | 9.7 | 15.1 |
|  25 | 1.7 | 2.7 | 12.0 | 12.0 | 23.2 |
|  30 | 2.0 | 2.8 | 13.0 | 13.0 | 28.8 |
|  35 | 2.1 | 3.2 | 15.5 | 15.4 | 40.5 |
|  40 | 2.6 | 3.7 | 17.3 | 17.5 | 51.1 |
|  45 | 3.3 | 4.6 | 19.1 | 19.3 | 67.9 |
|  50 | 4.1 | 5.4 | 24.3 | 24.5 | 83.5 |

#### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf |
|--:|------:|------:|--------:|
|   5 | 4.7 | 11.0 | 13.8 |
|  10 | 19.4 | 40.0 | 81.4 |
|  15 | 29.2 | 63.8 | 156.9 |
|  20 | 45.8 | 89.8 | 438.5 |
|  25 | 77.1 | 141.8 | 903.0 |
|  30 | 110.6 | 183.6 | 1307.5 |
|  35 | 138.3 | 236.3 | 2209.1 |
|  40 | 199.7 | 314.7 | 3188.7 |
|  45 | 266.0 | 405.1 | 4574.0 |
|  50 | 370.9 | 532.9 | 6282.8 |


---

## Firth Cox Proportional Hazards

Comparison of [firthmodels](https://github.com/jzluo/firthmodels)
and [coxphf](https://cran.r-project.org/web/packages/coxphf/index.html)
for Firth-penalized Cox proportional hazards regression.

### Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels** | 0.8.0 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **coxphf** | 1.13.4 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |

### Configuration

| Parameter | Value |
|-----------|-------|
| Observations (n) | 500 |
| Event rate | 20% |
| Features (k) | 5, 10, 15, 20, 25, 30 |
| Runs per config | 10 |
| Solver max_iter | 50 |
| Solver xtol | 1e-06 |
| Solver gtol | 0.0001 |

All implementations agree within chosen tolerance (coefficients 1e-06, CIs 1e-06, p-values 1e-06).

### Results

![Cox benchmark scaling plot](cox_results.png)

#### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
|   5 | 1.1 | 1.8 | 4.3 |
|  10 | 0.7 | 2.5 | 16.8 |
|  15 | 0.9 | 3.6 | 49.8 |
|  20 | 1.9 | 8.0 | 137.7 |
|  25 | 2.6 | 11.1 | 268.9 |
|  30 | 3.3 | 14.4 | 464.5 |

#### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
|   5 | 2.6 | 12.9 | 26.9 |
|  10 | 7.6 | 45.6 | 360.4 |
|  15 | 18.7 | 115.3 | 2019.4 |
|  20 | 52.4 | 294.7 | 6447.7 |
|  25 | 100.1 | 535.3 | 16706.7 |
|  30 | 148.2 | 870.7 | 34470.1 |


---

## Reproducing These Results

```bash
# Run logistic regression benchmarks
python benchmarks/benchmark_logistic.py -o benchmarks/logistic_results.csv

# Run Cox PH benchmarks
python benchmarks/benchmark_cox.py -o benchmarks/cox_results.csv

# Generate plots and README
python benchmarks/generate_report.py
```
