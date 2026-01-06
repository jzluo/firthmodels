# Firth Logistic Regression Benchmark

Comparison of [firthmodels](https://github.com/jzluo/firthmodels),
R [brglm2](https://cran.r-project.org/web/packages/brglm2/index.html),
and R [logistf](https://cran.r-project.org/web/packages/logistf/index.html)
packages for Firth-penalized logistic regression.

## System

| | |
|-----|-----|
| **OS** | Pop!_OS 24.04 LTS |
| **CPU** | AMD Ryzen 5 5600X 6-Core Processor |

## Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels (numba)** | 0.3.0 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **firthmodels (numpy)** | 0.3.0 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **brglm2 (AS-mean)** | 1.0.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |
| **brglm2 (MPL_Jeffreys)** | 1.0.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |
| **logistf** | 1.26.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |

## Benchmark Configuration

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

## Results

![Benchmark scaling plot](logistic_report.png)

### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | brglm2<br>(AS-mean) | brglm2<br>(MPL_Jeffreys) | logistf |
|--:|------:|------:|------------:|-------------:|--------:|
|   5 | 0.5 | 1.2 | 4.2 | 4.3 | 2.1 |
|  10 | 1.2 | 1.7 | 7.8 | 8.3 | 4.6 |
|  15 | 1.2 | 1.9 | 9.7 | 8.7 | 7.2 |
|  20 | 1.4 | 2.1 | 10.4 | 10.1 | 15.7 |
|  25 | 1.9 | 2.8 | 13.1 | 13.0 | 26.9 |
|  30 | 2.2 | 3.1 | 13.6 | 14.4 | 31.7 |
|  35 | 2.6 | 3.2 | 17.1 | 16.0 | 44.3 |
|  40 | 2.8 | 3.7 | 17.7 | 18.5 | 54.8 |
|  45 | 3.5 | 3.8 | 19.9 | 21.0 | 73.1 |
|  50 | 4.4 | 5.6 | 24.0 | 24.2 | 88.8 |

### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf |
|--:|------:|------:|--------:|
|   5 | 5.2 | 11.4 | 13.7 |
|  10 | 23.2 | 49.0 | 82.9 |
|  15 | 35.3 | 75.1 | 156.7 |
|  20 | 56.8 | 118.2 | 452.0 |
|  25 | 94.8 | 164.5 | 930.5 |
|  30 | 136.3 | 209.8 | 1369.9 |
|  35 | 182.0 | 287.2 | 2357.2 |
|  40 | 252.3 | 381.6 | 3364.6 |
|  45 | 338.0 | 504.4 | 4759.4 |
|  50 | 480.8 | 659.8 | 6629.1 |


---


## Command used to run benchmark

```bash
python benchmarks/benchmark_logistic.py --report benchmarks/logistic_report.md --csv benchmarks/logistic_results.csv --n-runs 20
```
