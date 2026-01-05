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
| **firthmodels (numba)** | @ commit 0863ddf | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **firthmodels (numpy)** | @ commit 0863ddf | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
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

All implementations agree within chosen tolerance (coefficients 1e-06, CIs 1e-06, p-values 1e-06). Python results are verified against both R packages.

## Results

![Benchmark scaling plot](logistic_report.png)

### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | numba | numpy | brglm2<br>(AS-mean) | brglm2<br>(MPL_Jeffreys) | logistf |
|--:|------:|------:|------------:|-------------:|--------:|
|   5 | 0.5 | 1.3 | 4.3 | 4.3 | 2.1 |
|  10 | 1.1 | 1.6 | 7.9 | 7.6 | 4.4 |
|  15 | 1.2 | 1.9 | 8.6 | 8.6 | 7.0 |
|  20 | 1.3 | 2.0 | 10.1 | 10.6 | 16.0 |
|  25 | 1.8 | 2.5 | 12.7 | 13.1 | 25.3 |
|  30 | 2.0 | 2.9 | 14.4 | 13.3 | 30.8 |
|  35 | 2.3 | 3.2 | 15.5 | 15.7 | 44.3 |
|  40 | 2.8 | 3.7 | 17.6 | 17.6 | 54.0 |
|  45 | 3.6 | 4.6 | 19.4 | 20.9 | 71.0 |
|  50 | 4.7 | 5.5 | 24.0 | 25.3 | 88.6 |

### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | numba | numpy | logistf |
|--:|------:|------:|--------:|
|   5 | 5.1 | 11.4 | 13.7 |
|  10 | 22.9 | 43.1 | 61.6 |
|  15 | 33.6 | 67.4 | 156.6 |
|  20 | 53.4 | 97.6 | 448.1 |
|  25 | 90.4 | 157.1 | 931.4 |
|  30 | 126.5 | 211.4 | 1359.9 |
|  35 | 165.1 | 278.2 | 2332.4 |
|  40 | 235.7 | 362.5 | 3346.8 |
|  45 | 320.1 | 484.0 | 4924.2 |
|  50 | 448.2 | 632.4 | 6644.8 |


---


## Command used to run benchmark

```bash
python benchmarks/benchmark_logistic.py --report benchmarks/logistic_report.md --csv benchmarks/logistic_results.csv --n-runs 20
```
