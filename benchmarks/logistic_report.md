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

| Library | Language | Version | BLAS |
|---------|----------|---------|------|
| **firthmodels (numba)** | Python | @ commit 83ce231 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **firthmodels (numpy)** | Python | @ commit 83ce231 | /usr/lib/x86_64-linux-gnu/openblas-pthread/ (openblas 0.3.26) |
| **brglm2 (AS-mean)** | R | 1.0.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |
| **brglm2 (MPL_Jeffreys)** | R | 1.0.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |
| **logistf** | R | 1.26.1 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |

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

All implementations produce numerically equivalent results (verified with tolerances: coefficients 1e-06, CIs 1e-06, p-values 1e-06).

## Results

![Benchmark scaling plot](logistic_report.png)

### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | numba | numpy | brglm2<br>(AS-mean) | brglm2<br>(MPL_Jeffreys) | logistf |
|--:|------:|------:|------------:|-------------:|--------:|
|   5 | 0.5 | 1.3 | 4.4 | 4.4 | 2.2 |
|  10 | 1.1 | 1.6 | 8.4 | 8.0 | 4.6 |
|  15 | 1.2 | 1.9 | 9.6 | 8.7 | 7.1 |
|  20 | 1.4 | 2.2 | 10.9 | 10.1 | 17.5 |
|  25 | 1.8 | 3.0 | 12.6 | 12.6 | 25.1 |
|  30 | 2.0 | 2.8 | 14.4 | 14.4 | 29.8 |
|  35 | 2.2 | 3.2 | 15.8 | 17.2 | 44.7 |
|  40 | 3.0 | 3.9 | 17.8 | 17.5 | 55.2 |
|  45 | 3.5 | 4.6 | 19.6 | 20.3 | 72.5 |
|  50 | 4.4 | 5.9 | 24.2 | 25.1 | 93.8 |

### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | numba | numpy | logistf |
|--:|------:|------:|--------:|
|   5 | 5.2 | 11.5 | 13.8 |
|  10 | 23.4 | 43.3 | 83.2 |
|  15 | 34.1 | 68.1 | 157.9 |
|  20 | 54.9 | 98.8 | 465.1 |
|  25 | 91.9 | 161.3 | 969.8 |
|  30 | 128.2 | 204.8 | 1380.1 |
|  35 | 163.4 | 281.9 | 2311.6 |
|  40 | 236.4 | 377.9 | 3712.8 |
|  45 | 326.1 | 511.5 | 5661.9 |
|  50 | 467.0 | 651.5 | 6635.7 |


---


## Command used to run benchmark

```bash
python benchmarks/benchmark_logistic.py --report benchmarks/logistic_report.md --csv benchmarks/logistic_results.csv --n-runs 20
```
