# Firth Cox Proportional Hazards Benchmark

Comparison of [firthmodels](https://github.com/jzluo/firthmodels)
and [coxphf](https://cran.r-project.org/web/packages/coxphf/index.html)
for Firth-penalized Cox proportional hazards regression.

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
| **coxphf** | 1.13.4 | /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 |

## Benchmark Configuration

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

## Results

![Benchmark scaling plot](cox_report.png)

### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
|   5 | 2.9 | 4.3 | 3.5 |
|  10 | 1.8 | 5.3 | 22.2 |
|  15 | 2.1 | 6.9 | 49.8 |
|  20 | 4.2 | 13.8 | 114.2 |
|  25 | 5.3 | 34.2 | 225.4 |
|  30 | 7.5 | 54.9 | 388.8 |

### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
|   5 | 4.5 | 19.9 | 23.7 |
|  10 | 13.1 | 65.8 | 364.6 |
|  15 | 36.3 | 176.7 | 1691.5 |
|  20 | 115.0 | 526.8 | 5681.9 |
|  25 | 191.1 | 2221.7 | 16089.0 |
|  30 | 297.1 | 3950.2 | 34704.4 |

---

## Command used to run benchmark

```bash
python benchmarks/benchmark_cox.py --report benchmarks/cox_report.md --csv benchmarks/cox_results.csv --n-runs 10
```
