# Benchmarks

Benchmarking of [firthmodels](https://github.com/jzluo/firthmodels) against implementations
of Firth-penalized logistic regression (R [logistf](https://cran.r-project.org/package=logistf),
[brglm2](https://cran.r-project.org/package=brglm2)) and Cox regression
(R [coxphf](https://cran.r-project.org/package=coxphf)).

## Summary

For the full workflow, firthmodels' Numba backend is **10.5x
faster than logistf** (k=50) and **158x faster than
coxphf** (k=30). Without Numba, the pure NumPy backend is 7.3x
faster than logistf and 23.4x faster than coxphf.

| Workload | Baseline | Speedup at k=5 | Speedup at largest k |
|---|---|--:|--:|
| Logistic: fit + Wald | next-fastest package | 3.7x | 5.2x |
| Logistic: fit + LRT + profile CI | logistf | 2.5x | 10.5x |
| Cox: fit + Wald | coxphf | 11.2x | 130x |
| Cox: fit + LRT + profile CI | coxphf | 8.1x | 158x |

The largest observed deviation across coefficients, profile CI bounds, and p-values is 8.0e-07 (see [Correctness](#correctness)).

Python is timed with time.perf_counter() around each call, after JIT-warming the Numba
backend. R packages are timed in-process with `microbenchmark` inside a single
R session, so R startup and data transfer are excluded, while formula parsing and
model-frame construction are included because logistf and coxphf only
offer formula interfaces. brglm2 is run with `check_aliasing=FALSE` to avoid
extra overhead from its default aliasing check. The reported value is the fastest of 10 runs.

---

## Firth logistic regression

Compared against R [logistf](https://cran.r-project.org/package=logistf) and
[brglm2](https://cran.r-project.org/package=brglm2) on simulated data with
n = 1,000 observations, a 20% target event rate, and
k = 5 to 50 features.

![Logistic benchmark scaling, log time axis](logistic_results.png)

The time axis is log-scale, so a constant vertical gap means a constant speedup ratio.
All values are the minimum observed wall-clock time across repeated runs, in milliseconds.

### Fit + Wald inference

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf | brglm2<br>(AS-mean) | brglm2<br>(MPL-Jeffreys) | numba speedup<br>vs next fastest | numpy speedup<br>vs next fastest |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | **0.53** | 0.92 | <ins>2.0</ins> | 4.1 | 4.1 | 3.7x | 2.1x |
| 10 | **0.74** | 1.2 | <ins>3.5</ins> | 4.8 | 4.8 | 4.7x | 2.8x |
| 15 | **0.77** | 1.3 | <ins>5.4</ins> | 5.6 | 5.5 | 7.0x | 4.2x |
| 20 | **1.1** | 1.6 | 8.0 | 6.4 | <ins>6.4</ins> | 6.1x | 4.0x |
| 25 | **1.5** | 2.3 | 11.7 | 8.1 | <ins>8.0</ins> | 5.4x | 3.6x |
| 30 | **1.7** | 2.5 | 15.1 | <ins>8.8</ins> | 8.9 | 5.3x | 3.5x |
| 35 | **1.8** | 2.8 | 20.3 | 10.5 | <ins>10.5</ins> | 5.9x | 3.7x |
| 40 | **2.1** | 3.2 | 25.3 | 11.9 | <ins>11.8</ins> | 5.5x | 3.7x |
| 45 | **2.8** | 3.8 | 33.4 | 13.3 | <ins>13.2</ins> | 4.8x | 3.5x |
| 50 | **3.3** | 4.4 | 41.5 | 17.3 | <ins>17.1</ins> | 5.2x | 3.9x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns. The two brglm2
fitting methods have nearly identical timings (within 1.1% at every k).

### Full workflow: fit + LRT + profile likelihood CI

brglm2 is not included here because it does not provide penalized LRT
p-values or profile likelihood CIs.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf | numba speedup<br>vs logistf | numpy speedup<br>vs logistf |
|--:|--:|--:|--:|--:|--:|
| 5 | **4.8** | 11.3 | <ins>12.0</ins> | 2.5x | 1.1x |
| 10 | **11.9** | 25.3 | <ins>42.4</ins> | 3.5x | 1.7x |
| 15 | **19.1** | 40.3 | <ins>102.8</ins> | 5.4x | 2.6x |
| 20 | **36.4** | 64.6 | <ins>209.1</ins> | 5.7x | 3.2x |
| 25 | **68.5** | 120.3 | <ins>418.9</ins> | 6.1x | 3.5x |
| 30 | **95.4** | 165.7 | <ins>629.5</ins> | 6.6x | 3.8x |
| 35 | **115.4** | 208.8 | <ins>1,054</ins> | 9.1x | 5.0x |
| 40 | **162.8** | 269.2 | <ins>1,531</ins> | 9.4x | 5.7x |
| 45 | **222.1** | 334.8 | <ins>2,226</ins> | 10.0x | 6.6x |
| 50 | **299.2** | 429.5 | <ins>3,128</ins> | 10.5x | 7.3x |

---

## Firth Cox proportional hazards

Compared against R [coxphf](https://cran.r-project.org/package=coxphf) on
simulated survival data with n = 500 observations, a 20% event rate,
and k = 5 to 30 features.

![Cox benchmark scaling, log time axis](cox_results.png)

The time axis is log-scale, so a constant vertical gap means a constant speedup ratio.
All values are the minimum observed wall-clock time across repeated runs, in milliseconds.

### Fit + Wald inference

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf | numba speedup<br>vs coxphf | numpy speedup<br>vs coxphf |
|--:|--:|--:|--:|--:|--:|
| 5 | **0.30** | 1.5 | <ins>3.4</ins> | 11.2x | 2.2x |
| 10 | **0.41** | 2.3 | <ins>9.0</ins> | 22.0x | 4.0x |
| 15 | **0.57** | 3.3 | <ins>24.9</ins> | 43.4x | 7.5x |
| 20 | **0.86** | 5.7 | <ins>56.1</ins> | 65.4x | 9.8x |
| 25 | **1.2** | 7.9 | <ins>122.2</ins> | 98.7x | 15.6x |
| 30 | **1.6** | 10.5 | <ins>211.6</ins> | 130x | 20.2x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns.

### Full workflow: fit + LRT + profile likelihood CI

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf | numba speedup<br>vs coxphf | numpy speedup<br>vs coxphf |
|--:|--:|--:|--:|--:|--:|
| 5 | **2.2** | 12.8 | <ins>17.9</ins> | 8.1x | 1.4x |
| 10 | **7.1** | 45.6 | <ins>163.0</ins> | 22.9x | 3.6x |
| 15 | **17.4** | 114.1 | <ins>928.5</ins> | 53.4x | 8.1x |
| 20 | **30.8** | 227.2 | <ins>2,547</ins> | 82.6x | 11.2x |
| 25 | **59.4** | 404.1 | <ins>7,418</ins> | 125x | 18.4x |
| 30 | **98.1** | 660.6 | <ins>15,467</ins> | 158x | 23.4x |

---

## Correctness

The benchmark scripts abort if firthmodels disagrees with the R reference by more than
1e-06 on coefficients, profile CI bounds, or p-values (and also cross-check
the numba backend against the numpy backend).

Maximum over all coefficients and all k during the benchmark run (recorded in the results file's run metadata):

| Comparison | Quantity | Max abs. deviation |
|---|---|--:|
| Logistic fit vs logistf | Coefficients | 2.2e-16 |
| Logistic fit vs brglm2 AS_mean | Coefficients | 7.3e-07 |
| Logistic fit vs brglm2 MPL_Jeffreys | Coefficients | 7.3e-07 |
| Logistic full vs logistf | Coefficients | 2.2e-16 |
| Logistic full vs logistf | Profile CI bounds | 5.4e-13 |
| Logistic full vs logistf | LRT p-values | 4.9e-10 |
| Logistic numba vs numpy | Coefficients | 2.2e-16 |
| Cox fit vs coxphf | Coefficients | 8.0e-07 |
| Cox full vs coxphf | Coefficients | 8.0e-07 |
| Cox full vs coxphf | Profile CI bounds | 4.3e-08 |
| Cox full vs coxphf | LRT p-values | 2.6e-08 |
| Cox numba vs numpy | Coefficients | 2.8e-16 |

Note: the benchmark run(s) recorded firthmodels 0.8.1, while the currently installed version is 0.8.2. Timings describe the recorded version.

---

## Environment

Collected at report-generation time on the benchmark machine.

| Component | Version |
|---|---|
| OS | CachyOS |
| CPU | AMD Ryzen 5 5600X 6-Core Processor |
| Python | 3.12.13 |
| firthmodels | 0.8.2 |
| NumPy / SciPy / Numba | 2.3.5 / 1.16.3 / 0.64.0 |
| NumPy BLAS | /usr/lib/libopenblas.so.0.3 (openblas 0.3.34, runtime) |
| R | 4.6.1 |
| logistf / brglm2 / coxphf | 1.26.1 / 1.1.0 / 1.13.4 |
| R BLAS | /usr/lib/libopenblas.so.0.3 |

NumPy and R link to the same BLAS library (`/usr/lib/libopenblas.so.0.3`). BLAS threading is left at library defaults for both stacks.
R packages are compiled from source with `-march=x86-64-v3 -mtune=haswell -O3 -flto=auto`,
so the R timings are not slowed by a conservative build.

<details>
<summary>Full R package compile flags (current configuration)</summary>

```
CFLAGS:  -march=x86-64-v3 -mtune=haswell -O3 -pipe -fno-plt -fexceptions -Wp,-D_FORTIFY_SOURCE=3 -Wformat -Werror=format-security -fstack-clash-protection -fcf-protection -mpclmul -g1 -ffile-prefix-map=/startdir/src=/usr/src/debug/r -flto=auto -ffat-lto-objects
FCFLAGS: -O3 -march=x86-64-v3 -mtune=haswell
```

</details>

---

## Reproducing these results

Requires R with the logistf, brglm2, coxphf, survival, microbenchmark, and
jsonlite packages installed.

```bash
# Run both benchmarks (writes results JSON files holding the timings, R
# reference values, and run metadata), then regenerate the plots and this
# README
benchmarks/run_benchmarks.sh
```
