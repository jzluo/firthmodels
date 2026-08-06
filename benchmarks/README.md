# Benchmarks

Benchmarking of [firthmodels](https://github.com/jzluo/firthmodels) against implementations
of Firth-penalized logistic regression (R [logistf](https://cran.r-project.org/package=logistf),
[brglm2](https://cran.r-project.org/package=brglm2)) and Cox regression
(R [coxphf](https://cran.r-project.org/package=coxphf)).

## Summary

For the full workflow, firthmodels' Numba backend is **10.8x
faster than logistf** (k=50) and **159x faster than
coxphf** (k=30). Without Numba, the pure NumPy backend is 7.5x
faster than logistf and 23.5x faster than coxphf.

| Workload | Baseline | Speedup at k=5 | Speedup at largest k |
|---|---|--:|--:|
| Logistic: fit + Wald | next-fastest package | 4.6x | 5.2x |
| Logistic: fit + LRT + profile CI | logistf | 2.7x | 10.8x |
| Cox: fit + Wald | coxphf | 11.6x | 129x |
| Cox: fit + LRT + profile CI | coxphf | 8.4x | 159x |

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
| 5 | **0.51** | 0.90 | <ins>2.4</ins> | 4.0 | 4.0 | 4.6x | 2.6x |
| 10 | **0.72** | 1.1 | <ins>3.4</ins> | 4.7 | 4.7 | 4.8x | 3.0x |
| 15 | **0.75** | 1.2 | <ins>5.3</ins> | 5.4 | 5.4 | 7.1x | 4.5x |
| 20 | **1.0** | 1.5 | 7.8 | <ins>6.2</ins> | 6.2 | 6.0x | 4.1x |
| 25 | **1.5** | 2.2 | 11.4 | <ins>7.9</ins> | 7.9 | 5.4x | 3.6x |
| 30 | **1.6** | 2.3 | 14.7 | <ins>8.6</ins> | 8.6 | 5.2x | 3.7x |
| 35 | **1.7** | 2.7 | 19.8 | 10.2 | <ins>10.2</ins> | 5.8x | 3.8x |
| 40 | **2.2** | 3.0 | 24.8 | 11.6 | <ins>11.6</ins> | 5.4x | 3.8x |
| 45 | **2.7** | 3.7 | 33.0 | 13.6 | <ins>13.0</ins> | 4.8x | 3.5x |
| 50 | **3.2** | 4.2 | 40.7 | 17.1 | <ins>16.9</ins> | 5.2x | 4.0x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns. The two brglm2
fitting methods have nearly identical timings (within 4.6% at every k).

### Full workflow: fit + LRT + profile likelihood CI

brglm2 is not included here because it does not provide penalized LRT
p-values or profile likelihood CIs.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf | numba speedup<br>vs logistf | numpy speedup<br>vs logistf |
|--:|--:|--:|--:|--:|--:|
| 5 | **4.5** | 10.4 | <ins>12.0</ins> | 2.7x | 1.2x |
| 10 | **11.5** | 23.7 | <ins>41.9</ins> | 3.6x | 1.8x |
| 15 | **18.3** | 37.7 | <ins>101.1</ins> | 5.5x | 2.7x |
| 20 | **35.0** | 61.3 | <ins>205.2</ins> | 5.9x | 3.3x |
| 25 | **66.5** | 116.3 | <ins>416.4</ins> | 6.3x | 3.6x |
| 30 | **93.5** | 149.5 | <ins>623.8</ins> | 6.7x | 4.2x |
| 35 | **113.1** | 200.4 | <ins>1,058</ins> | 9.3x | 5.3x |
| 40 | **164.3** | 260.9 | <ins>1,536</ins> | 9.3x | 5.9x |
| 45 | **217.3** | 324.1 | <ins>2,216</ins> | 10.2x | 6.8x |
| 50 | **291.7** | 415.5 | <ins>3,136</ins> | 10.8x | 7.5x |

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
| 5 | **0.29** | 1.5 | <ins>3.4</ins> | 11.6x | 2.3x |
| 10 | **0.40** | 2.2 | <ins>8.7</ins> | 22.0x | 4.0x |
| 15 | **0.57** | 3.2 | <ins>24.4</ins> | 42.7x | 7.6x |
| 20 | **0.86** | 5.6 | <ins>55.1</ins> | 64.0x | 9.9x |
| 25 | **1.2** | 7.7 | <ins>120.1</ins> | 99.6x | 15.5x |
| 30 | **1.6** | 10.3 | <ins>206.5</ins> | 129x | 20.1x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns.

### Full workflow: fit + LRT + profile likelihood CI

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf | numba speedup<br>vs coxphf | numpy speedup<br>vs coxphf |
|--:|--:|--:|--:|--:|--:|
| 5 | **2.1** | 11.9 | <ins>17.7</ins> | 8.4x | 1.5x |
| 10 | **6.9** | 43.6 | <ins>161.6</ins> | 23.6x | 3.7x |
| 15 | **17.2** | 111.3 | <ins>918.9</ins> | 53.3x | 8.3x |
| 20 | **30.8** | 220.3 | <ins>2,519</ins> | 81.7x | 11.4x |
| 25 | **58.2** | 398.4 | <ins>7,331</ins> | 126x | 18.4x |
| 30 | **96.4** | 651.3 | <ins>15,323</ins> | 159x | 23.5x |

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
| Logistic full vs logistf | Profile CI bounds | 5.3e-13 |
| Logistic full vs logistf | LRT p-values | 2.7e-10 |
| Logistic numba vs numpy | Coefficients | 2.2e-16 |
| Cox fit vs coxphf | Coefficients | 8.0e-07 |
| Cox full vs coxphf | Coefficients | 8.0e-07 |
| Cox full vs coxphf | Profile CI bounds | 4.3e-08 |
| Cox full vs coxphf | LRT p-values | 2.5e-08 |
| Cox numba vs numpy | Coefficients | 3.9e-16 |

---

## Environment

Collected at report-generation time on the benchmark machine.

| Component | Version |
|---|---|
| OS | CachyOS |
| CPU | AMD Ryzen 5 5600X 6-Core Processor |
| Python | 3.12.13 |
| firthmodels | 0.8.0 |
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
