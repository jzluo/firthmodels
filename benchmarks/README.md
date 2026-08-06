# Benchmarks

Benchmarking of [firthmodels](https://github.com/jzluo/firthmodels) against implementations
of Firth-penalized logistic regression (R [logistf](https://cran.r-project.org/package=logistf),
[brglm2](https://cran.r-project.org/package=brglm2)) and Cox regression
(R [coxphf](https://cran.r-project.org/package=coxphf)).

## Summary

For the full workflow, firthmodels' Numba backend is **10.8x
faster than logistf** (k=50) and **159x faster than
coxphf** (k=30). Without Numba, the pure NumPy backend is 7.5x
faster than logistf and 23.7x faster than coxphf.

| Workload | Baseline | Speedup at k=5 | Speedup at largest k |
|---|---|--:|--:|
| Logistic: fit + Wald | next-fastest package | 3.8x | 5.3x |
| Logistic: fit + LRT + profile CI | logistf | 2.7x | 10.8x |
| Cox: fit + Wald | coxphf | 11.4x | 132x |
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
| 5 | **0.50** | 0.87 | <ins>1.9</ins> | 4.0 | 4.0 | 3.8x | 2.2x |
| 10 | **0.72** | 1.1 | <ins>3.4</ins> | 4.6 | 4.6 | 4.7x | 3.0x |
| 15 | **0.75** | 1.2 | <ins>5.3</ins> | 5.4 | 5.4 | 7.1x | 4.4x |
| 20 | **1.0** | 1.5 | 7.8 | 6.3 | <ins>6.2</ins> | 6.1x | 4.1x |
| 25 | **1.5** | 2.2 | 11.4 | 7.8 | <ins>7.8</ins> | 5.4x | 3.6x |
| 30 | **1.6** | 2.3 | 14.7 | <ins>8.5</ins> | 8.7 | 5.2x | 3.7x |
| 35 | **1.7** | 2.6 | 19.9 | <ins>10.2</ins> | 10.2 | 5.8x | 4.0x |
| 40 | **2.2** | 3.1 | 24.9 | 11.6 | <ins>11.5</ins> | 5.2x | 3.8x |
| 45 | **2.7** | 3.7 | 33.2 | 13.1 | <ins>13.0</ins> | 4.8x | 3.5x |
| 50 | **3.2** | 4.3 | 40.8 | 17.0 | <ins>16.7</ins> | 5.3x | 3.9x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns. The two brglm2
fitting methods have nearly identical timings (within 1.9% at every k).

### Full workflow: fit + LRT + profile likelihood CI

brglm2 is not included here because it does not provide penalized LRT
p-values or profile likelihood CIs.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf | numba speedup<br>vs logistf | numpy speedup<br>vs logistf |
|--:|--:|--:|--:|--:|--:|
| 5 | **4.5** | 10.6 | <ins>11.9</ins> | 2.7x | 1.1x |
| 10 | **11.5** | 23.7 | <ins>41.9</ins> | 3.6x | 1.8x |
| 15 | **18.2** | 38.1 | <ins>101.0</ins> | 5.6x | 2.7x |
| 20 | **34.9** | 61.7 | <ins>206.2</ins> | 5.9x | 3.3x |
| 25 | **66.4** | 117.1 | <ins>418.7</ins> | 6.3x | 3.6x |
| 30 | **92.9** | 150.3 | <ins>624.0</ins> | 6.7x | 4.2x |
| 35 | **111.8** | 190.2 | <ins>1,059</ins> | 9.5x | 5.6x |
| 40 | **166.4** | 261.9 | <ins>1,533</ins> | 9.2x | 5.9x |
| 45 | **213.9** | 326.7 | <ins>2,230</ins> | 10.4x | 6.8x |
| 50 | **290.9** | 420.3 | <ins>3,135</ins> | 10.8x | 7.5x |

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
| 5 | **0.29** | 1.4 | <ins>3.3</ins> | 11.4x | 2.3x |
| 10 | **0.40** | 2.2 | <ins>8.7</ins> | 21.9x | 4.0x |
| 15 | **0.57** | 3.2 | <ins>24.1</ins> | 42.5x | 7.5x |
| 20 | **0.85** | 5.6 | <ins>54.8</ins> | 64.1x | 9.8x |
| 25 | **1.2** | 7.7 | <ins>119.6</ins> | 99.8x | 15.5x |
| 30 | **1.6** | 10.3 | <ins>210.3</ins> | 132x | 20.5x |

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns.

### Full workflow: fit + LRT + profile likelihood CI

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf | numba speedup<br>vs coxphf | numpy speedup<br>vs coxphf |
|--:|--:|--:|--:|--:|--:|
| 5 | **2.1** | 11.9 | <ins>17.4</ins> | 8.4x | 1.5x |
| 10 | **6.8** | 43.6 | <ins>160.0</ins> | 23.4x | 3.7x |
| 15 | **17.2** | 111.3 | <ins>912.8</ins> | 53.0x | 8.2x |
| 20 | **31.4** | 221.8 | <ins>2,516</ins> | 80.2x | 11.3x |
| 25 | **58.5** | 398.4 | <ins>7,281</ins> | 124x | 18.3x |
| 30 | **97.1** | 651.4 | <ins>15,425</ins> | 159x | 23.7x |

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
| firthmodels | 0.8.1 |
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
