#!/usr/bin/env python3
"""Generate combined benchmarks README.md from CSV results.

Usage:
    python benchmarks/generate_readme.py

Reads:
    benchmarks/logistic_results.csv
    benchmarks/cox_results.csv

Outputs:
    benchmarks/README.md
"""

import platform as plat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BENCHMARKS_DIR = Path(__file__).parent
LOGISTIC_CSV = BENCHMARKS_DIR / "logistic_results.csv"
COX_CSV = BENCHMARKS_DIR / "cox_results.csv"
README_PATH = BENCHMARKS_DIR / "README.md"

# Benchmark parameters (must match the individual benchmark scripts)
LOGISTIC_N_SAMPLES = 1000
LOGISTIC_EVENT_RATE = 0.20
LOGISTIC_MAX_ITER = 50
LOGISTIC_TOL = 1e-6
LOGISTIC_COEF_TOL = 1e-6
LOGISTIC_CI_TOL = 1e-6
LOGISTIC_PVAL_TOL = 1e-6

COX_N_SAMPLES = 500
COX_EVENT_RATE = 0.20
COX_MAX_ITER = 50
COX_XTOL = 1e-6
COX_GTOL = 1e-4
COX_COEF_TOL = 1e-6
COX_CI_TOL = 1e-6
COX_PVAL_TOL = 1e-6


# -----------------------------------------------------------------------------
# System and version info (adapted from benchmark scripts)
# -----------------------------------------------------------------------------
def get_system_info() -> dict[str, str]:
    """Get OS and CPU info in a cross-platform way."""
    info = {}
    system = plat.system()

    # OS info
    if system == "Linux":
        try:
            os_release = plat.freedesktop_os_release()
            info["os"] = os_release.get("PRETTY_NAME", f"Linux {plat.release()}")
        except OSError:
            info["os"] = f"Linux {plat.release()}"
    elif system == "Darwin":
        mac_ver = plat.mac_ver()[0]
        info["os"] = f"macOS {mac_ver}"
    elif system == "Windows":
        info["os"] = f"Windows {plat.version()}"
    else:
        info["os"] = plat.platform()

    # CPU info
    cpu = plat.processor()
    unhelpful = not cpu or cpu in ("arm", "arm64", "x86_64", "i386", "AMD64")

    if unhelpful:
        try:
            if system == "Linux":
                result = subprocess.run(
                    ["grep", "-m1", "model name", "/proc/cpuinfo"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    cpu = result.stdout.split(":")[1].strip()
            elif system == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    cpu = result.stdout.strip()
            elif system == "Windows":
                import winreg  # type: ignore[import-not-found]

                key = winreg.OpenKey(  # type: ignore[attr-defined]
                    winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined]
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0]  # type: ignore[attr-defined]
        except Exception:
            pass

    if not cpu:
        cpu = plat.machine()

    info["cpu"] = cpu
    return info


def get_python_version_info() -> dict[str, str]:
    """Get version info for Python libraries and BLAS backend."""
    import sys
    from importlib.metadata import version

    info = get_system_info()
    info["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    info["firthmodels_version"] = version("firthmodels")
    info["numpy_version"] = version("numpy")
    info["scipy_version"] = version("scipy")

    try:
        info["numba_version"] = version("numba")
    except Exception:
        info["numba_version"] = "not installed"

    # NumPy BLAS info
    try:
        blas_info = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
        if blas_info.get("found"):
            lib_dir = blas_info.get("lib directory", "")
            name = blas_info.get("name", "unknown")
            ver = blas_info.get("version", "")
            info["numpy_blas"] = f"{lib_dir} ({name} {ver})".strip()
        else:
            info["numpy_blas"] = "unknown"
    except Exception:
        info["numpy_blas"] = "unknown"

    return info


def get_r_version_info() -> dict[str, str]:
    """Get R package versions and BLAS backend."""
    r_script = """
    cat("R_VERSION:", paste(R.version$major, R.version$minor, sep="."), "\\n")
    tryCatch({
        cat("LOGISTF_VERSION:", as.character(packageVersion("logistf")), "\\n")
    }, error=function(e) cat("LOGISTF_VERSION: unknown\\n"))
    tryCatch({
        cat("BRGLM2_VERSION:", as.character(packageVersion("brglm2")), "\\n")
    }, error=function(e) cat("BRGLM2_VERSION: unknown\\n"))
    tryCatch({
        cat("COXPHF_VERSION:", as.character(packageVersion("coxphf")), "\\n")
    }, error=function(e) cat("COXPHF_VERSION: unknown\\n"))
    si <- sessionInfo()
    cat("R_BLAS:", si$BLAS, "\\n")
    """

    info = {
        "r_version": "unknown",
        "logistf_version": "unknown",
        "brglm2_version": "unknown",
        "coxphf_version": "unknown",
        "r_blas": "unknown",
    }

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("R_VERSION:"):
                    info["r_version"] = line.split(":")[1].strip()
                elif line.startswith("LOGISTF_VERSION:"):
                    info["logistf_version"] = line.split(":")[1].strip()
                elif line.startswith("BRGLM2_VERSION:"):
                    info["brglm2_version"] = line.split(":")[1].strip()
                elif line.startswith("COXPHF_VERSION:"):
                    info["coxphf_version"] = line.split(":")[1].strip()
                elif line.startswith("R_BLAS:"):
                    info["r_blas"] = line.split(":", 1)[1].strip()
    except FileNotFoundError:
        pass  # R not installed, keep defaults

    return info


# -----------------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------------
def generate_logistic_section(
    df: pd.DataFrame, version_info: dict[str, str], n_runs: int
) -> str:
    """Generate the logistic regression section of the README."""
    firthmodels_ver = version_info.get("firthmodels_version", "unknown")
    logistf_ver = version_info.get("logistf_version", "unknown")
    brglm2_ver = version_info.get("brglm2_version", "unknown")
    numpy_blas = version_info.get("numpy_blas", "unknown")
    r_blas = version_info.get("r_blas", "unknown")

    section = f"""## Firth Logistic Regression

Comparison of [firthmodels](https://github.com/jzluo/firthmodels),
R [brglm2](https://cran.r-project.org/web/packages/brglm2/index.html),
and R [logistf](https://cran.r-project.org/web/packages/logistf/index.html)
for Firth-penalized logistic regression.

### Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels** | {firthmodels_ver} | {numpy_blas} |
| **brglm2** | {brglm2_ver} | {r_blas} |
| **logistf** | {logistf_ver} | {r_blas} |

### Configuration

| Parameter | Value |
|-----------|-------|
| Observations (n) | {LOGISTIC_N_SAMPLES:,} |
| Event rate | {LOGISTIC_EVENT_RATE:.0%} |
| Features (k) | {", ".join(str(k) for k in df["k"].tolist())} |
| Runs per config | {n_runs} |
| Solver max_iter | {LOGISTIC_MAX_ITER} |
| Solver tolerance | {LOGISTIC_TOL} |

brglm2 runs with `check_aliasing=FALSE` since the benchmark data is guaranteed full rank.

All implementations agree within chosen tolerance (coefficients {LOGISTIC_COEF_TOL}, CIs {LOGISTIC_CI_TOL}, p-values {LOGISTIC_PVAL_TOL}).

### Results

![Logistic benchmark scaling plot](logistic_report.png)

#### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | brglm2<br>(AS-mean) | brglm2<br>(MPL_Jeffreys) | logistf |
|--:|------:|------:|------------:|-------------:|--------:|
"""

    for _, row in df.iterrows():
        section += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_fit_ms']:.1f} | "
            f"{row['numpy_fit_ms']:.1f} | "
            f"{row['brglm2_as_fit_ms']:.1f} | "
            f"{row['brglm2_mpl_fit_ms']:.1f} | "
            f"{row['logistf_fit_ms']:.1f} |\n"
        )

    section += """
#### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | logistf |
|--:|------:|------:|--------:|
"""

    for _, row in df.iterrows():
        section += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_full_ms']:.1f} | "
            f"{row['numpy_full_ms']:.1f} | "
            f"{row['logistf_full_ms']:.1f} |\n"
        )

    return section


def generate_cox_section(
    df: pd.DataFrame, version_info: dict[str, str], n_runs: int
) -> str:
    """Generate the Cox PH section of the README."""
    firthmodels_ver = version_info.get("firthmodels_version", "unknown")
    coxphf_ver = version_info.get("coxphf_version", "unknown")
    numpy_blas = version_info.get("numpy_blas", "unknown")
    r_blas = version_info.get("r_blas", "unknown")

    section = f"""## Firth Cox Proportional Hazards

Comparison of [firthmodels](https://github.com/jzluo/firthmodels)
and [coxphf](https://cran.r-project.org/web/packages/coxphf/index.html)
for Firth-penalized Cox proportional hazards regression.

### Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels** | {firthmodels_ver} | {numpy_blas} |
| **coxphf** | {coxphf_ver} | {r_blas} |

### Configuration

| Parameter | Value |
|-----------|-------|
| Observations (n) | {COX_N_SAMPLES:,} |
| Event rate | {COX_EVENT_RATE:.0%} |
| Features (k) | {", ".join(str(k) for k in df["k"].tolist())} |
| Runs per config | {n_runs} |
| Solver max_iter | {COX_MAX_ITER} |
| Solver xtol | {COX_XTOL} |
| Solver gtol | {COX_GTOL} |

All implementations agree within chosen tolerance (coefficients {COX_COEF_TOL}, CIs {COX_CI_TOL}, p-values {COX_PVAL_TOL}).

### Results

![Cox benchmark scaling plot](cox_report.png)

#### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
"""

    for _, row in df.iterrows():
        section += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_fit_ms']:.1f} | "
            f"{row['numpy_fit_ms']:.1f} | "
            f"{row['coxphf_fit_ms']:.1f} |\n"
        )

    section += """
#### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
"""

    for _, row in df.iterrows():
        section += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_full_ms']:.1f} | "
            f"{row['numpy_full_ms']:.1f} | "
            f"{row['coxphf_full_ms']:.1f} |\n"
        )

    return section


def generate_readme(
    logistic_df: pd.DataFrame,
    cox_df: pd.DataFrame,
    version_info: dict[str, str],
    logistic_n_runs: int = 20,
    cox_n_runs: int = 10,
) -> str:
    """Generate the combined README content."""
    os_info = version_info.get("os", "unknown")
    cpu_info = version_info.get("cpu", "unknown")
    python_ver = version_info.get("python_version", "unknown")
    numpy_ver = version_info.get("numpy_version", "unknown")
    scipy_ver = version_info.get("scipy_version", "unknown")
    numba_ver = version_info.get("numba_version", "unknown")
    r_ver = version_info.get("r_version", "unknown")

    readme = f"""# Benchmarks

Benchmarking of implementations of Firth-penalized logistic regression and Cox regression.

## Environment

| | |
|-----|-----|
| **OS** | {os_info} |
| **CPU** | {cpu_info} |
| **Python** | {python_ver} |
| **NumPy** | {numpy_ver} |
| **SciPy** | {scipy_ver} |
| **Numba** | {numba_ver} |
| **R** | {r_ver} |

---

{generate_logistic_section(logistic_df, version_info, logistic_n_runs)}

---

{generate_cox_section(cox_df, version_info, cox_n_runs)}

---

## Reproducing These Results

```bash
# Run logistic regression benchmarks
python benchmarks/benchmark_logistic.py \\
    --csv benchmarks/logistic_results.csv \\
    --plot benchmarks/logistic_report.png \\
    --n-runs 30

# Run Cox PH benchmarks
python benchmarks/benchmark_cox.py \\
    --csv benchmarks/cox_results.csv \\
    --plot benchmarks/cox_report.png \\
    --n-runs 15

# Regenerate this README
python benchmarks/generate_readme.py
```
"""

    return readme


def main():
    # Check that CSV files exist
    if not LOGISTIC_CSV.exists():
        print(f"Error: {LOGISTIC_CSV} not found", file=sys.stderr)
        print("Run benchmark_logistic.py with --csv first", file=sys.stderr)
        sys.exit(1)

    if not COX_CSV.exists():
        print(f"Error: {COX_CSV} not found", file=sys.stderr)
        print("Run benchmark_cox.py with --csv first", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading benchmark results...", file=sys.stderr)
    logistic_df = pd.read_csv(LOGISTIC_CSV)
    cox_df = pd.read_csv(COX_CSV)

    # Gather version info
    print("Collecting version info...", file=sys.stderr)
    version_info = get_python_version_info()
    version_info.update(get_r_version_info())

    # Generate README
    print("Generating README...", file=sys.stderr)
    readme_content = generate_readme(logistic_df, cox_df, version_info)

    # Write output
    with open(README_PATH, "w") as f:
        f.write(readme_content)

    print(f"README written to {README_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
