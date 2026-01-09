#!/usr/bin/env python3
"""Generate combined benchmarks README.md from CSV results.

Usage:
    python benchmarks/generate_readme.py
    python benchmarks/generate_readme.py --logistic-results path/to/logistic.csv
    python benchmarks/generate_readme.py --cox-results path/to/cox.csv
    python benchmarks/generate_readme.py -o path/to/output.md
"""

import argparse
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
LOGISTIC_PLOT = BENCHMARKS_DIR / "logistic_report.png"
COX_PLOT = BENCHMARKS_DIR / "cox_report.png"
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
# Plotting
# -----------------------------------------------------------------------------
def save_logistic_plot(
    df: pd.DataFrame,
    output_path: str,
    version_info: dict[str, str] | None = None,
) -> None:
    """Save logistic benchmark scaling plot to file as 2x2 grid.

    Top row: all libraries
    Bottom row: truncate logistf at crossover point for better readability
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed, skipping plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Add version info as suptitle
    if version_info:
        py_info = f"firthmodels {version_info.get('firthmodels_version', '?')}"
        r_info = (
            f"logistf {version_info.get('logistf_version', '?')}, "
            f"brglm2 {version_info.get('brglm2_version', '?')}"
        )
        fig.suptitle(f"{py_info} | {r_info}", fontsize=10, y=0.98)

    # --- Top row: all libraries, full range ---

    # Fit only (all)
    ax = axes[0, 0]
    ax.plot(df["k"], df["numba_fit_ms"], "o-", label="firthmodels (numba)", linewidth=2)
    ax.plot(df["k"], df["numpy_fit_ms"], "x-", label="firthmodels (numpy)", linewidth=2)
    ax.plot(df["k"], df["logistf_fit_ms"], "s-", label="logistf", linewidth=2)
    ax.plot(
        df["k"], df["brglm2_as_fit_ms"], "^-", label="brglm2 (AS-mean)", linewidth=2
    )
    ax.plot(df["k"], df["brglm2_mpl_fit_ms"], "v-", label="brglm2 (MPL)", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fit Only")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Full workflow (all)
    ax = axes[0, 1]
    ax.plot(
        df["k"], df["numba_full_ms"], "o-", label="firthmodels (numba)", linewidth=2
    )
    ax.plot(
        df["k"], df["numpy_full_ms"], "x-", label="firthmodels (numpy)", linewidth=2
    )
    ax.plot(df["k"], df["logistf_full_ms"], "s-", label="logistf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Full Workflow")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom row: limit y-axis to focus on faster implementations ---
    # logistf line may go off the top of the chart
    last_row = df.iloc[-1]

    # Fit-only: y-limit based on max of numpy/brglm2 at final k
    fit_max = max(
        last_row["numpy_fit_ms"],
        last_row["brglm2_as_fit_ms"],
        last_row["brglm2_mpl_fit_ms"],
    )
    locator = MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])
    fit_ticks = locator.tick_values(0, fit_max * 1.1)
    fit_ylim = fit_ticks[fit_ticks >= fit_max][0]

    # Full workflow: y-limit based on numpy at final k
    full_max = last_row["numpy_full_ms"]
    full_ticks = locator.tick_values(0, full_max * 1.1)
    full_ylim = full_ticks[full_ticks >= full_max][0]

    # Fit only (zoomed)
    ax = axes[1, 0]
    ax.plot(df["k"], df["numba_fit_ms"], "o-", label="firthmodels (numba)", linewidth=2)
    ax.plot(df["k"], df["numpy_fit_ms"], "x-", label="firthmodels (numpy)", linewidth=2)
    ax.plot(df["k"], df["logistf_fit_ms"], "s-", label="logistf", linewidth=2)
    ax.plot(
        df["k"], df["brglm2_as_fit_ms"], "^-", label="brglm2 (AS-mean)", linewidth=2
    )
    ax.plot(df["k"], df["brglm2_mpl_fit_ms"], "v-", label="brglm2 (MPL)", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, fit_ylim)
    ax.set_title("Fit Only (zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Full workflow (zoomed)
    ax = axes[1, 1]
    ax.plot(
        df["k"], df["numba_full_ms"], "o-", label="firthmodels (numba)", linewidth=2
    )
    ax.plot(
        df["k"], df["numpy_full_ms"], "x-", label="firthmodels (numpy)", linewidth=2
    )
    ax.plot(df["k"], df["logistf_full_ms"], "s-", label="logistf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, full_ylim)
    ax.set_title("Full Workflow (zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}", file=sys.stderr)


def save_cox_plot(
    df: pd.DataFrame,
    output_path: str,
    version_info: dict[str, str] | None = None,
) -> None:
    """Save Cox benchmark scaling plot to file as 2x2 grid."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed, skipping plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    if version_info:
        py_info = f"firthmodels {version_info.get('firthmodels_version', '?')}"
        r_info = f"coxphf {version_info.get('coxphf_version', '?')}"
        fig.suptitle(f"{py_info} | {r_info}", fontsize=10, y=0.98)

    # --- Top row: all libraries, full range ---
    ax = axes[0, 0]
    ax.plot(df["k"], df["numba_fit_ms"], "o-", label="firthmodels (numba)", linewidth=2)
    ax.plot(df["k"], df["numpy_fit_ms"], "x-", label="firthmodels (numpy)", linewidth=2)
    ax.plot(df["k"], df["coxphf_fit_ms"], "s-", label="coxphf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Fit Only")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        df["k"], df["numba_full_ms"], "o-", label="firthmodels (numba)", linewidth=2
    )
    ax.plot(
        df["k"], df["numpy_full_ms"], "x-", label="firthmodels (numpy)", linewidth=2
    )
    ax.plot(df["k"], df["coxphf_full_ms"], "s-", label="coxphf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Full Workflow")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Bottom row: zoomed ---
    last_row = df.iloc[-1]
    locator = MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])

    fit_max = max(last_row["numpy_fit_ms"], last_row["numba_fit_ms"])
    fit_ticks = locator.tick_values(0, fit_max * 1.1)
    fit_ylim = fit_ticks[fit_ticks >= fit_max][0]

    full_max = last_row["numpy_full_ms"]
    full_ticks = locator.tick_values(0, full_max * 1.1)
    full_ylim = full_ticks[full_ticks >= full_max][0]

    ax = axes[1, 0]
    ax.plot(df["k"], df["numba_fit_ms"], "o-", label="firthmodels (numba)", linewidth=2)
    ax.plot(df["k"], df["numpy_fit_ms"], "x-", label="firthmodels (numpy)", linewidth=2)
    ax.plot(df["k"], df["coxphf_fit_ms"], "s-", label="coxphf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, fit_ylim)
    ax.set_title("Fit Only (zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(
        df["k"], df["numba_full_ms"], "o-", label="firthmodels (numba)", linewidth=2
    )
    ax.plot(
        df["k"], df["numpy_full_ms"], "x-", label="firthmodels (numpy)", linewidth=2
    )
    ax.plot(df["k"], df["coxphf_full_ms"], "s-", label="coxphf", linewidth=2)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, full_ylim)
    ax.set_title("Full Workflow (zoomed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}", file=sys.stderr)


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
python benchmarks/benchmark_logistic.py -o benchmarks/logistic_results.csv

# Run Cox PH benchmarks
python benchmarks/benchmark_cox.py -o benchmarks/cox_results.csv

# Generate plots and README
python benchmarks/generate_readme.py
```
"""

    return readme


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined benchmarks README.md from CSV results"
    )
    parser.add_argument(
        "--logistic-results",
        type=str,
        default=str(BENCHMARKS_DIR / "logistic_results.csv"),
        help="Path to logistic results CSV (default: benchmarks/logistic_results.csv)",
    )
    parser.add_argument(
        "--cox-results",
        type=str,
        default=str(BENCHMARKS_DIR / "cox_results.csv"),
        help="Path to Cox results CSV (default: benchmarks/cox_results.csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=str(BENCHMARKS_DIR / "README.md"),
        help="Output path for README (default: benchmarks/README.md)",
    )
    args = parser.parse_args()

    logistic_csv = Path(args.logistic_results)
    cox_csv = Path(args.cox_results)

    # Check that CSV files exist
    if not logistic_csv.exists():
        print(f"Error: {logistic_csv} not found", file=sys.stderr)
        print("Run benchmark_logistic.py with -o first", file=sys.stderr)
        sys.exit(1)

    if not cox_csv.exists():
        print(f"Error: {cox_csv} not found", file=sys.stderr)
        print("Run benchmark_cox.py with -o first", file=sys.stderr)
        sys.exit(1)

    # Load data
    print("Loading benchmark results...", file=sys.stderr)
    logistic_df = pd.read_csv(logistic_csv)
    cox_df = pd.read_csv(cox_csv)

    # Gather version info
    print("Collecting version info...", file=sys.stderr)
    version_info = get_python_version_info()
    version_info.update(get_r_version_info())

    # Generate plots
    print("Generating plots...", file=sys.stderr)
    save_logistic_plot(logistic_df, str(LOGISTIC_PLOT), version_info)
    save_cox_plot(cox_df, str(COX_PLOT), version_info)

    # Generate README
    print("Generating README...", file=sys.stderr)
    readme_content = generate_readme(logistic_df, cox_df, version_info)

    # Write output
    with open(args.out, "w") as f:
        f.write(readme_content)

    print(f"README written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
