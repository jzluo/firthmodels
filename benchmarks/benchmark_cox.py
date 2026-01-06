"""
Benchmark: firthmodels FirthCoxPH vs R coxphf.

Measures scaling behavior as k increases.

Run with: python benchmark_cox.py

Requires: R with coxphf, survival, microbenchmark, jsonlite packages installed.
"""

import argparse
import json
import platform as plat
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from firthmodels import FirthCoxPH

# -----------------------------------------------------------------------------
# Benchmark parameters
# -----------------------------------------------------------------------------
N_SAMPLES = 500
EVENT_RATE = 0.20
K_VALUES = list(range(5, 35, 5))
N_RUNS = 10
BASE_SEED = 0

# Solver parameters
MAX_ITER = 50
MAX_HALFSTEP = 5
XTOL = 1e-6
GTOL = 1e-4

# Tolerance for checking numerical agreement between the implementations
COEF_TOL = 1e-6
CI_TOL = 1e-6
PVAL_TOL = 1e-6


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def make_benchmark_data(
    n: int, k: int, event_rate: float = 0.20, base_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic survival data for Firth Cox PH benchmarking.

    Uses exponential survival times with hazard proportional to exp(X @ beta).

    Parameters
    ----------
    n : int
        Number of observations
    k : int
        Number of features
    event_rate : float
        Target proportion of events (uncensored observations)
    base_seed : int
        Base random seed (combined with k for reproducibility)

    Returns
    -------
    X : ndarray of shape (n, k)
        Feature matrix (standardized)
    time : ndarray of shape (n,)
        Observed time (min of event time and censoring time)
    event : ndarray of shape (n,)
        Event indicator (True = event, False = censored)
    """
    rng = np.random.default_rng(base_seed * 1000 + k)

    # Independent standard normal features
    X = rng.standard_normal((n, k))

    # Small coefficients: keeps Var(X @ beta) constant across k
    beta = rng.normal(0, 0.1 / np.sqrt(k), size=k)
    eta = X @ beta

    # Exponential survival times: T ~ Exp(lambda * exp(eta))
    baseline_hazard = 0.1
    U = rng.uniform(0, 1, n)
    T = -np.log(U) / (baseline_hazard * np.exp(eta))

    # Censoring to achieve target event rate
    censor_quantile = 1 - event_rate
    censor_time = np.quantile(T, censor_quantile)

    time = np.minimum(T, censor_time)
    event = T <= censor_time

    return X, time, event


# -----------------------------------------------------------------------------
# R package validation
# -----------------------------------------------------------------------------
def validate_r_packages() -> None:
    """Check that R and required packages are installed before starting."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError("Rscript returned non-zero exit code")
    except FileNotFoundError:
        raise RuntimeError(
            "Rscript not found. Install R and ensure it's on PATH."
        ) from None

    r_script = """
    for (pkg in c("coxphf", "survival", "microbenchmark", "jsonlite")) {
        if (!requireNamespace(pkg, quietly=TRUE)) {
            stop(paste("Package not installed:", pkg))
        }
    }
    cat("OK\\n")
    """
    result = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
    if result.returncode != 0 or "OK" not in result.stdout:
        raise RuntimeError(
            f"Missing R packages. Install with: "
            f"install.packages(c('coxphf', 'survival', 'microbenchmark', 'jsonlite'))\n"
            f"R stderr: {result.stderr}"
        )


# -----------------------------------------------------------------------------
# Python (firthmodels) benchmarks
# -----------------------------------------------------------------------------
def warmup_numba() -> None:
    """Global warmup to ensure all numba code paths are JIT compiled."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3))
    time = rng.exponential(1, 100)
    event = rng.binomial(1, 0.5, 100).astype(bool)

    model = FirthCoxPH(backend="numba")
    model.fit(X, (event, time))
    model.lrt()
    model.conf_int(method="pl")


def make_model(backend: Literal["auto", "numba", "numpy"] = "numba") -> FirthCoxPH:
    """Create FirthCoxPH model with benchmark parameters."""
    return FirthCoxPH(
        max_iter=MAX_ITER,
        max_halfstep=MAX_HALFSTEP,
        gtol=GTOL,
        xtol=XTOL,
        backend=backend,
    )


def time_python_fit(
    X: np.ndarray,
    surv_time: np.ndarray,
    event: np.ndarray,
    n_runs: int = N_RUNS,
    backend: Literal["auto", "numba", "numpy"] = "numba",
) -> tuple[np.ndarray, np.ndarray]:
    """Time firthmodels fit only (no LRT, no profile CI)."""
    import time as time_module

    times = []
    y = (event, surv_time)
    for _ in range(n_runs):
        model = make_model(backend=backend)
        start = time_module.perf_counter()
        model.fit(X, y)
        times.append(time_module.perf_counter() - start)

    return np.array(times), model.coef_


def time_python_full(
    X: np.ndarray,
    surv_time: np.ndarray,
    event: np.ndarray,
    n_runs: int = N_RUNS,
    backend: Literal["auto", "numba", "numpy"] = "numba",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time firthmodels fit + LRT + profile CI."""
    import time as time_module

    times = []
    y = (event, surv_time)
    for _ in range(n_runs):
        model = make_model(backend=backend)
        start = time_module.perf_counter()
        model.fit(X, y)
        model.lrt()
        ci = model.conf_int(method="pl", max_iter=MAX_ITER, tol=GTOL)
        times.append(time_module.perf_counter() - start)

    return np.array(times), model.coef_, ci, model.lrt_pvalues_


# -----------------------------------------------------------------------------
# R benchmarks
# -----------------------------------------------------------------------------
def prepare_data_files(k_values: list[int], tmpdir: Path) -> None:
    """Generate and save CSV files for R benchmarks."""
    for k in k_values:
        X, surv_time, event = make_benchmark_data(N_SAMPLES, k, EVENT_RATE, BASE_SEED)
        csv_path = tmpdir / f"data_k{k}.csv"
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(k)])
        df.insert(0, "event", event.astype(int))
        df.insert(0, "time", surv_time)
        df.to_csv(csv_path, index=False)


def run_r_coxphf(
    k_values: list[int], data_dir: Path, n_runs: int = N_RUNS
) -> dict[int, dict]:
    """Run coxphf benchmarks for all k values.

    Returns dict mapping k -> results dict with numpy arrays.
    """
    script_path = Path(__file__).parent / "coxphf_bench.R"
    k_str = ",".join(str(k) for k in k_values)

    result = subprocess.run(
        [
            "Rscript",
            str(script_path),
            str(data_dir),
            k_str,
            str(n_runs),
            str(MAX_ITER),
            str(MAX_HALFSTEP),
            str(XTOL),
            str(GTOL),
        ],
        stdout=subprocess.PIPE,
        stderr=None,  # Let stderr go to terminal for progress
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("R coxphf batch script failed")

    raw = json.loads(result.stdout)

    # Convert to expected format with numpy arrays and int keys
    results = {}
    for k_str, data in raw.items():
        k = int(k_str)
        results[k] = {
            "fit_times": np.array(data["fit_times"]),
            "fit_coef": np.array(data["fit_coef"]),
            "full_times": np.array(data["full_times"]),
            "full_coef": np.array(data["full_coef"]),
            "full_ci": np.column_stack([data["ci_lower"], data["ci_upper"]]),
            "full_pval": np.array(data["full_pval"]),
        }

    return results


# -----------------------------------------------------------------------------
# Numerical agreement verification
# -----------------------------------------------------------------------------
def check_agreement(
    k: int,
    py_coef: np.ndarray,
    r_coef: np.ndarray,
    label: str,
    py_ci: np.ndarray | None = None,
    r_ci: np.ndarray | None = None,
    py_pval: np.ndarray | None = None,
    r_pval: np.ndarray | None = None,
) -> None:
    """Check that Python and R results agree within tolerance."""
    # Check coefficients
    coef_diff = np.max(np.abs(py_coef - r_coef))
    if coef_diff > COEF_TOL:
        raise AssertionError(f"k={k} ({label}): coef diff {coef_diff:.2e} > {COEF_TOL}")

    # Check CIs if provided
    if py_ci is not None and r_ci is not None:
        ci_diff = np.max(np.abs(py_ci - r_ci))
        if ci_diff > CI_TOL:
            raise AssertionError(f"k={k} ({label}): CI diff {ci_diff:.2e} > {CI_TOL}")

    # Check p-values if provided
    if py_pval is not None and r_pval is not None:
        pval_diff = np.max(np.abs(py_pval - r_pval))
        if pval_diff > PVAL_TOL:
            raise AssertionError(
                f"k={k} ({label}): p-value diff {pval_diff:.2e} > {PVAL_TOL}"
            )


# -----------------------------------------------------------------------------
# Version and system info
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
    """Get version info for Python libraries, BLAS backend, and system info."""
    from importlib.metadata import version

    info = get_system_info()
    info["firthmodels_version"] = version("firthmodels")

    # NumPy BLAS info
    try:
        blas_info = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
        if blas_info.get("found"):
            lib_dir = blas_info.get("lib directory", "")
            name = blas_info.get("name", "unknown")
            version = blas_info.get("version", "")
            info["numpy_blas"] = f"{lib_dir} ({name} {version})".strip()
        else:
            info["numpy_blas"] = "unknown"
    except Exception:
        info["numpy_blas"] = "unknown"

    return info


def get_r_version_info() -> dict[str, str]:
    """Get R package versions and BLAS backend."""
    r_script = """
    cat("COXPHF_VERSION:", as.character(packageVersion("coxphf")), "\\n")
    si <- sessionInfo()
    cat("R_BLAS:", si$BLAS, "\\n")
    """

    result = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)

    info = {}
    if result.returncode == 0:
        blas_path = ""
        for line in result.stdout.strip().split("\n"):
            if line.startswith("COXPHF_VERSION:"):
                info["coxphf_version"] = line.split(":")[1].strip()
            elif line.startswith("R_BLAS:"):
                blas_path = line.split(":", 1)[1].strip()

        info["r_blas"] = blas_path if blas_path else "unknown"
    else:
        info["coxphf_version"] = "unknown"
        info["r_blas"] = "unknown"

    return info


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------
def compute_min(times: np.ndarray) -> float:
    """Return minimum time (best achievable performance)."""
    return float(np.min(times))


# -----------------------------------------------------------------------------
# Main benchmark runner
# -----------------------------------------------------------------------------
def run_benchmarks(
    k_values: list[int] = K_VALUES,
    n_runs: int = N_RUNS,
    skip_verification: bool = False,
    coxphf_reduce_after: int | None = 25,
    run_firthmodels: bool = True,
    run_coxphf: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run all benchmarks and return results as DataFrame."""
    version_info = get_python_version_info()

    if run_coxphf:
        print("Validating R packages...", file=sys.stderr, flush=True)
        validate_r_packages()
        print("Collecting version info...", file=sys.stderr, flush=True)
        version_info.update(get_r_version_info())

    if run_firthmodels:
        print("Warming up numba JIT...", file=sys.stderr, flush=True)
        warmup_numba()

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        print("Generating data files...", file=sys.stderr, flush=True)
        prepare_data_files(k_values, tmpdir)

        # --- Python benchmarks (both backends) ---
        py_results: dict[str, dict[int, dict]] = {"numba": {}, "numpy": {}}
        if run_firthmodels:
            for backend in ["numba", "numpy"]:
                print(
                    f"Running Python (firthmodels, {backend}) benchmarks...",
                    file=sys.stderr,
                    flush=True,
                )
                for k in k_values:
                    print(f"  k={k}...", file=sys.stderr, flush=True)
                    X, surv_time, event = make_benchmark_data(
                        N_SAMPLES, k, EVENT_RATE, BASE_SEED
                    )

                    py_fit_times, py_fit_coef = time_python_fit(
                        X,
                        surv_time,
                        event,
                        n_runs,
                        backend,  # type: ignore[arg-type]
                    )
                    py_full_times, py_full_coef, py_ci, py_pval = time_python_full(
                        X,
                        surv_time,
                        event,
                        n_runs,
                        backend,  # type: ignore[arg-type]
                    )

                    py_results[backend][k] = {
                        "fit_times": py_fit_times,
                        "fit_coef": py_fit_coef,
                        "full_times": py_full_times,
                        "full_coef": py_full_coef,
                        "full_ci": py_ci,
                        "full_pval": py_pval,
                    }

        # --- R benchmarks ---
        coxphf_results: dict[int, dict] = {}
        if run_coxphf:
            if coxphf_reduce_after is not None:
                k_low = [k for k in k_values if k <= coxphf_reduce_after]
                k_high = [k for k in k_values if k > coxphf_reduce_after]

                if k_low:
                    print(
                        f"Running R (coxphf) benchmarks for k<={coxphf_reduce_after} ({n_runs} runs)...",
                        file=sys.stderr,
                        flush=True,
                    )
                    coxphf_results.update(run_r_coxphf(k_low, tmpdir, n_runs))
                if k_high:
                    coxphf_runs = n_runs if n_runs <= 3 else max(3, n_runs // 3)
                    print(
                        f"Running R (coxphf) benchmarks for k>{coxphf_reduce_after} ({coxphf_runs} runs)...",
                        file=sys.stderr,
                        flush=True,
                    )
                    coxphf_results.update(run_r_coxphf(k_high, tmpdir, coxphf_runs))
            else:
                print(
                    f"Running R (coxphf) benchmarks ({n_runs} runs)...",
                    file=sys.stderr,
                    flush=True,
                )
                coxphf_results = run_r_coxphf(k_values, tmpdir, n_runs)

        # --- Verification and result collection ---
        print(
            "Verifying results and computing statistics...", file=sys.stderr, flush=True
        )
        results = []

        for k in k_values:
            X, surv_time, event = make_benchmark_data(
                N_SAMPLES, k, EVENT_RATE, BASE_SEED
            )

            # Verification
            if run_firthmodels and run_coxphf and not skip_verification:
                py = py_results["numba"][k]
                r = coxphf_results[k]
                check_agreement(k, py["fit_coef"], r["fit_coef"], "fit")
                check_agreement(
                    k,
                    py["full_coef"],
                    r["full_coef"],
                    "full",
                    py["full_ci"],
                    r["full_ci"],
                    py["full_pval"],
                    r["full_pval"],
                )
                # Also check numba vs numpy
                py_numpy = py_results["numpy"][k]
                check_agreement(
                    k, py["fit_coef"], py_numpy["fit_coef"], "numba vs numpy"
                )

            # Collect timing results
            row: dict = {
                "k": k,
                "n": N_SAMPLES,
                "events": int(event.sum()),
                "event_rate": event.sum() / N_SAMPLES,
            }

            if run_firthmodels:
                py_numba = py_results["numba"][k]
                py_numpy = py_results["numpy"][k]
                row["numba_fit_ms"] = compute_min(py_numba["fit_times"]) * 1000
                row["numpy_fit_ms"] = compute_min(py_numpy["fit_times"]) * 1000
                row["numba_full_ms"] = compute_min(py_numba["full_times"]) * 1000
                row["numpy_full_ms"] = compute_min(py_numpy["full_times"]) * 1000

            if run_coxphf:
                r = coxphf_results[k]
                row["coxphf_fit_ms"] = compute_min(r["fit_times"]) * 1000
                row["coxphf_full_ms"] = compute_min(r["full_times"]) * 1000
                # Store for reproducibility
                row["coxphf_fit_coef"] = json.dumps(r["fit_coef"].tolist())
                row["coxphf_full_coef"] = json.dumps(r["full_coef"].tolist())
                row["coxphf_full_ci"] = json.dumps(r["full_ci"].tolist())
                row["coxphf_full_pval"] = json.dumps(r["full_pval"].tolist())

            results.append(row)

    return pd.DataFrame(results), version_info


def print_table(df: pd.DataFrame) -> None:
    """Print results as markdown table with minimum times."""
    print("\n## Fit Only - minimum time in ms\n")
    print("| k | numba | numpy | coxphf |")
    print("|--:|------:|------:|-------:|")
    for _, row in df.iterrows():
        print(
            f"| {int(row['k']):3d} | "
            f"{row['numba_fit_ms']:.1f} | "
            f"{row['numpy_fit_ms']:.1f} | "
            f"{row['coxphf_fit_ms']:.1f} |"
        )

    print("\n## Full Workflow (fit + LRT + profile CI) - minimum time in ms\n")
    print("| k | numba | numpy | coxphf |")
    print("|--:|------:|------:|-------:|")
    for _, row in df.iterrows():
        print(
            f"| {int(row['k']):3d} | "
            f"{row['numba_full_ms']:.1f} | "
            f"{row['numpy_full_ms']:.1f} | "
            f"{row['coxphf_full_ms']:.1f} |"
        )


def save_plot(
    df: pd.DataFrame,
    output_path: str = "benchmark_cox_scaling.png",
    version_info: dict[str, str] | None = None,
) -> None:
    """Save scaling plot to file as 2x2 grid."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot", file=sys.stderr)
        return

    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    if version_info:
        py_info = f"firthmodels {version_info.get('firthmodels_version', '?')}"
        py_blas = f"BLAS: {version_info.get('numpy_blas', '?')}"
        r_info = f"coxphf {version_info.get('coxphf_version', '?')}"
        r_blas = f"BLAS: {version_info.get('r_blas', '?')}"
        fig.suptitle(f"{py_info} | {py_blas}\n{r_info} | {r_blas}", fontsize=10, y=0.98)

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
    print(f"Plot saved to {output_path}", file=sys.stderr)


def generate_report(
    df: pd.DataFrame,
    report_path: str,
    plot_filename: str,
    n_runs: int,
    version_info: dict[str, str] | None = None,
    command: list[str] | None = None,
) -> None:
    """Generate a markdown report with benchmark results."""
    if version_info:
        firthmodels_ver = version_info.get("firthmodels_version", "unknown")
        coxphf_ver = version_info.get("coxphf_version", "unknown")
        numpy_blas = version_info.get("numpy_blas", "unknown")
        r_blas = version_info.get("r_blas", "unknown")
        os_info = version_info.get("os", "unknown")
        cpu_info = version_info.get("cpu", "unknown")
    else:
        firthmodels_ver = ""
        coxphf_ver = numpy_blas = r_blas = "unknown"
        os_info = cpu_info = "unknown"

    report = f"""# Firth Cox Proportional Hazards Benchmark

Comparison of [firthmodels](https://github.com/jzluo/firthmodels)
and [coxphf](https://cran.r-project.org/web/packages/coxphf/index.html)
for Firth-penalized Cox proportional hazards regression.

## System

| | |
|-----|-----|
| **OS** | {os_info} |
| **CPU** | {cpu_info} |

## Libraries Compared

| Library | Version | BLAS |
|---------|---------|------|
| **firthmodels (numba)** | {firthmodels_ver} | {numpy_blas} |
| **firthmodels (numpy)** | {firthmodels_ver} | {numpy_blas} |
| **coxphf** | {coxphf_ver} | {r_blas} |

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Observations (n) | {N_SAMPLES:,} |
| Event rate | {EVENT_RATE:.0%} |
| Features (k) | {", ".join(str(k) for k in df["k"].tolist())} |
| Runs per config | {n_runs} |
| Solver max_iter | {MAX_ITER} |
| Solver xtol | {XTOL} |
| Solver gtol | {GTOL} |

All implementations agree within chosen tolerance (coefficients {COEF_TOL}, CIs {CI_TOL}, p-values {PVAL_TOL}).

## Results

![Benchmark scaling plot]({plot_filename})

### Fit Only

Time to fit the model and perform Wald inference. Values are minimum time across runs in milliseconds.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
"""

    for _, row in df.iterrows():
        report += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_fit_ms']:.1f} | "
            f"{row['numpy_fit_ms']:.1f} | "
            f"{row['coxphf_fit_ms']:.1f} |\n"
        )

    report += f"""
### Full Workflow (Fit + LRT + Profile CI)

Time to fit the model, compute penalized likelihood ratio test p-values for all coefficients, and profile likelihood confidence intervals.

| k | firthmodels<br>(numba) | firthmodels<br>(numpy) | coxphf |
|--:|------:|------:|-------:|
"""

    for _, row in df.iterrows():
        report += (
            f"| {int(row['k']):3d} | "
            f"{row['numba_full_ms']:.1f} | "
            f"{row['numpy_full_ms']:.1f} | "
            f"{row['coxphf_full_ms']:.1f} |\n"
        )

    report += "\n---\n"

    if command:
        cmd_str = "python " + " ".join(shlex.quote(arg) for arg in command)
        report += f"""
## Command used to run benchmark

```bash
{cmd_str}
```
"""

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Report saved to {report_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark firthmodels FirthCoxPH vs R coxphf"
    )
    parser.add_argument(
        "-n",
        "--n-runs",
        type=int,
        default=N_RUNS,
        help=f"Number of benchmark runs (default: {N_RUNS})",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default=None,
        help="Comma-separated k values (default: 5,10,...,50)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip numerical agreement verification",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save plot to file (e.g., benchmark_cox_scaling.png)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Generate markdown report (e.g., benchmark_cox_report.md)",
    )
    parser.add_argument(
        "--coxphf-reduce-after",
        type=int,
        default=15,
        metavar="K",
        help="Reduce coxphf runs to max(3, n/3) for k > K. Use 0 to disable. (default: 25)",
    )
    args = parser.parse_args()

    k_values = (
        K_VALUES
        if args.k_values is None
        else [int(x) for x in args.k_values.split(",")]
    )

    coxphf_reduce_after = (
        args.coxphf_reduce_after if args.coxphf_reduce_after > 0 else None
    )

    # Run benchmarks
    df, version_info = run_benchmarks(
        k_values=k_values,
        n_runs=args.n_runs,
        skip_verification=args.skip_verification,
        coxphf_reduce_after=coxphf_reduce_after,
    )

    # Output
    print_table(df)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nResults saved to {args.csv}", file=sys.stderr)

    if args.plot:
        save_plot(df, args.plot, version_info)

    if args.report:
        report_path = Path(args.report)
        plot_path = report_path.with_suffix(".png")
        if not args.plot:
            save_plot(df, str(plot_path), version_info)
        else:
            plot_path = Path(args.plot)
        generate_report(
            df, args.report, plot_path.name, args.n_runs, version_info, sys.argv
        )


if __name__ == "__main__":
    main()
