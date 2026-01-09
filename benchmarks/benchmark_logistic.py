"""
Benchmark: firthmodels, R logistf, and R brglm2.

Measures scaling behavior as k increases.

Run with: python benchmark_logistic.py

Requires: R with logistf, brglm2, microbenchmark, jsonlite packages installed.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from firthmodels import FirthLogisticRegression

# -----------------------------------------------------------------------------
# Benchmark parameters
# -----------------------------------------------------------------------------
N_SAMPLES = 1000
EVENT_RATE = 0.20
K_VALUES = list(range(5, 55, 5))  # [5, 10, 15, ..., 50]
N_RUNS = 30
BASE_SEED = 42

# Solver parameters
MAX_ITER = 50
MAX_HALFSTEP = 0
TOL = 1e-6

# Tolerance for checking numerical agreement between the implementations
COEF_TOL = 1e-6  # Coefficients
CI_TOL = 1e-6  # Confidence intervals (profile CI)
PVAL_TOL = 1e-6  # P-values


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def make_benchmark_data(
    n: int, k: int, event_rate: float = 0.20, base_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for Firth logistic regression benchmarking.

    Designed for consistent convergence across libraries (firthmodels,
    logistf, brglm2) including for profile likelihood CIs.

    Uses small coefficients scaled by 1/sqrt(k) to keep Var(X @ beta) constant
    and avoid separation regardless of k.

    Parameters
    ----------
    n : int
        Number of observations
    k : int
        Number of features (excluding intercept)
    event_rate : float
        Target proportion of y=1
    base_seed : int
        Base random seed (combined with k for reproducibility)

    Returns
    -------
    X : ndarray of shape (n, k)
        Feature matrix (standardized)
    y : ndarray of shape (n,)
        Binary response (0/1)
    """
    # Deterministic seed per k value
    rng = np.random.default_rng(base_seed * 1000 + k)

    # Independent standard normal features
    X = rng.standard_normal((n, k))

    # Small coefficients: Var(X @ beta) approx 0.01 for all k
    beta = rng.normal(0, 0.1 / np.sqrt(k), size=k)

    # Intercept targeting event_rate
    intercept = np.log(event_rate / (1 - event_rate))

    # Generate response
    eta = intercept + X @ beta
    p = 1 / (1 + np.exp(-eta))
    y = rng.binomial(1, p)

    return X, y


# -----------------------------------------------------------------------------
# R package validation
# -----------------------------------------------------------------------------
def validate_r_packages() -> None:
    """Check that R and required packages are installed before starting."""
    # Check Rscript is available
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

    # Check required packages
    r_script = """
    for (pkg in c("logistf", "brglm2", "microbenchmark", "jsonlite")) {
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
            f"install.packages(c('logistf', 'brglm2', 'microbenchmark', 'jsonlite'))\n"
            f"R stderr: {result.stderr}"
        )


# -----------------------------------------------------------------------------
# Python (firthmodels) benchmarks
# -----------------------------------------------------------------------------
def warmup_numba() -> None:
    """Global warmup to ensure all numba code paths are JIT compiled.

    This must be called before any timing to avoid JIT overhead in benchmarks.
    """
    # Small dataset to minimize warmup time
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    y = rng.binomial(1, 0.5, 100)

    model = FirthLogisticRegression(backend="numba")
    model.fit(X, y)
    model.lrt()
    model.conf_int(method="pl")


def make_model(
    backend: Literal["auto", "numba", "numpy"] = "numba",
) -> FirthLogisticRegression:
    """Create FirthLogisticRegression model with benchmark parameters."""
    return FirthLogisticRegression(
        max_iter=MAX_ITER,
        max_halfstep=MAX_HALFSTEP,
        gtol=TOL,
        xtol=TOL,
        backend=backend,
    )


def time_python_fit(
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = N_RUNS,
    backend: Literal["auto", "numba", "numpy"] = "numba",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Time firthmodels fit only (no LRT, no profile CI)."""
    times = []
    for _ in range(n_runs):
        model = make_model(backend=backend)
        start = time.perf_counter()
        model.fit(X, y)
        times.append(time.perf_counter() - start)

    return np.array(times), model.coef_, model.intercept_


def time_python_full(
    X: np.ndarray,
    y: np.ndarray,
    n_runs: int = N_RUNS,
    backend: Literal["auto", "numba", "numpy"] = "numba",
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Time firthmodels fit + LRT + profile CI."""
    times = []
    for _ in range(n_runs):
        model = make_model(backend=backend)
        start = time.perf_counter()
        model.fit(X, y)
        model.lrt()
        ci = model.conf_int(method="pl", max_iter=MAX_ITER, tol=TOL)
        times.append(time.perf_counter() - start)

    return np.array(times), model.coef_, model.intercept_, ci, model.lrt_pvalues_


# -----------------------------------------------------------------------------
# R benchmarks
# -----------------------------------------------------------------------------
def prepare_data_files(k_values: list[int], tmpdir: Path) -> None:
    """Generate and save CSV files for R benchmarks."""
    for k in k_values:
        X, y = make_benchmark_data(N_SAMPLES, k, EVENT_RATE, BASE_SEED)
        csv_path = tmpdir / f"data_k{k}.csv"
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(k)])
        df["y"] = y
        df.to_csv(csv_path, index=False)


def run_r_logistf(
    k_values: list[int], data_dir: Path, n_runs: int = N_RUNS
) -> dict[int, dict]:
    """Run logistf benchmarks for all k values.

    Returns dict mapping k -> results dict with numpy arrays.
    """
    script_path = Path(__file__).parent / "logistf_bench.R"
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
            str(TOL),
        ],
        stdout=subprocess.PIPE,
        stderr=None,  # Let stderr go to terminal for progress
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("R logistf batch script failed")

    raw = json.loads(result.stdout)

    # Convert to expected format with numpy arrays and int keys
    results = {}
    for k_str, data in raw.items():
        k = int(k_str)
        results[k] = {
            "fit_times": np.array(data["fit_times"]),
            "fit_coef": np.array(data["fit_coef"]),
            "fit_intercept": data["fit_intercept"],
            "full_times": np.array(data["full_times"]),
            "full_coef": np.array(data["full_coef"]),
            "full_intercept": data["full_intercept"],
            "full_ci": np.column_stack([data["ci_lower"], data["ci_upper"]]),
            "full_pval": np.array(data["full_pval"]),
        }

    return results


def run_r_brglm2(
    k_values: list[int], data_dir: Path, n_runs: int = N_RUNS
) -> dict[str, dict[int, dict]]:
    """Run brglm2 benchmarks for all k values.

    Runs both AS_mean and MPL_Jeffreys methods.
    Returns dict mapping method -> k -> results dict with numpy arrays.
    """
    script_path = Path(__file__).parent / "brglm2_bench.R"
    k_str = ",".join(str(k) for k in k_values)

    result = subprocess.run(
        [
            "Rscript",
            str(script_path),
            str(data_dir),
            k_str,
            str(n_runs),
            str(MAX_ITER),
            str(TOL),
        ],
        stdout=subprocess.PIPE,
        stderr=None,  # Let stderr go to terminal for progress
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("R brglm2 batch script failed")

    raw = json.loads(result.stdout)

    # Convert to expected format with numpy arrays and int keys
    results: dict[str, dict[int, dict]] = {"AS_mean": {}, "MPL_Jeffreys": {}}
    for method, method_data in raw.items():
        for k_str, data in method_data.items():
            k = int(k_str)
            results[method][k] = {
                "fit_times": np.array(data["fit_times"]),
                "fit_coef": np.array(data["fit_coef"]),
                "fit_intercept": data["fit_intercept"],
            }

    return results


# -----------------------------------------------------------------------------
# Numerical agreement verification
# -----------------------------------------------------------------------------
def check_agreement(
    k: int,
    py_coef: np.ndarray,
    py_intercept: float,
    r_coef: np.ndarray,
    r_intercept: float,
    label: str,
    py_ci: np.ndarray | None = None,
    r_ci: np.ndarray | None = None,
    py_pval: np.ndarray | None = None,
    r_pval: np.ndarray | None = None,
) -> None:
    """Check that Python and R results agree within tolerance."""
    # Check coefficients
    coef_diff = np.max(np.abs(py_coef - r_coef))
    intercept_diff = np.abs(py_intercept - r_intercept)
    if coef_diff > COEF_TOL or intercept_diff > COEF_TOL:
        raise AssertionError(
            f"k={k} ({label}): coef diff {coef_diff:.2e}, "
            f"intercept diff {intercept_diff:.2e} > {COEF_TOL}"
        )

    # Check CIs if provided
    if py_ci is not None and r_ci is not None:
        # Python CI: shape (k+1, 2), intercept is LAST row (index -1)
        # R CI (logistf): shape (k+1, 2), intercept is FIRST row (index 0)
        # Reorder Python to match R by moving intercept from last to first
        py_ci_reordered = np.vstack([py_ci[-1:], py_ci[:-1]])
        ci_diff = np.max(np.abs(py_ci_reordered - r_ci))
        if ci_diff > CI_TOL:
            raise AssertionError(f"k={k} ({label}): CI diff {ci_diff:.2e} > {CI_TOL}")

    # Check p-values if provided
    if py_pval is not None and r_pval is not None:
        # Same intercept position difference as CIs: Python last, R first
        py_pval_reordered = np.concatenate([py_pval[-1:], py_pval[:-1]])
        pval_diff = np.max(np.abs(py_pval_reordered - r_pval))
        if pval_diff > PVAL_TOL:
            raise AssertionError(
                f"k={k} ({label}): p-value diff {pval_diff:.2e} > {PVAL_TOL}"
            )


# -----------------------------------------------------------------------------
# Version and BLAS info
# -----------------------------------------------------------------------------
def get_python_version_info() -> dict[str, str]:
    """Get version info for Python libraries, BLAS backend, and system info."""
    from importlib.metadata import version

    info = get_system_info()
    info["firthmodels_version"] = version("firthmodels")

    # NumPy BLAS info from config dict
    try:
        blas_info = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
        if blas_info.get("found"):
            lib_dir = blas_info.get("lib directory", "")
            name = blas_info.get("name", "unknown")
            blas_version = blas_info.get("version", "")
            info["numpy_blas"] = f"{lib_dir} ({name} {blas_version})".strip()
        else:
            info["numpy_blas"] = "unknown"
    except Exception:
        info["numpy_blas"] = "unknown"

    return info


def get_r_version_info() -> dict[str, str]:
    """Get R package versions and BLAS backend."""
    r_script = """
    cat("LOGISTF_VERSION:", as.character(packageVersion("logistf")), "\\n")
    cat("BRGLM2_VERSION:", as.character(packageVersion("brglm2")), "\\n")
    si <- sessionInfo()
    cat("R_BLAS:", si$BLAS, "\\n")
    """

    result = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)

    info = {}
    if result.returncode == 0:
        blas_path = ""
        for line in result.stdout.strip().split("\n"):
            if line.startswith("LOGISTF_VERSION:"):
                info["logistf_version"] = line.split(":")[1].strip()
            elif line.startswith("BRGLM2_VERSION:"):
                info["brglm2_version"] = line.split(":")[1].strip()
            elif line.startswith("R_BLAS:"):
                blas_path = line.split(":", 1)[1].strip()

        # Show full BLAS path so user can verify R and numpy use the same library
        info["r_blas"] = blas_path if blas_path else "unknown"
    else:
        info["logistf_version"] = "unknown"
        info["brglm2_version"] = "unknown"
        info["r_blas"] = "unknown"

    return info


def get_system_info() -> dict[str, str]:
    """Get OS and CPU info in a cross-platform way."""
    import platform as plat

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

    # CPU info - platform.processor() often returns unhelpful values
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
                import winreg

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


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------
def compute_min(times: np.ndarray) -> float:
    """Return minimum time (best achievable performance)."""
    return float(np.min(times))


# -----------------------------------------------------------------------------
# Saved results loading
# -----------------------------------------------------------------------------
def load_saved_results(
    saved_csv: str,
    k_values: list[int],
    load_firthmodels: bool = False,
    load_brglm2: bool = True,
    load_logistf: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict] | None, dict[str, dict[int, dict]] | None]:
    """Load results from a saved CSV file.

    Parameters
    ----------
    saved_csv : str
        Path to CSV with previous results.
    k_values : list of int
        Which k values to load.
    load_firthmodels : bool
        Whether to load firthmodels timing columns.
    load_brglm2 : bool
        Whether to load brglm2 results.
    load_logistf : bool
        Whether to load logistf results.

    Returns
    -------
    saved_df : DataFrame
        The loaded DataFrame (for timing columns).
    logistf_results : dict or None
        logistf results if load_logistf=True, else None.
    brglm2_results : dict or None
        brglm2 results if load_brglm2=True, else None.
    """
    saved_df = pd.read_csv(saved_csv)

    # Determine required columns based on what we're loading
    required_cols = {"k", "n"}
    if load_firthmodels:
        required_cols.update(
            {
                "numba_fit_ms",
                "numpy_fit_ms",
                "numba_full_ms",
                "numpy_full_ms",
            }
        )
    if load_logistf:
        required_cols.update(
            {
                "logistf_fit_ms",
                "logistf_full_ms",
                "logistf_fit_coef",
                "logistf_fit_intercept",
                "logistf_full_coef",
                "logistf_full_intercept",
                "logistf_full_ci",
                "logistf_full_pval",
            }
        )
    if load_brglm2:
        required_cols.update(
            {
                "brglm2_as_fit_ms",
                "brglm2_mpl_fit_ms",
                "brglm2_as_fit_coef",
                "brglm2_as_fit_intercept",
                "brglm2_mpl_fit_coef",
                "brglm2_mpl_fit_intercept",
            }
        )

    missing_cols = required_cols - set(saved_df.columns)
    if missing_cols:
        raise ValueError(f"Saved CSV missing columns: {missing_cols}")

    saved_n = int(saved_df["n"].iloc[0])
    if saved_n != N_SAMPLES:
        raise ValueError(
            f"Saved CSV was created with n={saved_n}, but current N_SAMPLES={N_SAMPLES}."
        )

    saved_k = set(saved_df["k"].tolist())
    missing = set(k_values) - saved_k
    if missing:
        raise ValueError(f"k values {missing} not in saved CSV")

    # Reconstruct results from saved data
    logistf_dict: dict[int, dict] = {}
    brglm2_dict: dict[str, dict[int, dict]] = {"AS_mean": {}, "MPL_Jeffreys": {}}

    for _, row in saved_df.iterrows():
        k = int(row["k"])
        if k not in k_values:
            continue

        if load_logistf:
            logistf_dict[k] = {
                "fit_coef": np.array(json.loads(row["logistf_fit_coef"])),
                "fit_intercept": row["logistf_fit_intercept"],
                "full_coef": np.array(json.loads(row["logistf_full_coef"])),
                "full_intercept": row["logistf_full_intercept"],
                "full_ci": np.array(json.loads(row["logistf_full_ci"])),
                "full_pval": np.array(json.loads(row["logistf_full_pval"])),
            }
        if load_brglm2:
            brglm2_dict["AS_mean"][k] = {
                "fit_coef": np.array(json.loads(row["brglm2_as_fit_coef"])),
                "fit_intercept": row["brglm2_as_fit_intercept"],
            }
            brglm2_dict["MPL_Jeffreys"][k] = {
                "fit_coef": np.array(json.loads(row["brglm2_mpl_fit_coef"])),
                "fit_intercept": row["brglm2_mpl_fit_intercept"],
            }

    logistf_results = logistf_dict if load_logistf else None
    brglm2_results = brglm2_dict if load_brglm2 else None
    return saved_df, logistf_results, brglm2_results


# -----------------------------------------------------------------------------
# Main benchmark runner
# -----------------------------------------------------------------------------
def run_benchmarks(
    k_values: list[int] = K_VALUES,
    n_runs: int = N_RUNS,
    skip_verification: bool = False,
    logistf_reduce_after: int | None = 25,
    saved: str | None = None,
    run_firthmodels: bool = True,
    run_brglm2: bool = True,
    run_logistf: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run all benchmarks and return results as DataFrame.

    Parameters
    ----------
    k_values : list of int
        Which k values to benchmark.
    n_runs : int
        Number of benchmark runs.
    skip_verification : bool
        Skip numerical agreement verification.
    logistf_reduce_after : int or None
        Reduce logistf runs to max(3, n/3) for k > this value. None to disable.
    saved : str or None
        Path to CSV with saved results. Non-selected libraries load from here.
    run_firthmodels : bool
        Whether to run firthmodels (Python) benchmarks.
    run_brglm2 : bool
        Whether to run brglm2 benchmarks.
    run_logistf : bool
        Whether to run logistf benchmarks.

    Returns
    -------
    df : DataFrame
        Benchmark results
    version_info : dict
        Version and BLAS info for all libraries
    """
    version_info = get_python_version_info()

    # Load saved results for libraries we're not running
    saved_df = None
    logistf_results: dict[int, dict] | None = None
    brglm2_results: dict[str, dict[int, dict]] | None = None

    if saved:
        load_firthmodels = not run_firthmodels
        load_brglm2 = not run_brglm2
        load_logistf = not run_logistf
        if load_firthmodels or load_brglm2 or load_logistf:
            print(f"Loading saved results from {saved}...", file=sys.stderr, flush=True)
            saved_df, logistf_results, brglm2_results = load_saved_results(
                saved,
                k_values,
                load_firthmodels=load_firthmodels,
                load_brglm2=load_brglm2,
                load_logistf=load_logistf,
            )

    # Validate R packages and collect version info if running any R benchmarks
    if run_brglm2 or run_logistf:
        print("Validating R packages...", file=sys.stderr, flush=True)
        validate_r_packages()
        print("Collecting version info...", file=sys.stderr, flush=True)
        version_info.update(get_r_version_info())

    # Global numba warmup before any timing
    if run_firthmodels:
        print("Warming up numba JIT...", file=sys.stderr, flush=True)
        warmup_numba()

    # Create temp directory for data files (needed for R benchmarks)
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

                    X, y = make_benchmark_data(N_SAMPLES, k, EVENT_RATE, BASE_SEED)

                    py_fit_times, py_fit_coef, py_fit_intercept = time_python_fit(
                        X,
                        y,
                        n_runs,
                        backend,  # type: ignore[arg-type]
                    )
                    py_full_times, py_full_coef, py_full_intercept, py_ci, py_pval = (
                        time_python_full(X, y, n_runs, backend)  # type: ignore[arg-type]
                    )

                    py_results[backend][k] = {
                        "fit_times": py_fit_times,
                        "fit_coef": py_fit_coef,
                        "fit_intercept": py_fit_intercept,
                        "full_times": py_full_times,
                        "full_coef": py_full_coef,
                        "full_intercept": py_full_intercept,
                        "full_ci": py_ci,
                        "full_pval": py_pval,
                    }

        # --- R benchmarks (brglm2 first, then logistf) ---
        if run_brglm2:
            print("Running R (brglm2) benchmarks...", file=sys.stderr, flush=True)
            brglm2_results = run_r_brglm2(k_values, tmpdir, n_runs)

        if run_logistf:
            if logistf_reduce_after is not None:
                k_low = [k for k in k_values if k <= logistf_reduce_after]
                k_high = [k for k in k_values if k > logistf_reduce_after]

                logistf_results = {}
                if k_low:
                    print(
                        f"Running R (logistf) benchmarks for k<={logistf_reduce_after} ({n_runs} runs)...",
                        file=sys.stderr,
                        flush=True,
                    )
                    logistf_results.update(run_r_logistf(k_low, tmpdir, n_runs))
                if k_high:
                    logistf_runs = n_runs if n_runs <= 3 else max(3, n_runs // 3)
                    print(
                        f"Running R (logistf) benchmarks for k>{logistf_reduce_after} ({logistf_runs} runs)...",
                        file=sys.stderr,
                        flush=True,
                    )
                    logistf_results.update(run_r_logistf(k_high, tmpdir, logistf_runs))
            else:
                print(
                    f"Running R (logistf) benchmarks ({n_runs} runs)...",
                    file=sys.stderr,
                    flush=True,
                )
                logistf_results = run_r_logistf(k_values, tmpdir, n_runs)

        # --- Verification and result collection ---
        print(
            "Verifying results and computing statistics...", file=sys.stderr, flush=True
        )
        results = []

        # At this point, all results dicts are populated (either run or loaded from saved)
        assert logistf_results is not None
        assert brglm2_results is not None

        for k in k_values:
            X, y = make_benchmark_data(N_SAMPLES, k, EVENT_RATE, BASE_SEED)

            logistf = logistf_results[k]
            brglm2_as = brglm2_results["AS_mean"][k]
            brglm2_mpl = brglm2_results["MPL_Jeffreys"][k]

            # Verification requires running firthmodels
            if run_firthmodels and not skip_verification:
                py = py_results["numba"][k]
                check_agreement(
                    k,
                    py["fit_coef"],
                    py["fit_intercept"],
                    logistf["fit_coef"],
                    logistf["fit_intercept"],
                    "fit vs logistf",
                )
                check_agreement(
                    k,
                    py["fit_coef"],
                    py["fit_intercept"],
                    brglm2_as["fit_coef"],
                    brglm2_as["fit_intercept"],
                    "fit vs brglm2 AS_mean",
                )
                check_agreement(
                    k,
                    py["fit_coef"],
                    py["fit_intercept"],
                    brglm2_mpl["fit_coef"],
                    brglm2_mpl["fit_intercept"],
                    "fit vs brglm2 MPL_Jeffreys",
                )
                check_agreement(
                    k,
                    py["full_coef"],
                    py["full_intercept"],
                    logistf["full_coef"],
                    logistf["full_intercept"],
                    "full vs logistf",
                    py["full_ci"],
                    logistf["full_ci"],
                    py["full_pval"],
                    logistf["full_pval"],
                )
                py_numpy = py_results["numpy"][k]
                check_agreement(
                    k,
                    py["fit_coef"],
                    py["fit_intercept"],
                    py_numpy["fit_coef"],
                    py_numpy["fit_intercept"],
                    "numba vs numpy",
                )

            # Get saved row if needed for any library
            saved_row = None
            if saved_df is not None:
                saved_row = saved_df[saved_df["k"] == k].iloc[0]

            # Python timings
            if run_firthmodels:
                py_numba = py_results["numba"][k]
                py_numpy = py_results["numpy"][k]
                numba_fit_ms = compute_min(py_numba["fit_times"]) * 1000
                numpy_fit_ms = compute_min(py_numpy["fit_times"]) * 1000
                numba_full_ms = compute_min(py_numba["full_times"]) * 1000
                numpy_full_ms = compute_min(py_numpy["full_times"]) * 1000
            else:
                assert saved_row is not None
                numba_fit_ms = saved_row["numba_fit_ms"]
                numpy_fit_ms = saved_row["numpy_fit_ms"]
                numba_full_ms = saved_row["numba_full_ms"]
                numpy_full_ms = saved_row["numpy_full_ms"]

            # brglm2 timings
            if run_brglm2:
                brglm2_as_fit_ms = compute_min(brglm2_as["fit_times"]) * 1000
                brglm2_mpl_fit_ms = compute_min(brglm2_mpl["fit_times"]) * 1000
            else:
                assert saved_row is not None
                brglm2_as_fit_ms = saved_row["brglm2_as_fit_ms"]
                brglm2_mpl_fit_ms = saved_row["brglm2_mpl_fit_ms"]

            # logistf timings
            if run_logistf:
                logistf_fit_ms = compute_min(logistf["fit_times"]) * 1000
                logistf_full_ms = compute_min(logistf["full_times"]) * 1000
            else:
                assert saved_row is not None
                logistf_fit_ms = saved_row["logistf_fit_ms"]
                logistf_full_ms = saved_row["logistf_full_ms"]

            results.append(
                {
                    "k": k,
                    "n": N_SAMPLES,
                    "events": int(y.sum()),
                    "epv": y.sum() / (k + 1),
                    "numba_fit_ms": numba_fit_ms,
                    "numpy_fit_ms": numpy_fit_ms,
                    "brglm2_as_fit_ms": brglm2_as_fit_ms,
                    "brglm2_mpl_fit_ms": brglm2_mpl_fit_ms,
                    "logistf_fit_ms": logistf_fit_ms,
                    "numba_full_ms": numba_full_ms,
                    "numpy_full_ms": numpy_full_ms,
                    "logistf_full_ms": logistf_full_ms,
                    "brglm2_as_fit_coef": json.dumps(brglm2_as["fit_coef"].tolist()),
                    "brglm2_as_fit_intercept": brglm2_as["fit_intercept"],
                    "brglm2_mpl_fit_coef": json.dumps(brglm2_mpl["fit_coef"].tolist()),
                    "brglm2_mpl_fit_intercept": brglm2_mpl["fit_intercept"],
                    "logistf_fit_coef": json.dumps(logistf["fit_coef"].tolist()),
                    "logistf_fit_intercept": logistf["fit_intercept"],
                    "logistf_full_coef": json.dumps(logistf["full_coef"].tolist()),
                    "logistf_full_intercept": logistf["full_intercept"],
                    "logistf_full_ci": json.dumps(logistf["full_ci"].tolist()),
                    "logistf_full_pval": json.dumps(logistf["full_pval"].tolist()),
                }
            )

    return pd.DataFrame(results), version_info


def print_table(df: pd.DataFrame) -> None:
    """Print results as markdown table with minimum times."""
    print("\n## Fit Only - minimum time in ms\n")
    print("| k | numba | numpy | brglm2 (AS) | brglm2 (MPL) | logistf |")
    print("|--:|------:|------:|------------:|-------------:|--------:|")
    for _, row in df.iterrows():
        print(
            f"| {int(row['k']):3d} | "
            f"{row['numba_fit_ms']:.1f} | "
            f"{row['numpy_fit_ms']:.1f} | "
            f"{row['brglm2_as_fit_ms']:.1f} | "
            f"{row['brglm2_mpl_fit_ms']:.1f} | "
            f"{row['logistf_fit_ms']:.1f} |"
        )

    print("\n## Full Workflow (fit + LRT + profile CI) - minimum time in ms\n")
    print("| k | numba | numpy | logistf |")
    print("|--:|------:|------:|--------:|")
    for _, row in df.iterrows():
        print(
            f"| {int(row['k']):3d} | "
            f"{row['numba_full_ms']:.1f} | "
            f"{row['numpy_full_ms']:.1f} | "
            f"{row['logistf_full_ms']:.1f} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark firthmodels vs logistf and brglm2"
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
        "-o",
        "--out",
        type=str,
        default=None,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--saved",
        type=str,
        default=None,
        metavar="CSV",
        help="Load non-selected libraries from this CSV.",
    )
    parser.add_argument(
        "--firthmodels",
        action="store_true",
        help="Run firthmodels.",
    )
    parser.add_argument(
        "--brglm2",
        action="store_true",
        help="Run brglm2.",
    )
    parser.add_argument(
        "--logistf",
        action="store_true",
        help="Run logistf.",
    )
    parser.add_argument(
        "--logistf-reduce-after",
        type=int,
        default=25,
        metavar="K",
        help="Reduce logistf runs to max(3, n/3) for k > K. Use 0 to disable. (default: 25)",
    )
    args = parser.parse_args()

    k_values = (
        K_VALUES
        if args.k_values is None
        else [int(x) for x in args.k_values.split(",")]
    )

    # Convert 0 to None (disable run reduction)
    logistf_reduce_after = (
        args.logistf_reduce_after if args.logistf_reduce_after > 0 else None
    )

    # Determine which libraries to run
    any_selected = args.firthmodels or args.brglm2 or args.logistf

    if any_selected:
        run_firthmodels = args.firthmodels
        run_brglm2 = args.brglm2
        run_logistf = args.logistf
    elif args.saved:
        # --saved alone = run firthmodels only (backward compat with old --r-baseline)
        run_firthmodels, run_brglm2, run_logistf = True, False, False
    else:
        # Default: run everything
        run_firthmodels, run_brglm2, run_logistf = True, True, True

    # If no --saved and some libs not selected, must run them (with message)
    if not args.saved:
        missing = []
        if not run_firthmodels:
            missing.append("firthmodels")
        if not run_brglm2:
            missing.append("brglm2")
        if not run_logistf:
            missing.append("logistf")
        if missing:
            print(
                f"No --saved provided; also running {', '.join(missing)}",
                file=sys.stderr,
            )
            run_firthmodels = run_brglm2 = run_logistf = True

    # Run benchmarks
    df, version_info = run_benchmarks(
        k_values=k_values,
        n_runs=args.n_runs,
        skip_verification=args.skip_verification,
        logistf_reduce_after=logistf_reduce_after,
        saved=args.saved,
        run_firthmodels=run_firthmodels,
        run_brglm2=run_brglm2,
        run_logistf=run_logistf,
    )

    # Output
    print_table(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nResults saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
