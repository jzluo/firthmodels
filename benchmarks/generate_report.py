#!/usr/bin/env python3
"""Generate benchmarks README from CSV results.

Reads timing CSVs produced by benchmark_logistic.py / benchmark_cox.py,
together with the .meta.json sidecars those scripts write (run counts,
verification status, observed deviations, versions).

Solver and data-generation parameters quoted in the report are imported from
the benchmark scripts.

Usage:
    uv run python benchmarks/generate_report.py
    uv run python benchmarks/generate_report.py -o path/to/output.md
"""

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCHMARKS_DIR))

import benchmark_cox as bc  # noqa: E402
import benchmark_logistic as bl  # noqa: E402

# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------
# Categorical palette, colorblind-safe in this order on a white surface
# (adjacent-pair CVD delta-E >= 8, validated). Color follows the entity:
# firthmodels backends keep blue/orange in every figure; the primary R
# reference (logistf / coxphf) is always aqua.
COLOR_NUMBA = "#2a78d6"  # blue
COLOR_NUMPY = "#eb6834"  # orange
COLOR_R_REF = "#1baf7a"  # aqua: logistf / coxphf
COLOR_BRGLM2_AS = "#eda100"  # yellow
COLOR_BRGLM2_MPL = "#e87ba4"  # magenta

TEXT_COLOR = "#333333"

LineStyle = str | tuple[int, tuple[int, int]]
# (column, label, color, marker, linestyle)
SeriesSpec = tuple[str, str, str, str, LineStyle]

LOGISTIC_FIT_SERIES: list[SeriesSpec] = [
    ("numba_fit_ms", "firthmodels (numba)", COLOR_NUMBA, "o", "-"),
    ("numpy_fit_ms", "firthmodels (numpy)", COLOR_NUMPY, "s", "-"),
    ("logistf_fit_ms", "logistf", COLOR_R_REF, "^", "-"),
    ("brglm2_as_fit_ms", "brglm2 (AS-mean)", COLOR_BRGLM2_AS, "D", "-"),
    ("brglm2_mpl_fit_ms", "brglm2 (MPL-Jeffreys)", COLOR_BRGLM2_MPL, "v", (0, (4, 2))),
]
LOGISTIC_FULL_SERIES: list[SeriesSpec] = [
    ("numba_full_ms", "firthmodels (numba)", COLOR_NUMBA, "o", "-"),
    ("numpy_full_ms", "firthmodels (numpy)", COLOR_NUMPY, "s", "-"),
    ("logistf_full_ms", "logistf", COLOR_R_REF, "^", "-"),
]
COX_FIT_SERIES: list[SeriesSpec] = [
    ("numba_fit_ms", "firthmodels (numba)", COLOR_NUMBA, "o", "-"),
    ("numpy_fit_ms", "firthmodels (numpy)", COLOR_NUMPY, "s", "-"),
    ("coxphf_fit_ms", "coxphf", COLOR_R_REF, "^", "-"),
]
COX_FULL_SERIES: list[SeriesSpec] = [
    ("numba_full_ms", "firthmodels (numba)", COLOR_NUMBA, "o", "-"),
    ("numpy_full_ms", "firthmodels (numpy)", COLOR_NUMPY, "s", "-"),
    ("coxphf_full_ms", "coxphf", COLOR_R_REF, "^", "-"),
]


def save_scaling_plot(
    df: pd.DataFrame,
    panels: Sequence[tuple[str, Sequence[SeriesSpec]]],
    output_path: Path,
    suptitle: str,
) -> None:
    """Save a 1x2 scaling figure with a log time axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), layout="constrained")
    fig.suptitle(suptitle, fontsize=9, color=TEXT_COLOR)

    for ax, (title, series) in zip(axes, panels):
        # Draw slowest series first so legend order matches the visual
        # top-to-bottom order at the right edge of the panel.
        ordered = sorted(series, key=lambda s: df[s[0]].iloc[-1], reverse=True)
        for col, label, color, marker, linestyle in ordered:
            ax.plot(
                df["k"],
                df[col],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        ax.set_yscale("log")
        ax.set_xticks(df["k"])
        ax.set_xlabel("Number of features (k)", color=TEXT_COLOR)
        ax.set_ylabel("Time (ms, log scale)", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        ax.grid(True, which="major", alpha=0.3, linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_COLOR)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}", file=sys.stderr)


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def fmt_ms(value: float) -> str:
    """Format a millisecond value with resolution appropriate to its size."""
    if value < 1:
        return f"{value:.2f}"
    if value < 1000:
        return f"{value:.1f}"
    return f"{value:,.0f}"


def fmt_x(ratio: float) -> str:
    """Format a speedup ratio."""
    if ratio < 100:
        return f"{ratio:.1f}x"
    return f"{ratio:.0f}x"


def fmt_dev(value: float) -> str:
    """Format an observed numerical deviation."""
    if value == 0:
        return "0"
    return f"{value:.1e}"


def fmt_ranked_times(values: Sequence[float], r_indices: Sequence[int]) -> list[str]:
    """Format times, bolding the fastest overall and underlining the fastest
    R package -- the baseline of the table's speedup column."""
    formatted = [fmt_ms(v) for v in values]
    baseline = min(r_indices, key=lambda i: values[i])
    formatted[baseline] = f"<ins>{formatted[baseline]}</ins>"
    fastest = int(np.argmin(values))
    formatted[fastest] = f"**{formatted[fastest]}**"
    return formatted


def markdown_table(header: list[str], rows: list[list[str]], align: str) -> str:
    """Build a markdown table. align is one char per column: 'l' or 'r'."""
    sep = ["---" if a == "l" else "--:" for a in align]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(sep) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Run metadata
# -----------------------------------------------------------------------------
QUANTITY_LABELS = {
    "coef": "Coefficients",
    "ci": "Profile CI bounds",
    "pval": "LRT p-values",
}

# (comparison, quantity, max abs deviation)
DeviationRow = tuple[str, str, float]


def load_metadata(csv_path: Path) -> dict | None:
    """Load the .meta.json sidecar written by the benchmark scripts."""
    meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not read {meta_path}: {exc}", file=sys.stderr)
        return None


def metadata_deviation_rows(meta: dict | None, model_label: str) -> list[DeviationRow]:
    """Flatten benchmark-time deviations from run metadata into table rows."""
    if not meta:
        return []
    verification = meta.get("verification") or {}
    rows = []
    for label, quantities in (verification.get("max_abs_deviations") or {}).items():
        for quantity, value in quantities.items():
            rows.append(
                (
                    f"{model_label} {label}",
                    QUANTITY_LABELS.get(quantity, quantity),
                    float(value),
                )
            )
    return rows


def metadata_firthmodels_version(meta: dict | None) -> str | None:
    if not meta:
        return None
    return (meta.get("versions") or {}).get("firthmodels_version")


# -----------------------------------------------------------------------------
# Version and environment info
# -----------------------------------------------------------------------------
def get_environment_info() -> dict[str, str]:
    """Collect OS, CPU, Python and R stack versions, BLAS, and R compile flags."""
    from importlib.metadata import version

    info = bl.get_system_info()
    info["python"] = ".".join(str(v) for v in sys.version_info[:3])
    for pkg in ("firthmodels", "numpy", "scipy"):
        info[pkg] = version(pkg)
    try:
        info["numba"] = version("numba")
    except Exception:
        info["numba"] = "not installed"

    try:
        blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
        info["numpy_blas_build"] = (
            f"{blas.get('name', 'unknown')} {blas.get('version', '')}".strip()
        )
    except Exception:
        info["numpy_blas_build"] = "unknown"

    r_script = """
    cat("r:", paste(R.version$major, R.version$minor, sep="."), "\\n")
    for (p in c("logistf", "brglm2", "coxphf")) {
        v <- tryCatch(as.character(packageVersion(p)), error = function(e) "unknown")
        cat(p, ": ", v, "\\n", sep = "")
    }
    cat("r_blas:", sessionInfo()$BLAS, "\\n")
    for (f in c("CFLAGS", "FCFLAGS")) {
        val <- system2(file.path(R.home("bin"), "R"), c("CMD", "config", f),
                       stdout = TRUE)
        cat("r_", tolower(f), ": ", trimws(paste(val, collapse = " ")), "\\n",
            sep = "")
    }
    """
    for key in ("r", "logistf", "brglm2", "coxphf", "r_blas", "r_cflags", "r_fcflags"):
        info.setdefault(key, "unknown")
    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script], capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                key, _, value = line.partition(":")
                if key in info:
                    info[key] = value.strip()
    except FileNotFoundError:
        pass

    return info


def get_runtime_blas() -> tuple[str, str] | None:
    """Return (filepath, version) of the BLAS numpy actually loaded, if known."""
    try:
        import threadpoolctl
    except ImportError:
        return None
    for entry in threadpoolctl.threadpool_info():
        if entry.get("user_api") == "blas":
            return str(entry.get("filepath", "")), str(entry.get("version", "?"))
    return None


def key_compile_flags(cflags: str) -> str:
    """Pull the optimization-relevant tokens out of a full compiler flag string."""
    keep = [
        t for t in cflags.split() if t.startswith(("-O", "-march", "-mtune", "-flto"))
    ]
    return " ".join(keep) if keep else "unknown"


# -----------------------------------------------------------------------------
# Report sections
# -----------------------------------------------------------------------------
def agreement_sentence(worst: float | None, tol: float) -> str:
    """One-sentence summary of numerical agreement, derived from the
    deviations recorded in the run metadata."""
    if worst is None:
        return (
            "Numerical agreement between the implementations was not verified "
            "for these exact results (see [Correctness](#correctness))."
        )
    if worst <= tol:
        return (
            "The largest observed "
            f"deviation across coefficients, profile CI bounds, and p-values "
            f"is {fmt_dev(worst)} (see [Correctness](#correctness))."
        )
    return (
        f"Implementations agree only to within {fmt_dev(worst)}, which exceeds "
        f"the intended tolerance of {tol:g} -- see "
        "[Correctness](#correctness) before relying on these comparisons."
    )


def runs_sentence(logistic_meta: dict | None, cox_meta: dict | None) -> str:
    """Describe repetition counts from run metadata, or from script defaults."""

    def describe(meta: dict | None, script_default: int) -> str:
        if meta and "n_runs" in meta:
            n = str(meta["n_runs"])
            reduced = meta.get("r_reduced_runs")
            if reduced:
                libs = ", ".join(reduced["libraries"])
                n += (
                    f" ({libs} reduced to {reduced['n_runs']} "
                    f"for k > {reduced['for_k_above']})"
                )
            return n
        return f"{script_default} (script default; not recorded for these CSVs)"

    logistic = describe(logistic_meta, bl.N_RUNS)
    cox = describe(cox_meta, bc.N_RUNS)
    if logistic == cox:
        return f"The reported value is the fastest of {logistic} runs."
    return (
        "The reported value is the fastest observed run "
        f"({logistic} runs for logistic, {cox} for Cox)."
    )


def generate_summary(
    logistic_df: pd.DataFrame,
    cox_df: pd.DataFrame,
    agreement: str,
    logistic_meta: dict | None,
    cox_meta: dict | None,
) -> str:
    logistic_best_r_fit = logistic_df[
        ["logistf_fit_ms", "brglm2_as_fit_ms", "brglm2_mpl_fit_ms"]
    ].min(axis=1)

    rows = []
    for df, workload, baseline_label, baseline, col in [
        (
            logistic_df,
            "Logistic: fit + Wald",
            "next-fastest package",
            logistic_best_r_fit,
            "numba_fit_ms",
        ),
        (
            logistic_df,
            "Logistic: fit + LRT + profile CI",
            "logistf",
            logistic_df["logistf_full_ms"],
            "numba_full_ms",
        ),
        (
            cox_df,
            "Cox: fit + Wald",
            "coxphf",
            cox_df["coxphf_fit_ms"],
            "numba_fit_ms",
        ),
        (
            cox_df,
            "Cox: fit + LRT + profile CI",
            "coxphf",
            cox_df["coxphf_full_ms"],
            "numba_full_ms",
        ),
    ]:
        ratio = baseline / df[col]
        rows.append(
            [
                workload,
                baseline_label,
                fmt_x(ratio.iloc[0]),
                fmt_x(ratio.iloc[-1]),
            ]
        )

    k_smallest = {int(logistic_df["k"].iloc[0]), int(cox_df["k"].iloc[0])}
    small_header = (
        f"Speedup at k={next(iter(k_smallest))}"
        if len(k_smallest) == 1
        else "Speedup at smallest k"
    )
    table = markdown_table(
        ["Workload", "Baseline", small_header, "Speedup at largest k"],
        rows,
        "llrr",
    )

    logistic_full_ratio = logistic_df["logistf_full_ms"] / logistic_df["numba_full_ms"]
    cox_full_ratio = cox_df["coxphf_full_ms"] / cox_df["numba_full_ms"]
    numpy_logistic_x = (
        logistic_df["logistf_full_ms"] / logistic_df["numpy_full_ms"]
    ).iloc[-1]
    numpy_cox_x = (cox_df["coxphf_full_ms"] / cox_df["numpy_full_ms"]).iloc[-1]
    k_lo_max = int(logistic_df["k"].iloc[-1])
    k_cox_max = int(cox_df["k"].iloc[-1])

    runs = runs_sentence(logistic_meta, cox_meta)

    return f"""## Summary

For the full workflow, firthmodels' Numba backend is **{fmt_x(logistic_full_ratio.iloc[-1])}
faster than logistf** (k={k_lo_max}) and **{fmt_x(cox_full_ratio.iloc[-1])} faster than
coxphf** (k={k_cox_max}). Without Numba, the pure NumPy backend is {fmt_x(numpy_logistic_x)}
faster than logistf and {fmt_x(numpy_cox_x)} faster than coxphf.

{table}

{agreement}

Python is timed with time.perf_counter() around each call, after JIT-warming the Numba
backend. R packages are timed in-process with `microbenchmark` inside a single
R session, so R startup and data transfer are excluded, while formula parsing and
model-frame construction are included because logistf and coxphf only
offer formula interfaces. brglm2 is run with `check_aliasing=FALSE` to avoid
extra overhead from its default aliasing check. {runs}"""


def generate_logistic_section(df: pd.DataFrame, plot_name: str) -> str:
    best_r = df[["logistf_fit_ms", "brglm2_as_fit_ms", "brglm2_mpl_fit_ms"]].min(axis=1)

    fit_rows = [
        [
            str(int(row["k"])),
            *fmt_ranked_times(
                [
                    row["numba_fit_ms"],
                    row["numpy_fit_ms"],
                    row["logistf_fit_ms"],
                    row["brglm2_as_fit_ms"],
                    row["brglm2_mpl_fit_ms"],
                ],
                r_indices=[2, 3, 4],
            ),
            fmt_x(best_r.iloc[i] / row["numba_fit_ms"]),
            fmt_x(best_r.iloc[i] / row["numpy_fit_ms"]),
        ]
        for i, (_, row) in enumerate(df.iterrows())
    ]
    fit_table = markdown_table(
        [
            "k",
            "firthmodels<br>(numba)",
            "firthmodels<br>(numpy)",
            "logistf",
            "brglm2<br>(AS-mean)",
            "brglm2<br>(MPL-Jeffreys)",
            "numba speedup<br>vs next fastest",
            "numpy speedup<br>vs next fastest",
        ],
        fit_rows,
        "rrrrrrrr",
    )

    full_rows = [
        [
            str(int(row["k"])),
            *fmt_ranked_times(
                [row["numba_full_ms"], row["numpy_full_ms"], row["logistf_full_ms"]],
                r_indices=[2],
            ),
            fmt_x(row["logistf_full_ms"] / row["numba_full_ms"]),
            fmt_x(row["logistf_full_ms"] / row["numpy_full_ms"]),
        ]
        for _, row in df.iterrows()
    ]
    full_table = markdown_table(
        [
            "k",
            "firthmodels<br>(numba)",
            "firthmodels<br>(numpy)",
            "logistf",
            "numba speedup<br>vs logistf",
            "numpy speedup<br>vs logistf",
        ],
        full_rows,
        "rrrrrr",
    )

    n = int(df["n"].iloc[0])
    events_lo, events_hi = int(df["events"].min()), int(df["events"].max())
    epv_lo, epv_hi = df["epv"].min(), df["epv"].max()
    k_lo, k_hi = int(df["k"].iloc[0]), int(df["k"].iloc[-1])

    brglm2_rel_gap = (
        (df["brglm2_as_fit_ms"] - df["brglm2_mpl_fit_ms"]).abs()
        / df[["brglm2_as_fit_ms", "brglm2_mpl_fit_ms"]].min(axis=1)
    ).max()

    return f"""## Firth logistic regression

Compared against R [logistf](https://cran.r-project.org/package=logistf) and
[brglm2](https://cran.r-project.org/package=brglm2) on simulated data with
n = {n:,} observations, a {bl.EVENT_RATE:.0%} target event rate, and
k = {k_lo} to {k_hi} features.

![Logistic benchmark scaling, log time axis]({plot_name})

The time axis is log-scale, so a constant vertical gap means a constant speedup ratio.
All values are the minimum observed wall-clock time across repeated runs, in milliseconds.

### Fit + Wald inference

{fit_table}

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns. The two brglm2
fitting methods have nearly identical timings (within {brglm2_rel_gap:.1%} at every k).

### Full workflow: fit + LRT + profile likelihood CI

brglm2 is not included here because it does not provide penalized LRT
p-values or profile likelihood CIs.

{full_table}"""


def generate_cox_section(df: pd.DataFrame, plot_name: str) -> str:
    fit_rows = [
        [
            str(int(row["k"])),
            *fmt_ranked_times(
                [row["numba_fit_ms"], row["numpy_fit_ms"], row["coxphf_fit_ms"]],
                r_indices=[2],
            ),
            fmt_x(row["coxphf_fit_ms"] / row["numba_fit_ms"]),
            fmt_x(row["coxphf_fit_ms"] / row["numpy_fit_ms"]),
        ]
        for _, row in df.iterrows()
    ]
    fit_table = markdown_table(
        [
            "k",
            "firthmodels<br>(numba)",
            "firthmodels<br>(numpy)",
            "coxphf",
            "numba speedup<br>vs coxphf",
            "numpy speedup<br>vs coxphf",
        ],
        fit_rows,
        "rrrrrr",
    )

    full_rows = [
        [
            str(int(row["k"])),
            *fmt_ranked_times(
                [row["numba_full_ms"], row["numpy_full_ms"], row["coxphf_full_ms"]],
                r_indices=[2],
            ),
            fmt_x(row["coxphf_full_ms"] / row["numba_full_ms"]),
            fmt_x(row["coxphf_full_ms"] / row["numpy_full_ms"]),
        ]
        for _, row in df.iterrows()
    ]
    full_table = markdown_table(
        [
            "k",
            "firthmodels<br>(numba)",
            "firthmodels<br>(numpy)",
            "coxphf",
            "numba speedup<br>vs coxphf",
            "numpy speedup<br>vs coxphf",
        ],
        full_rows,
        "rrrrrr",
    )

    n = int(df["n"].iloc[0])
    events = int(df["events"].iloc[0])
    k_lo, k_hi = int(df["k"].iloc[0]), int(df["k"].iloc[-1])

    return f"""## Firth Cox proportional hazards

Compared against R [coxphf](https://cran.r-project.org/package=coxphf) on
simulated survival data with n = {n} observations, a 20% event rate,
and k = {k_lo} to {k_hi} features.

![Cox benchmark scaling, log time axis]({plot_name})

The time axis is log-scale, so a constant vertical gap means a constant speedup ratio.
All values are the minimum observed wall-clock time across repeated runs, in milliseconds.

### Fit + Wald inference

{fit_table}

The fastest time at each k is **bolded**. The next-fastest package (not including the firthmodels
NumPy backend) is <ins>underlined</ins> and is the baseline for the speedup columns.

### Full workflow: fit + LRT + profile likelihood CI

{full_table}"""


def generate_correctness_section(
    benchmark_rows: list[DeviationRow],
    meta_notes: list[str],
    version_note: str | None,
) -> str:
    parts = [
        f"""## Correctness

The benchmark scripts abort if firthmodels disagrees with the R reference by more than
{bl.COEF_TOL:g} on coefficients, profile CI bounds, or p-values (and also cross-check
the numba backend against the numpy backend)."""
    ]

    if benchmark_rows:
        table = markdown_table(
            ["Comparison", "Quantity", "Max abs. deviation"],
            [[c, q, fmt_dev(v)] for c, q, v in benchmark_rows],
            "llr",
        )
        parts.append(
            "Maximum over all coefficients and all k during the benchmark run "
            "(recorded in the .meta.json run metadata):\n\n" + table
        )

    for note in meta_notes:
        parts.append(note)

    if version_note:
        parts.append(version_note)

    return "\n\n".join(parts)


def runs_clause(
    meta: dict | None,
    script_default: int,
    reduced_lib: str,
    default_reduce_after: int,
) -> str:
    """Compact repetition-count clause from metadata, or an honest fallback."""
    if meta and "n_runs" in meta:
        text = f"{meta['n_runs']} per configuration"
        reduced = meta.get("r_reduced_runs")
        if reduced:
            libs = ", ".join(reduced["libraries"])
            text += f", {libs} reduced to {reduced['n_runs']} for k > {reduced['for_k_above']}"
        return text
    default_reduced = (
        script_default if script_default <= 3 else max(3, script_default // 3)
    )
    return (
        f"not recorded in metadata; the script default is {script_default} "
        f"per configuration, {reduced_lib} reduced to {default_reduced} "
        f"for k > {default_reduce_after}"
    )


def generate_environment_section(info: dict[str, str]) -> str:
    runtime_blas = get_runtime_blas()
    if runtime_blas:
        numpy_blas_display = f"{runtime_blas[0]} (openblas {runtime_blas[1]}, runtime)"
    else:
        numpy_blas_display = (
            f"{info.get('numpy_blas_build', 'unknown')} (build metadata)"
        )

    rows = [
        ["OS", info.get("os", "unknown")],
        ["CPU", info.get("cpu", "unknown")],
        ["Python", info.get("python", "unknown")],
        ["firthmodels", info.get("firthmodels", "unknown")],
        [
            "NumPy / SciPy / Numba",
            f"{info.get('numpy', '?')} / {info.get('scipy', '?')} / {info.get('numba', '?')}",
        ],
        ["NumPy BLAS", numpy_blas_display],
        ["R", info.get("r", "unknown")],
        [
            "logistf / brglm2 / coxphf",
            f"{info.get('logistf', '?')} / {info.get('brglm2', '?')} / {info.get('coxphf', '?')}",
        ],
        ["R BLAS", info.get("r_blas", "unknown")],
    ]
    table = markdown_table(["Component", "Version"], rows, "ll")

    r_blas_path = info.get("r_blas", "unknown")
    same_blas = (
        runtime_blas is not None
        and r_blas_path not in ("", "unknown")
        and os.path.exists(runtime_blas[0])
        and os.path.exists(r_blas_path)
        and os.path.realpath(runtime_blas[0]) == os.path.realpath(r_blas_path)
    )
    if same_blas:
        assert runtime_blas is not None
        blas_sentence = (
            "NumPy and R link to the same "
            f"BLAS library (`{os.path.realpath(runtime_blas[0])}`)."
        )
    else:
        blas_sentence = (
            "The BLAS libraries used by the two stacks are listed above; they "
            "could not be confirmed to be the same shared object."
        )

    flags_summary = key_compile_flags(info.get("r_cflags", ""))
    return f"""## Environment

Collected at report-generation time on the benchmark machine.

{table}

{blas_sentence} BLAS threading is left at library defaults for both stacks.
R packages are compiled from source with `{flags_summary}`,
so the R timings are not slowed by a conservative build.

<details>
<summary>Full R package compile flags (current configuration)</summary>

```
CFLAGS:  {info.get("r_cflags", "unknown")}
FCFLAGS: {info.get("r_fcflags", "unknown")}
```

</details>"""


def generate_report(
    logistic_df: pd.DataFrame,
    cox_df: pd.DataFrame,
    env_info: dict[str, str],
    logistic_meta: dict | None,
    cox_meta: dict | None,
    logistic_plot_name: str,
    cox_plot_name: str,
) -> str:
    benchmark_rows = metadata_deviation_rows(
        logistic_meta, "Logistic"
    ) + metadata_deviation_rows(cox_meta, "Cox")

    meta_notes = []
    for meta, name in [(logistic_meta, "logistic"), (cox_meta, "Cox")]:
        if meta is None:
            meta_notes.append(
                f"No run metadata was found alongside the {name} CSV (it "
                "predates metadata support), so benchmark-time verification "
                "cannot be confirmed from the files alone."
            )
        elif not (meta.get("verification") or {}).get("performed"):
            meta_notes.append(
                f"The {name} run metadata records that verification was NOT "
                "performed during that benchmark run."
            )

    worst = max((v for _, _, v in benchmark_rows), default=None)
    agreement = agreement_sentence(worst, bl.COEF_TOL)

    current_version = env_info.get("firthmodels", "unknown")
    version_note = None
    recorded = {
        v
        for v in (
            metadata_firthmodels_version(logistic_meta),
            metadata_firthmodels_version(cox_meta),
        )
        if v
    }
    if recorded and recorded != {current_version}:
        version_note = (
            "Note: the benchmark run(s) recorded firthmodels "
            f"{', '.join(sorted(recorded))}, while the currently installed "
            f"version is {current_version}. Timings describe the recorded "
            "version."
        )

    return f"""# Benchmarks

Benchmarking of [firthmodels](https://github.com/jzluo/firthmodels) against implementations
of Firth-penalized logistic regression (R [logistf](https://cran.r-project.org/package=logistf),
[brglm2](https://cran.r-project.org/package=brglm2)) and Cox regression
(R [coxphf](https://cran.r-project.org/package=coxphf)).

{generate_summary(logistic_df, cox_df, agreement, logistic_meta, cox_meta)}

---

{generate_logistic_section(logistic_df, logistic_plot_name)}

---

{generate_cox_section(cox_df, cox_plot_name)}

---

{generate_correctness_section(benchmark_rows, meta_notes, version_note)}

---

{generate_environment_section(env_info)}

---

## Reproducing these results

Requires R with the logistf, brglm2, coxphf, survival, microbenchmark, and
jsonlite packages installed.

```bash
# Run benchmarks (writes CSVs, R reference values, and .meta.json run metadata)
uv run python benchmarks/benchmark_logistic.py -o benchmarks/logistic_results.csv
uv run python benchmarks/benchmark_cox.py -o benchmarks/cox_results.csv

# Generate plots and this README
uv run python benchmarks/generate_report.py
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmarks README from CSV results"
    )
    parser.add_argument(
        "--logistic-results",
        type=Path,
        default=BENCHMARKS_DIR / "logistic_results.csv",
    )
    parser.add_argument(
        "--cox-results",
        type=Path,
        default=BENCHMARKS_DIR / "cox_results.csv",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=BENCHMARKS_DIR / "README.md",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to plot file names (default: none)",
    )
    args = parser.parse_args()

    for csv in (args.logistic_results, args.cox_results):
        if not csv.exists():
            print(f"Error: {csv} not found; run the benchmark first", file=sys.stderr)
            sys.exit(1)

    logistic_df = pd.read_csv(args.logistic_results)
    cox_df = pd.read_csv(args.cox_results)
    logistic_meta = load_metadata(args.logistic_results)
    cox_meta = load_metadata(args.cox_results)

    print("Collecting environment info...", file=sys.stderr)
    env_info = get_environment_info()

    print("Generating plots...", file=sys.stderr)
    logistic_plot = args.logistic_results.with_name(
        args.logistic_results.stem + args.suffix + ".png"
    )
    cox_plot = args.cox_results.with_name(args.cox_results.stem + args.suffix + ".png")
    save_scaling_plot(
        logistic_df,
        [
            ("Fit + Wald inference", LOGISTIC_FIT_SERIES),
            ("Fit + LRT + profile CI", LOGISTIC_FULL_SERIES),
        ],
        logistic_plot,
        f"Firth logistic regression, n={bl.N_SAMPLES} | "
        f"firthmodels {env_info.get('firthmodels', '?')} | "
        f"logistf {env_info.get('logistf', '?')}, brglm2 {env_info.get('brglm2', '?')}",
    )
    save_scaling_plot(
        cox_df,
        [
            ("Fit + Wald inference", COX_FIT_SERIES),
            ("Fit + LRT + profile CI", COX_FULL_SERIES),
        ],
        cox_plot,
        f"Firth Cox PH, n={bc.N_SAMPLES} | "
        f"firthmodels {env_info.get('firthmodels', '?')} | "
        f"coxphf {env_info.get('coxphf', '?')}",
    )

    print("Generating README...", file=sys.stderr)
    content = generate_report(
        logistic_df,
        cox_df,
        env_info,
        logistic_meta,
        cox_meta,
        logistic_plot.name,
        cox_plot.name,
    )
    args.out.write_text(content)
    print(f"README written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
