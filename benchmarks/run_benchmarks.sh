#!/usr/bin/env bash
# Run the benchmark suite and regenerate the report.
#
# Usage:
#   benchmarks/run_benchmarks.sh            # logistic + cox + report
#   benchmarks/run_benchmarks.sh logistic   # logistic only + report
#   benchmarks/run_benchmarks.sh cox        # cox only + report

set -euo pipefail
cd "$(dirname "$0")/.."

target="${1:-all}"
case "$target" in
    all|logistic|cox) ;;
    *) echo "Usage: $0 [logistic|cox]" >&2; exit 1 ;;
esac

if [[ "$target" == "all" || "$target" == "logistic" ]]; then
    uv run python benchmarks/benchmark_logistic.py -o benchmarks/logistic_results.csv
fi

if [[ "$target" == "all" || "$target" == "cox" ]]; then
    uv run python benchmarks/benchmark_cox.py -o benchmarks/cox_results.csv
fi

uv run python benchmarks/generate_report.py
