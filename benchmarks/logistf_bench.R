# Benchmark logistf for Firth logistic regression
#
# Usage: Rscript logistf_bench.R <data_dir> <k_values> <n_runs> <max_iter> <max_halfstep> <tol>
#   data_dir: Directory containing data_k{k}.csv files
#   k_values: Comma-separated k values (e.g., "5,10,15")
#   n_runs: Number of benchmark runs
#   max_iter, max_halfstep, tol: Solver parameters
#
# Outputs JSON to stdout with structure:
# {"5": {"fit_times": [...], "fit_coef": [...], ...}, "10": {...}}

library(logistf)
library(microbenchmark)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6) {
    stop("Usage: Rscript logistf_bench.R <data_dir> <k_values> <n_runs> <max_iter> <max_halfstep> <tol>")
}

data_dir <- args[1]
k_values <- as.integer(strsplit(args[2], ",")[[1]])
n_runs <- as.integer(args[3])
max_iter <- as.integer(args[4])
max_halfstep <- as.integer(args[5])
tol <- as.numeric(args[6])

ctrl <- logistf.control(maxit = max_iter, maxhs = max_halfstep,
                        lconv = tol, gconv = tol, xconv = tol)
plctrl <- logistpl.control(maxit = max_iter, maxhs = max_halfstep,
                           lconv = tol, xconv = tol)

results <- list()

for (k in k_values) {
    message("  k=", k, "...")

    df <- read.csv(paste0(data_dir, "/data_k", k, ".csv"))

    # Fit only: warmup then benchmark
    fit <- logistf(y ~ ., data = df, pl = FALSE, control = ctrl)
    mb_fit <- microbenchmark(
        fit <- logistf(y ~ ., data = df, pl = FALSE, control = ctrl),
        times = n_runs
    )

    fit_result <- list(
        fit_times = mb_fit$time / 1e9,
        fit_coef = unname(coef(fit)[-1]),
        fit_intercept = unname(coef(fit)[1])
    )

    rm(fit, mb_fit)

    # Full workflow: warmup then benchmark
    fit <- logistf(y ~ ., data = df, pl = TRUE, control = ctrl, plcontrol = plctrl)
    mb_full <- microbenchmark(
        fit <- logistf(y ~ ., data = df, pl = TRUE, control = ctrl, plcontrol = plctrl),
        times = n_runs
    )

    full_result <- list(
        full_times = mb_full$time / 1e9,
        full_coef = unname(coef(fit)[-1]),
        full_intercept = unname(coef(fit)[1]),
        ci_lower = unname(fit$ci.lower),
        ci_upper = unname(fit$ci.upper),
        full_pval = unname(fit$prob)
    )

    results[[as.character(k)]] <- c(fit_result, full_result)

    rm(df, fit, mb_full)
    gc(verbose = FALSE)
}

cat(toJSON(results, auto_unbox = TRUE, digits = 16))
