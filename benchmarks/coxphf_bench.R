# Benchmark coxphf for Firth-penalized Cox proportional hazards
#
# Usage: Rscript coxphf_bench.R <data_dir> <k_values> <n_runs> <max_iter> <max_halfstep> <xtol> <gtol>
#   data_dir: Directory containing data_k{k}.csv files (with time, event, x0, x1, ...)
#   k_values: Comma-separated k values (e.g., "5,10,15")
#   n_runs: Number of benchmark runs
#   max_iter, max_halfstep, xtol, gtol: Solver parameters
#
# Outputs JSON to stdout with structure:
# {"5": {"fit_times": [...], "fit_coef": [...], ...}, "10": {...}}

library(coxphf)
library(survival)
library(microbenchmark)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
    stop("Usage: Rscript coxphf_bench.R <data_dir> <k_values> <n_runs> <max_iter> <max_halfstep> <xtol> <gtol>")
}

data_dir <- args[1]
k_values <- as.integer(strsplit(args[2], ",")[[1]])
n_runs <- as.integer(args[3])
max_iter <- as.integer(args[4])
max_halfstep <- as.integer(args[5])
xtol <- as.numeric(args[6])
gtol <- as.numeric(args[7])

results <- list()

for (k in k_values) {
    message("  k=", k, "...")

    df <- read.csv(paste0(data_dir, "/data_k", k, ".csv"))

    # Build formula: Surv(time, event) ~ x0 + x1 + ... + x{k-1}
    covar_names <- paste0("x", 0:(k-1))
    formula_str <- paste("Surv(time, event) ~", paste(covar_names, collapse = " + "))
    fml <- as.formula(formula_str)

    # Fit only (pl=FALSE for Wald inference, faster)
    fit <- coxphf(fml, data = df, pl = FALSE,
                  maxit = max_iter, maxhs = max_halfstep,
                  epsilon = xtol, gconv = gtol)
    mb_fit <- microbenchmark(
        fit <- coxphf(fml, data = df, pl = FALSE,
                      maxit = max_iter, maxhs = max_halfstep,
                      epsilon = xtol, gconv = gtol),
        times = n_runs
    )

    fit_result <- list(
        fit_times = mb_fit$time / 1e9,
        fit_coef = unname(fit$coefficients),
        fit_loglik = fit$loglik[2],
        fit_iter = fit$iter
    )

    rm(fit, mb_fit)

    # Full workflow (pl=TRUE for profile likelihood CIs and LRT p-values)
    fit <- coxphf(fml, data = df, pl = TRUE,
                  maxit = max_iter, maxhs = max_halfstep,
                  epsilon = xtol, gconv = gtol)
    mb_full <- microbenchmark(
        fit <- coxphf(fml, data = df, pl = TRUE,
                      maxit = max_iter, maxhs = max_halfstep,
                      epsilon = xtol, gconv = gtol),
        times = n_runs
    )

    # coxphf returns CIs on exp(coef) scale, convert to log scale for comparison
    full_result <- list(
        full_times = mb_full$time / 1e9,
        full_coef = unname(fit$coefficients),
        full_loglik = fit$loglik[2],
        ci_lower = log(unname(fit$ci.lower)),
        ci_upper = log(unname(fit$ci.upper)),
        full_pval = unname(fit$prob),
        full_iter = fit$iter
    )

    results[[as.character(k)]] <- c(fit_result, full_result)

    rm(df, fit, mb_full)
    gc(verbose = FALSE)
}

cat(toJSON(results, auto_unbox = TRUE, digits = 16))
