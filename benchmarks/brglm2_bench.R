# Benchmark brglm2 for Firth logistic regression (AS_mean and MPL_Jeffreys)
#
# Usage: Rscript brglm2_bench.R <data_dir> <k_values> <n_runs> <max_iter> <tol>
#   data_dir: Directory containing data_k{k}.csv files
#   k_values: Comma-separated k values (e.g., "5,10,15")
#   n_runs: Number of benchmark runs
#   max_iter, tol: Solver parameters
#
# Outputs JSON to stdout with structure:
# {"AS_mean": {"5": {"fit_times": [...], ...}, ...}, "MPL_Jeffreys": {...}}

library(brglm2)
library(microbenchmark)
library(jsonlite)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
    stop("Usage: Rscript brglm2_bench.R <data_dir> <k_values> <n_runs> <max_iter> <tol>")
}

data_dir <- args[1]
k_values <- as.integer(strsplit(args[2], ",")[[1]])
n_runs <- as.integer(args[3])
max_iter <- as.integer(args[4])
tol <- as.numeric(args[5])

methods <- c("AS_mean", "MPL_Jeffreys")
results <- list()

for (method in methods) {
    message("  method=", method)
    results[[method]] <- list()

    for (k in k_values) {
        message("    k=", k, "...")

        df <- read.csv(paste0(data_dir, "/data_k", k, ".csv"))

        # Warmup
        fit <- glm(y ~ ., data = df, family = binomial(),
                   method = "brglmFit", type = method,
                   control = list(epsilon = tol, maxit = max_iter,
                                  check_aliasing = FALSE))

        # Benchmark
        mb <- microbenchmark(
            fit <- glm(y ~ ., data = df, family = binomial(),
                       method = "brglmFit", type = method,
                       control = list(epsilon = tol, maxit = max_iter,
                                      check_aliasing = FALSE)),
            times = n_runs
        )

        results[[method]][[as.character(k)]] <- list(
            fit_times = mb$time / 1e9,
            fit_coef = unname(coef(fit)[-1]),
            fit_intercept = unname(coef(fit)[1])
        )

        rm(df, fit, mb)
        gc(verbose = FALSE)
    }
}

cat(toJSON(results, auto_unbox = TRUE, digits = 16))
