#!/usr/bin/env Rscript
# Pool per-state log odds ratios three ways.
#
#   Rscript meta_pool.R <per_state.csv> <out.json>
#
# per_state.csv needs columns: site, log_or, se.
#
#   dersimonian_laird  the estimator S9 registers
#   common_effect      inverse-variance, for comparison with a one-stage model
#                      that imposes one age coefficient on every state
#   reml_hartung_knapp exploratory sensitivity analysis

suppressPackageStartupMessages({
  library(metafor)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
per_state <- read.csv(args[1])
stopifnot(all(c("site", "log_or", "se") %in% names(per_state)))

summarize <- function(fit) {
  list(
    log_or = unname(fit$beta[1]), se = fit$se,
    or = exp(unname(fit$beta[1])),
    ci = c(exp(fit$ci.lb), exp(fit$ci.ub)),
    p = fit$pval,
    tau2 = if (is.null(fit$tau2)) NA else fit$tau2,
    i2 = if (is.null(fit$I2)) NA else fit$I2,
    q = fit$QE, q_p = fit$QEp,
    k = fit$k
  )
}

random <- rma(yi = per_state$log_or, sei = per_state$se, method = "DL")
prediction <- predict(random, transf = exp)

results <- list(
  r_version = R.version.string,
  metafor_version = as.character(packageVersion("metafor")),
  fitted_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  dersimonian_laird = summarize(random),
  prediction_interval_or = c(prediction$pi.lb, prediction$pi.ub),
  common_effect = summarize(rma(yi = per_state$log_or, sei = per_state$se, method = "EE")),
  reml_hartung_knapp = summarize(
    rma(yi = per_state$log_or, sei = per_state$se, method = "REML", test = "knha"))
)

write(toJSON(results, auto_unbox = TRUE, digits = 12, pretty = TRUE), args[2])
cat("wrote", args[2], "\n")
