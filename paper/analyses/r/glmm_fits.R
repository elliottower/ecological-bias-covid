#!/usr/bin/env Rscript
# Binomial mixed models on site x elderly x male x comorbidity cells.
#
#   Rscript glmm_fits.R <cells.csv> <out.json>
#
# The patient-level covariates are binary, so these cell fits have the same
# likelihood as the patient-level fits.
#
#   s2_within_between_random_intercept  the exploratory correlated-random-effects
#                                       replacement for the registered S2, whose
#                                       between-state term is not identified
#   s9_one_stage_random_intercept       the one-stage model S9 registers
#   s9_one_stage_random_slope           the same with a state-varying age slope

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
cells <- read.csv(args[1])
stopifnot(all(c("site", "elderly", "male", "has_comorbidity", "deaths", "alive", "n") %in% names(cells)))

site_mean <- tapply(cells$n * cells$elderly, cells$site, sum) / tapply(cells$n, cells$site, sum)
cells$elderly_between <- as.numeric(site_mean[as.character(cells$site)])
cells$elderly_within <- cells$elderly - cells$elderly_between
cells$site <- factor(cells$site)

summarize <- function(fit, terms) {
  coefs <- summary(fit)$coefficients
  out <- list()
  for (term in terms) {
    est <- unname(coefs[term, "Estimate"])
    se <- unname(coefs[term, "Std. Error"])
    out[[term]] <- list(
      log_or = est, se = se,
      or = exp(est),
      ci = c(exp(est - 1.96 * se), exp(est + 1.96 * se)),
      p = unname(coefs[term, "Pr(>|z|)"])
    )
  }
  messages <- fit@optinfo$conv$lme4$messages
  out$random_effects_sd <- as.data.frame(VarCorr(fit))[, c("grp", "var1", "sdcor")]
  out$converged <- is.null(messages)
  out$convergence_messages <- if (is.null(messages)) character(0) else messages
  out$n_cells <- nrow(fit@frame)
  out$n_patients <- sum(cells$n)
  out
}

results <- list(
  r_version = R.version.string,
  lme4_version = as.character(packageVersion("lme4")),
  fitted_at = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  s2_within_between_random_intercept = summarize(
    glmer(cbind(deaths, alive) ~ elderly_within + elderly_between + (1 | site),
          data = cells, family = binomial),
    c("elderly_within", "elderly_between")),
  s9_one_stage_random_intercept = summarize(
    glmer(cbind(deaths, alive) ~ elderly + (1 | site),
          data = cells, family = binomial),
    c("elderly")),
  s9_one_stage_random_slope = summarize(
    glmer(cbind(deaths, alive) ~ elderly + (1 + elderly | site),
          data = cells, family = binomial),
    c("elderly"))
)

write(toJSON(results, auto_unbox = TRUE, digits = 12, pretty = TRUE), args[2])
cat("wrote", args[2], "\n")
