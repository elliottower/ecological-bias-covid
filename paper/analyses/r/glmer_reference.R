#!/usr/bin/env Rscript
# An independent fit of a binomial random-intercept model, for checking `glmm.py`.
#
#   Rscript glmer_reference.R <cells.csv> <out.json> <nAGQ>
#
# cells.csv carries deaths, alive, a `group` column and the design columns x1..xk.
# The comparison this serves is `validate_glmm.py`, which writes both sets of numbers.

suppressPackageStartupMessages({
  library(lme4)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
cells <- read.csv(args[1])
cells$group <- factor(cells$group)
nagq <- as.integer(args[3])

terms <- grep("^x[0-9]+$", names(cells), value = TRUE)
formula <- as.formula(paste("cbind(deaths, alive) ~",
                            paste(terms, collapse = " + "), "+ (1 | group)"))
cat("fitting:", deparse1(formula), "on", nrow(cells), "cells, nAGQ =", nagq, "\n")

started <- Sys.time()
fit <- glmer(formula, data = cells, family = binomial, nAGQ = nagq,
             control = glmerControl(optimizer = "bobyqa",
                                    optCtrl = list(maxfun = 100000)))
coefs <- summary(fit)$coefficients
messages <- fit@optinfo$conv$lme4$messages

write(toJSON(list(
  r_version = R.version.string,
  lme4_version = as.character(packageVersion("lme4")),
  seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
  nAGQ = nagq,
  formula = deparse1(formula),
  terms = c("(Intercept)", terms),
  estimate = unname(coefs[, "Estimate"]),
  se = unname(coefs[, "Std. Error"]),
  random_intercept_sd = as.data.frame(VarCorr(fit))$sdcor[1],
  log_likelihood = as.numeric(logLik(fit)),
  singular = isSingular(fit),
  converged = is.null(messages),
  convergence_messages = if (is.null(messages)) character(0) else messages
), auto_unbox = TRUE, digits = 12, pretty = TRUE), args[2])
cat("wrote", args[2], "\n")
