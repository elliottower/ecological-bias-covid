# Implementation amendment, 2026-09-29

**Amends:** nothing in `ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md`, which stays frozen at commit
`a38196b`, tag `registration-s11-s12`. This file fixes the implementation decisions the
registration left open, and it is committed and tagged before S11 and S12 are run with the
code it describes.

Every number in this file is a threshold, a node count or a design size fixed by choice
before the run. None of them is a measured result.

**Foreknowledge.** S11 and S12 were run on 2026-09-24 with an earlier implementation, and
those numbers were seen; they are archived in `results/superseded/`. H6 has never been
fitted: `lme4::glmer` segfaulted on its cell table, and no contextual estimate exists in any
result file. Every decision below about H6 is therefore made before the quantity it governs
has been computed. The decisions about S11 and S12 concern estimator mechanics and failure
handling, not the hypotheses' direction, and the earlier numbers are superseded by this run.

## A1. The estimator for H6

The registration names a random-intercept model and no estimator. H6 is fitted with
`lme4::glmer` at `nAGQ = 15` [@bates2015lme4], which evaluates the group integral by
adaptive Gauss-Hermite quadrature [@pinheiro1995approximations].

`glmer` segfaults inside `pwrssUpdate` on this design on the author's laptop, under R
4.1.2 with lme4 1.1-33 and Matrix 1.5-1, and the crash survived upgrading Matrix and
reinstalling lme4. It is a property of that installation and not of the model: on Debian
with R 4.2.2, lme4 1.1-31 and Matrix 1.5-3, the same sixteen-term design over 32 groups
and 160,000 cells fits in 181 seconds. H6 is therefore fitted in that environment, and
the versions are recorded beside the result.

`glmm.py` is an independent implementation of the same estimator, written in Python
before the segfault was known to be environment-specific, and it is kept as a
reproduction rather than as the primary fit. On the design above the two agree to
1.7 x 10^-5 in the largest coefficient, 3.4 x 10^-7 in the standard errors and
8.7 x 10^-7 in sigma; on S9's model, fitted on the snapshot's own cells, they agree to
8.9 x 10^-8 in the log odds ratio and 4.1 x 10^-8 in its standard error. A disagreement
between them beyond 10^-4 in any coefficient leaves H6 not evaluable.

S2 and S9 keep their `lme4` fits, so every mixed model in the paper is fitted by the same
software.

## A2. Uncertainty on the contextual coefficient

The reported interval is a **profile-likelihood** interval, obtained by fixing the
coefficient, re-optimizing every other coefficient and log σ, and inverting the likelihood
ratio [@venzon1988profile]. With 32 states this is preferred to a Wald interval. A Wald
interval from the Hessian of the marginal log-likelihood, and a standard error conditional on
the fitted σ, are reported beside it as computational checks and are not the interval.

## A3. The boundary

The variance component is fitted at its boundary directly: the same design refitted as a
plain binomial GLM is the exact σ = 0 maximum, and the two solutions are compared by
deviance. Below 10⁻⁶ the component is at the boundary. Where the deviance difference is read
as a test, its null is the 50:50 mixture of a point mass at zero and chi-square on one degree
of freedom [@self1987asymptotic; @stram1994variance], and the mixture p-value is what the
result file carries.

**A boundary solution leaves H6 not evaluable.** The registration voids a hypothesis whose
model does not converge and is silent on a converged boundary solution. At σ = 0 the model
treats records within a state as independent, so an interval on a state-level covariate would
rest on 3,993,464 independent observations of 32 distinct values. The estimate is reported
descriptively and the registered criterion is not applied.

## A4. Numerical acceptance gates

H6 is evaluable only if every gate below passes. A failure leaves the estimate reported
descriptively with all diagnostics, and the registered criterion unapplied.

| Gate | Condition |
|---|---|
| Registered covariates | the model-5 centered spline, sex, nine comorbidity flags, the standardized state elderly share |
| Convergence | `glmer` reports no convergence warning, and the Python reproduction converges from at least one variance start |
| Agreement | the two implementations differ by less than 10^-4 in every coefficient |
| Boundary | deviance difference against the σ = 0 fit above 10⁻⁶ |
| Finite estimate | every coefficient and standard error finite |
| Hessian definiteness | the marginal Hessian is positive definite |
| Hessian stability | the standard errors change by less than 10⁻³ in relative terms between finite-difference steps of 10⁻⁴ and 3 × 10⁻⁴ |
| Conditioning | the Hessian's condition number is below 10¹⁰ |
| Gradient | the largest absolute finite-difference gradient, divided by the log-likelihood, is below 10⁻⁶ |
| Profile | the profile returns two finite endpoints bracketing the estimate |

## A5. Which interval is primary, decided before H6 is fitted

`validate_glmm.py` estimates repeated-sample coverage of the Wald and profile intervals at
12, 32 and 120 groups and at variance components of 0.35 and 0. The rule, fixed here:

- Profile coverage within 0.93–0.97 at 32 groups: the profile interval is primary, and the
  Wald interval is reported as a diagnostic.
- Profile coverage outside that range but above 0.90: the profile interval is primary and the
  paper states the coverage estimate beside it.
- Profile coverage at or below 0.90: H6 is reported descriptively and no registered decision
  is taken from it.

The interval is not chosen after seeing which one makes H6 hold.

## A6. The validity domain

The catalogue admits states 1–32 and municipality codes 1–570 within a state, with 997, 998
and 999 for a municipality and 99 for a sector as the "no especificado" family. A record
outside those ranges carries no residence state, no municipality or no sector, and is
therefore outside the site definition that uses the invalid field. A treating-unit state
outside 1–32 would change the primary cohort, so the loader refuses the file instead. Every
exclusion category is counted into `results/mexico_loader_audit.json`.

An invalid or out-of-range onset date leaves the calendar month missing rather than rendering
as a string, so models 5 and 6 are fitted on the same records.

## A7. Failed draws

A bootstrap draw that raises a numerical failure, fails to converge, or carries no
between-site exposure variation is recorded by reason and contributes no estimate. No draw is
regenerated to replace a failure. Each interval reports the draws requested, the draws valid
and the failures by reason; more than 1% failing, or fewer than 100 valid draws, marks the
analysis unstable, and below 100 valid draws no interval is reported at all. An interval
computed with any failures present is conditional on the valid draws and says so.

## A8. What H6 estimates

H6 estimates the coefficient on a state's standardized elderly share in a random-intercept
model adjusted for the model-5 record-level covariates. It is not a full
correlated-random-effects specification, and the paper describes it as a residual
model-based state-level association rather than a contextual effect.

## A9. Execution

S11 and S12 refuse to run while `paper/analyses` has uncommitted changes. Every result file
carries the execution commit, the snapshot hash, the package versions and a hash of every
analysis source file. Results are replaced atomically and the previous file is archived into
`results/superseded/`.

## Maximum claim under this amendment

The registration's claims are unchanged. This file adds no hypothesis and removes none. What
it fixes is which estimator produces H6's number, which interval is reported, what makes H6
unevaluable, and what happens to a failed draw. If H6 is evaluable, the paper can say that a
state's elderly share retains, or does not retain, an association with an individual death
under the model-5 covariates, with a profile interval. If any gate fails, the paper reports
the estimate descriptively and draws no registered conclusion from it.
