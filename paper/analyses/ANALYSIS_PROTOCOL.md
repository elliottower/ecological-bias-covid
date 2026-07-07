# Analysis Protocol with Provenance Tagging

**Date:** 2026-07-07
**Author:** Elliot Tower
**Paper:** When Site-Level Averages Mislead: Two Failure Modes of Ecological Bias in Federated COVID-19 Analysis
**Status:** FROZEN — commit SHA below timestamps this protocol before any analysis is run.

## Purpose

All 5 primary analyses in the paper are already complete. This protocol
specifies 10 supplementary analyses designed to close specific reviewer
objections. Each analysis is tagged by provenance:

- **SIMULATION**: Outcome genuinely unknown. Hypothesis and falsification
  criterion are predictive.
- **REANALYSIS**: The underlying data has already been observed and primary
  models fitted. The honest claim is: "we specified this procedure before
  running *this specific* analysis, and we commit to reporting it regardless
  of outcome." We do not claim to have predicted the result.

## Binding commitments

1. **Completeness:** All 10 analyses (S1, S2, S3, S4, S5, S6, S7, S8, S9,
   S10) will be run and reported regardless of outcome, including results
   that contradict or weaken the paper's claims. No analysis may be
   quietly dropped as "not informative."
2. **Falsification criteria are binding.** If a criterion fires, the
   corresponding paper section must be revised to reflect it.
3. **Main-text vs. supplement is frozen now:**
   - **Main text:** S2 (within/between decomposition), S3 (expected
     ecological coefficient), S9 (two-stage vs one-stage IPD)
   - **Supplement:** S1, S4, S5, S6, S7, S8, S10
   This split will not change based on which results are strongest.
4. **RNG seeds for simulations are frozen** in this document.
   Re-rolling with different seeds is not permitted.

---

## S1: Power vs. Bias Decomposition Simulation

**Provenance:** SIMULATION (outcome unknown)
**Script:** `s1_power_simulation.py`
**Closes:** "The CDC null (p=0.12) is just low power at n=9, not bias."
**RNG seed:** 20260707 (numpy legacy RandomState via `np.random.seed`)
**Method:** Simulate 2,000 ecological regressions per condition. For the
"CDC-matched" condition, draw site compositions from the **observed 9 CDC
sites** (exact elderly proportions, comorbidity rates, sex ratios, and
site sizes from `cdc_ecological_fallacy.json`):

| Site                    |         n | elderly | comorbidity |  male |
|-------------------------|----------:|--------:|------------:|------:|
| American Indian/Alas    |   142,780 |  0.0000 |      0.0711 | 0.489 |
| Asian, Non-Hispanic     |   174,062 |  0.0782 |      0.0343 | 0.327 |
| Black, Non-Hispanic     | 1,920,609 |  0.0338 |      0.0765 | 0.266 |
| Hispanic/Latino         | 1,195,959 |  0.0936 |      0.0424 | 0.474 |
| Missing                 |   192,529 |  0.0000 |      0.0047 | 0.529 |
| Multiple/Other, Non-    |   157,859 |  0.0000 |      0.0470 | 0.307 |
| Native Hawaiian/Othe    |    13,548 |  0.0358 |      0.1421 | 0.154 |
| Unknown                 | 2,730,140 |  0.1346 |      0.0371 | 0.646 |
| White, Non-Hispanic     | 6,165,036 |  0.0741 |      0.1110 | 0.587 |

Observed cross-site correlations: rho(elderly, comorbidity) = -0.048,
rho(elderly, sex) = +0.412, rho(comorbidity, sex) = -0.421.

Generate individual-level data within each site from a logistic model:
logit(p_death) = intercept + log(OR_age) * is_elderly + log(OR_comorb) *
has_comorbidity + log(OR_sex) * is_male, using the known individual-level
ORs (age 80+ OR=9.88, comorbidity OR=1.5, male OR=1.7).

Vary the condition:
  (a) **CDC-matched** (n=9): exact CDC site compositions and sizes
  (b) **CDC-resampled** (n=20, 50, 100, 500): resample sites from the
      empirical distribution of CDC site compositions (with replacement),
      preserving the observed compositional variation
  (c) **Independent confounders** (n=9, 20, 50, 100, 500): each site
      draws elderly proportion from the observed CDC marginal distribution
      (resampled with replacement from the 9 observed values), but
      comorbidity and sex proportions are drawn independently from their
      own marginal distributions (also resampled). This preserves the
      range of elderly proportions while breaking the cross-site
      correlations (rho(elderly,comorbidity), rho(elderly,sex)) that
      create confounding. Site sizes are drawn from the CDC size
      distribution. Note: this is NOT "zero variation" (which would make
      the regression degenerate), but rather "variation without correlated
      confounders."

Report power (fraction of simulations where ecological regression
mortality ~ elderly_proportion is significant at alpha=0.05).

**Decision rule:** If power under CDC-matched confounding exceeds 80% at
n<=50 (condition b), the bias claim is wrong and the CDC null is primarily
a power problem. If power stays below 50% at n=500 in condition (b) while
exceeding 80% in condition (c), the null is driven by compositional
confounding, not low power.
**Falsification:** Power > 80% at n<=50 under CDC-matched confounding
would force us to reframe failure mode 1 as a power limitation, not a
bias demonstration. The corresponding section and abstract claim would
be revised.
**Scope of revision:** Note that the observed cross-site correlations
show rho(elderly, comorbidity) = -0.048 (essentially zero), while
rho(elderly, sex) = +0.412 and rho(sex, comorbidity) = -0.421. The
paper's §2.3 currently explains failure mode 1 as elderly proportion
being confounded by comorbidity composition. S1 may reveal that the
actual confounding pathway runs through sex (elderly-sex correlation)
rather than through comorbidity directly. If so, §2.3's mechanism
narrative must be revised to match the simulation's findings. This is
explicitly anticipated and the protocol commits to reporting whichever
mechanism the simulation supports.
**N simulations:** 2,000 per condition (fixed).

## S2: Within/Between Effect Decomposition (Mundlak Model)

**Provenance:** REANALYSIS (Mexico individual data already observed; primary
individual-level logistic already fitted)
**Script:** `s2_mundlak_decomposition.py`
**Closes:** "You assert distortion but never decompose it."
**Placement:** Main text
**Method:** On Mexico individual-level data (3.99M patients, 32 states),
fit a Mundlak fixed-effects approximation: a standard logistic regression
(not a GLMM) with both within and between components as covariates:
  death ~ age_70plus_within + age_70plus_between + state_dummies
where age_70plus_within = individual age indicator minus state mean,
and age_70plus_between = state mean of age indicator. State fixed effects
(31 dummies) absorb unobserved state-level heterogeneity.
Report within-site coefficient, between-site coefficient, and their
difference (= aggregation bias).
**Estimator rationale:** A true GLMM with random intercepts on 3.99M
binary outcomes is computationally fragile (PQL/Laplace may not converge
in statsmodels at this scale). The Mundlak fixed-effects approximation
yields the same within/between decomposition without convergence risk.
This estimator choice is frozen — we will not switch estimators based on
results.
**Procedure commitment:** We report the within and between coefficients
and their ratio regardless of outcome. The gap is the aggregation bias
as a number.
**Falsification:** If the between-site coefficient is statistically
indistinguishable from the within-site coefficient (overlap of 95% CIs),
there is no meaningful aggregation bias in the Mexico data and failure
mode 2 must be softened from "distorts" to "attenuates."

## S3: Expected Ecological Coefficient Under No Bias

**Provenance:** REANALYSIS (Mexico individual data and state-level
demographics already observed)
**Script:** `s3_expected_ecological.py`
**Closes:** "How do you know beta=+1.31 is wrong and not just a valid
different summary?"
**Placement:** Main text
**Method:** For each of Mexico's 32 states, predict site-level mortality
by integrating the fitted individual-level logistic model over the state's
observed covariate distribution. Regress predicted mortality on elderly
proportion. The slope of this "expected ecological regression" is what
beta SHOULD be absent aggregation bias.
**Procedure commitment:** Report the expected slope, the observed slope
(+1.31), and the ratio (attenuation factor).
**Falsification:** If expected and observed ecological slopes agree within
20%, the ecological regression is a valid summary of the individual-level
relationship and failure mode 2 is overstated.

## S4: Ecological-Inference Bounds (Duncan-Davis)

**Provenance:** REANALYSIS (CDC site-level data already observed)
**Script:** `s4_ecological_bounds.py`
**Closes:** "Purely negative paper with no constructive contribution."
**Placement:** Supplement
**Method:** For each CDC pseudo-site, compute Duncan-Davis bounds on the
individual-level **mortality rate** among elderly (80+) patients. The
bounds use only site-level data: total mortality rate m, elderly
proportion p, and non-elderly mortality rate range [0, m/p]. For each
site: m_elderly is in [max(0, (m - (1-p)*1) / p), min(1, m/p)].
Report the intersection of bounds across all 9 sites.
**Prerequisite:** The script must first compute the observed individual-
level elderly (80+) mortality **rate** from `cdc_full.csv` (streaming,
as done for age-group mortality in `make_fig1.py`). The paper currently
reports the elderly effect only as OR=9.88; the falsification comparison
requires the raw rate. This rate is computed by the script, not assumed.
**Procedure commitment:** Report the bounds regardless of width. If wide,
this supports the paper (ecological data is uninformative). If tight, this
is a constructive finding (ecological data bounds the truth).
**Falsification:** If bounds are tight (width < 5 percentage points)
AND the bounded mortality-rate interval for elderly patients is
inconsistent with the observed individual-level elderly mortality rate
(i.e., the true elderly mortality rate falls outside the bounds), either
the bounds method is misapplied or the individual-level analysis has an
error. Either way, something in the paper is wrong. Note: bounds are on
mortality **rates**, not odds ratios; comparison to OR=9.88 requires
converting to the same scale.

## S5: Reduced-Confounding Positive Control on CDC and Mexico

**Provenance:** REANALYSIS (CDC/Mexico data already observed)
**Script:** `s5_positive_control.py`
**Closes:** "How do we know your method flags real problems, not noise?"
**Placement:** Supplement
**Method:** Run ecological regression with sex ratio (proportion male)
as predictor of site-level mortality, on both CDC and Mexico. Sex has a
known individual-level association (male OR=1.7) but sex composition
varies DIFFERENTLY across sites than age composition. This is a
**reduced-confounding positive control** (not a negative control): sex
has a real effect, but its cross-site confounding structure differs from
age, so comparing their ecological performance isolates the role of
compositional confounding.

Observed cross-site correlations in CDC: rho(elderly, comorbidity) =
-0.048, rho(sex, comorbidity) = -0.421. If sex-mortality ecological
regression succeeds where age-mortality fails, the failure is specific
to age's confounding structure, not to ecological regression in general.

**Procedure commitment:** Report slope, p-value, and R-squared for
sex-mortality ecological regression on both datasets, regardless of
whether it is significant.
**Falsification:** If sex-mortality ecological regression ALSO fails
(non-significant on CDC), the pseudo-site construction may be too noisy
for ANY ecological regression, not specifically for age. This would narrow
the paper's claims to "pseudo-sites are bad" rather than "ecological
regression is biased by age-specific confounding."

## S6: Alternative Pseudo-Site Definitions

**Provenance:** REANALYSIS (CDC data already observed)
**Script:** `s6_alternative_pseudosites.py`
**Closes:** "It's an artifact of using race/ethnicity as pseudo-sites."
**Placement:** Supplement
**Method:** Re-partition CDC data into pseudo-sites using:
  (a) Random partitions into 9 groups (500 replications)
  (b) Random partitions into 50 groups (500 replications)
Run ecological regression (mortality ~ elderly proportion) on each.
Report power (fraction significant at alpha=0.05) with Wilson score 95%
confidence interval on the power estimate.
**Procedure commitment:** Report power and its CI for all grouping
strategies regardless of outcome.
**Falsification:** If random partitions into 9 groups consistently detect
the age-mortality relationship (power > 80%, lower bound of Wilson CI
> 75%), the failure is specific to race/ethnicity grouping (which
maximizes compositional confounding), not a general property of
ecological regression at n=9. This would narrow failure mode 1 to
"groupings that maximize confounding" rather than "ecological regression
in general."
**N replications:** 500 per condition (SE on power ~2% at p=0.5).

## S7: E-Value Sensitivity for 4CE Moderators

**Provenance:** REANALYSIS (4CE meta-regression results already observed)
**Script:** `s7_evalue_sensitivity.py`
**Closes:** "4CE is aggregate-only, so it's all confounded."
**Placement:** Supplement
**Method:** Compute E-values for the elderly-proportion moderator
(beta=+14.2, SE=0.83) in the 4CE meta-regression, using the VanderWeele
& Ding (2017) formula.
**Procedure commitment:** Report the E-value as a descriptive sensitivity
quantity. We make no directional prediction — the E-value is what it is.
There is no hypothesis to test; this is a sensitivity analysis.
**Interpretation guide:** E-value > 3 means substantial unmeasured
confounding is needed to explain the moderator. E-value < 1.5 means
modest confounding suffices, substantially weakening the 4CE claims.

## S8: Leave-One-Site-Out Stability

**Provenance:** REANALYSIS (4CE PMI data already computed)
**Script:** `s8_leave_one_out.py`
**Closes:** "Your 350-fold divergence is driven by one bad site."
**Placement:** Supplement
**Method:** For each of the ~20 4CE sites, drop it and recompute
fixed-effects PMI and random-effects PMI. Report the range of FE PMIs
and RE/FE divergence ratios.
**Procedure commitment:** Report all leave-one-out results. If one site
drives the divergence, name it.
**Falsification:** If dropping one specific site reduces the RE/FE
divergence below 10-fold, the 350-fold artifact IS site-specific and
the paper must report it as such rather than as a general property of
the data.

## S9: Two-Stage vs. One-Stage IPD on Mexico

**Provenance:** REANALYSIS (Mexico individual data already observed;
both the ecological and individual-level models already fitted)
**Script:** `s9_two_stage_vs_one_stage.py`
**Closes:** "Would proper IPD methods have fixed this?"
**Placement:** Main text
**Method:** On Mexico's 32 states:
  (a) Ecological: per-state mortality rate regressed on elderly proportion
      (existing analysis, beta=+1.31)
  (b) Two-stage IPD: fit per-state logistic (death ~ age_70plus), pool
      ORs via random-effects meta-analysis
  (c) One-stage IPD: fit death ~ age_70plus + (1|state) on all 3.99M
      records
Report all three estimates side by side.
**Procedure commitment:** Report all three estimates and their CIs,
regardless of whether they agree or disagree.
**Falsification:** If two-stage IPD also produces a distorted estimate
(OR differing from the pooled individual-level OR by more than 50%),
the problem is heterogeneity across states, not ecological aggregation
per se. This would force reframing failure mode 2 from "ecological
regression distorts" to "heterogeneity distorts all pooling."

## S10: Consistency-Test Calibration Under Known Null

**Provenance:** SIMULATION (outcome unknown)
**Script:** `s10_consistency_calibration.py`
**Closes:** "Is the consistency test even calibrated?"
**RNG seed:** 20260707 (numpy legacy RandomState via `np.random.seed`)
**Method:** Generate 2,000 simulated datasets with heterogeneous coverage
matching the 4CE structure in both dimensionality and coverage pattern.
The 4CE consistency test used 30 Elixhauser comorbidity categories,
yielding 30*(30-1)/2 = 435 pairwise correlations. Each simulated dataset
has 20 sites. Each site draws **binary** comorbidity indicators (0/1 for
each of 30 variables) from the same Bernoulli probability vector (drawn
once per simulation from Uniform(0.01, 0.30) for each variable), then
computes the 435-element pairwise correlation vector. Coverage is matched
to 4CE: 10 sites report all 435 pairs, 10 sites report a random subset
of 100-300 pairs (simulating partial Elixhauser coverage). Compute the
consistency statistic (mean sum of squared pairwise differences,
normalized by shared coverage) on each simulated dataset. Report the
null distribution.
**Decision rule:** False-positive rate at z=25 must be <= 0.001
(conservatively below nominal alpha). False-positive rate at z=2 must
be <= 0.06 (at nominal alpha=0.05, allowing for simulation noise).
**Falsification:** If FPR at z=25 exceeds 1%, the consistency test is
miscalibrated and the z=25 finding in the paper is unreliable. The
consistency test section would need to be revised or removed, and z=25
could not be reported as evidence of cross-site inconsistency.
**N simulations:** 2,000 (fixed).

---

## Software and Environment (pinned)

- Python 3.13
- numpy 2.5.0 (legacy RandomState RNG via `np.random.seed`)
- scipy 1.18.0
- statsmodels 0.14.6
- All scripts run on local machine (Intel Mac, uv environment)
- Mexico individual-level data: ~4M records, fits in memory
- CDC individual-level data: ~106M records, streamed where needed

## Commit SHA

**97f70946176a2ccad3b7ac116c127efd71827942**
Frozen: 2026-07-07
Repository: github.com/elliottower/ecological-bias-covid (private)
