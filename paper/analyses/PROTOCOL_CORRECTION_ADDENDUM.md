# Correction addendum to the frozen analysis protocol

**Date:** 2026-09-22
**Applies to:** `ANALYSIS_PROTOCOL.md`, frozen at commit `97f7094` on 2026-07-07

`ANALYSIS_PROTOCOL.md` is not edited. This file records what the first implementation of S2, S3
and S9 did, what was corrected, and which results are registered and which exploratory.

## 1. The first S2, S3 and S9 runs used a different dataset

The protocol specifies S2 and S9 on "Mexico individual-level data (3.99M patients, 32 states)"
and S3 against the observed ecological slope of +1.31. The scripts run on 2026-07-07 read
`data/mexico_covid/COVID19MEXICO.csv`, a smaller extract, with no confirmed-case filter and with
the state of residence (`ENTIDAD_RES`) as the site.

| | primary analysis | 2026-07-07 S2, S3, S9 runs |
|---|---|---|
| file | `220103COVID19MEXICO.csv` | `COVID19MEXICO.csv` |
| cases | confirmed only (CLASIFICACION_FINAL 1–3) | unfiltered |
| site | `ENTIDAD_UM`, treating medical unit | `ENTIDAD_RES`, residence |
| records | 3,993,464 | 229,125 |
| ecological slope | +1.31 (R² 0.50) | +0.23 (R² 0.18) |

Found on 2026-09-22 while auditing the manuscript against the result files: the supplement's
Mexico numbers could not be reconciled with the main text's. The superseded outputs are kept in
`results/superseded_229k_extract/`.

**Correction.** S2, S3 and S9 now share `mexico_confirmed_cases.py`, which applies the primary
analysis's definitions to the 3 January 2022 snapshot
(`datos_abiertos_covid19_03.01.2022.zip`, sha256 `63993520…f272dc`; CSV sha256 `25ccc890…2dbc6d`)
and refuses to return data unless the file hash matches and every state's size, mortality rate,
elderly, male and comorbidity share reproduce the primary analysis.

## 2. S3 coding errors

- Sex was coded `SEXO == 1`, which is female in this dataset. The superseded output's "male
  OR = 0.61" is the odds ratio for women.
- The comorbidity indicator used eight flags; the primary analysis uses nine, adding smoking.
- State mortality was predicted from the product of three marginal proportions, which treats
  age, sex and comorbidity as independent within a state. The protocol says to integrate over the
  state's observed covariate distribution, so the corrected script averages the model's predicted
  probability over each state's patients. The product-of-marginals version is reported as a
  sensitivity analysis.

**Naming.** "Expected ecological slope" becomes "model-implied ecological slope", and the
observed-minus-implied gap is reported as a discrepancy rather than as aggregation bias: state
context, omitted patient variables, ascertainment, coding and misspecification are all in it.
The slope difference with a bootstrap interval over states is primary; the ratio is secondary.

## 3. S2 is not estimable as registered

The registered model carries the state mean of the age indicator beside 31 state dummies. A state
mean is constant within a state, so it lies in the span of the intercept and the dummies, and its
coefficient is not identified. The 2026-07-07 run fitted it anyway and reported a between-state
standard error of about 1.3 × 10⁶; the registered CI-overlap criterion fired on that.

**Correction.** The script now establishes the collinearity before fitting and records the
registered between-state coefficient as non-estimable. The within-state coefficient is identified
and is reported. A correlated-random-effects within-between model, fitted with `lme4::glmer` on
site × age × sex × comorbidity cells, is reported as an **exploratory correction**, not as the
registered estimator.

## 4. S9 did not fit the registered one-stage model

The protocol registers `death ~ age_70plus + (1|state)`. Both the 2026-07-07 script and its first
correction fitted state fixed effects instead, which is a different model with different
assumptions.

**Correction.** The registered random-intercept model is fitted with `lme4::glmer` on
site × age cells. Estimators are aligned before they are compared: common-effect two-stage
pooling beside the random-intercept model, DerSimonian–Laird pooling beside a random-slope model.
Differences are reported on the log-odds-ratio scale. State fixed effects are kept as a
sensitivity analysis, and REML with Hartung–Knapp is added as an exploratory alternative to the
registered DerSimonian–Laird pooling.

**The registered falsification criterion** compares two-stage pooling with "the pooled
individual-level OR". That comparator ignores state, so it carries between-state confounding and
the noncollapsibility of the odds ratio. The criterion is applied and reported as registered; its
registered reading, that a gap shows "heterogeneity distorts all pooling", does not follow, and
the paper does not use it.

## 5. Registered interpretations that do not hold

- **S4.** Intersecting the per-site Duncan–Davis bounds assumes one elderly mortality rate common
  to every site. A pooled bound needs the bounded elderly death counts summed across sites over
  the total elderly patients.
- **S6.** Randomly partitioning millions of patients produces groups with nearly identical age
  distributions. Its low power shows that random grouping removes between-group exposure
  variation, not that ecological regression is inherently underpowered.
- **S7.** The E-value is computed for a change in elderly share from 0 to 1, far outside the
  observed 0.01–0.43, on an ecological moderator that cannot separate composition from context.
  The registered result is preserved and marked uninterpretable; it carries no weight in the
  paper.

## 6. Status of every affected result

| Result | Status |
|---|---|
| Loader audit, `results/mexico_loader_audit.json` | provenance record |
| S2 registered between-state coefficient | not estimable; no value reported |
| S2 within-state coefficient | registered, corrected data |
| S2 within-between random-intercept model | exploratory correction |
| S3 observed and model-implied slopes, difference, bootstrap | registered analysis on corrected data, with added uncertainty |
| S3 product-of-marginals slope | sensitivity analysis |
| S9 two-stage DerSimonian–Laird, one-stage random intercept | registered, corrected data and corrected model |
| S9 random slope, common-effect pooling, REML with Hartung–Knapp, state fixed effects | exploratory |
| S9 registered falsification criterion | applied as registered; its registered reading is not used |
| Superseded 2026-07-07 outputs | preserved in `results/superseded_229k_extract/` |

---

## 7. Implementation notes for S11 and S12 (2026-09-26)

The registration was frozen at commit `a38196b`, tag `registration-s11-s12`, and pushed before
either extension was run. Four implementation matters are recorded here because they are
decisions the registration did not fix.

**The H3 grouping, corrected before the reported run.** A first implementation keyed the paired
contrast on treating state and municipality together, which produced 4,498 cells rather than 386
municipalities. Municipalities nest in the state of residence, so the reported analysis groups by
municipality alone, compares it with the state of residence on the same records, and clusters the
bootstrap on state of residence. The earlier implementation's paired number is not reported.

**The spline basis is centered.** `patsy`'s natural cubic spline spans the constant function, so
an uncentered basis is collinear with the intercept and the fit does not converge. The reported
models use `constraints='center'`, which changes the parameterization and not the fitted values.
Five knots sit at the registered percentiles of age among confirmed cases: 16, 29, 39, 51 and 73,
with the outer two as boundary knots.

**The bootstraps run on cell matrices.** Resampling sites changes only the per-cell counts, so the
design matrix is built once and the model is refitted on reweighted counts in every draw, which is
what the registration requires of each draw. Rebuilding record-level frames per draw would not
have finished. Each model in S12 is built, bootstrapped and released before the next is built, and
every model sees the same sequence of state multiplicities.

**Counts that differ from the registration's expectations.** Eleven health-care sectors met the
1,000-record rule, not the approximately thirteen anticipated; the snapshot carries fifteen sector
codes, four of them with fewer than 1,000 records. 2,453 distinct municipalities appear, 386 of
which meet the primary size rule. Every record carries a valid onset date inside the registered
range, spanning 24 months from February 2020, so models 5 and 6 were fitted on the full cohort.
`ID_REGISTRO` is unique, so one row is one record; it is not a person identifier, and the paper
says records throughout.
