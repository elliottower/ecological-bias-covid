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

## 8. What H4 and H5 mean (2026-09-26)

The frozen criteria read "the model-5 slope difference is smaller than the model-2 difference and
its bootstrap CI excludes 0", and the same shape for model 6 against model 5. "Its" can be read as
the remaining difference or as the reduction. The code implements the first: the remaining
discrepancy must stay above zero, and the reduction is reported as a point estimate.

That reading is stated here rather than resolved by running both. Testing a hypothesis under two
readings of its own wording is two attempts at one criterion, and the registration fixed one
criterion. The reduction from model 2 to model 5 is +0.201 and from model 5 to model 6 is +0.062;
neither is a registered test and neither carries a p-value or an interval in the paper.

Per-draw bootstrap values are archived so that a reader who prefers the other reading can compute
it, but the paper reports the registered criterion under the reading given here.

## 9. Results files are archived, not overwritten (2026-09-26)

Every result file is now written through `paths.write_result`, which moves any existing file into
`results/superseded/` under the timestamp that file carries before writing the new one. Run logs
carry the run name.

One file predates the rule and is gone: the first S11 run of 2026-09-24, whose paired municipality
contrast used the grouping corrected in §7. Its printed values are transcribed in
`results/superseded/s11_site_definitions_2026-09-24_first_run_NOTE.md` and are marked there as
reconstructed from the console output rather than recovered from the file.

## 10. Sentinel codes and the catalogue that defines them (2026-09-26)

The registration names code 999 for an unknown municipality. The official catalogue
(`diccionario_datos_abiertos.zip`, Secretaría de Salud, catalogue dated 2024-07-08, archived
beside the snapshot) gives three codes in that family for `MUNICIPIO_RES` — 997 `NO APLICA`,
998 `SE IGNORA`, 999 `NO ESPECIFICADO` — and real municipality codes run from 1 to 570 within a
state. All three are treated as no municipality, which widens the registered exclusion by two
codes; the records affected are 2,874 under 999 and none under 997 or 998, out of 3,993,464
confirmed cases. A municipality is keyed as state × 1000 + code, since `MUNICIPIO_RES` numbers
municipalities within a state rather than nationally.

`SECTOR` carries one such code, 99 `NO ESPECIFICADO`, on 20 records; those records carry no
sector and are outside the sector site definition. Fifteen sector codes appear in the snapshot.

The loader recomputes every count above into `results/mexico_loader_audit.json` at run time,
including the size of the largest pseudo-site the sentinel codes would have formed had they been
treated as a municipality.

## 11. The contextual model is fitted in Python (2026-09-26)

**The decisions this section used to carry now live in
`IMPLEMENTATION_AMENDMENT_2026-09-26.md`**, committed and tagged before S11 and S12 are run.
That file fixes the estimator, the interval, the boundary rule, the numerical acceptance
gates and the validity domain; this section records the history that led to them.

H6's design — treating-unit state, age, sex and the nine comorbidity flags — collapses to
161,655 cells, since most comorbidity combinations never occur. `lme4::glmer` segfaults
inside `pwrssUpdate` on that table, and the crash survived upgrading `Matrix` from 1.3-4 to
1.5.1 and reinstalling `lme4`. `r/contextual_model.R` is deleted; `glmm.py` fits H6 by
adaptive Gauss-Hermite quadrature, and `r/glmer_reference.R` remains only to fit comparison
models for validation.

**Validation.** `validate_glmm.py` writes seven checks to `results/glmm_validation.json`:
S9's one-stage model refitted and compared with the `lme4` fit; a simulated table of H6's
exact shape fitted by both implementations, or `glmer`'s failure on it recorded; the same
sixteen-term design at a size `glmer` should finish; the fit at 7, 11, 15, 21 and 25
quadrature nodes; the gradient at the optimum and the Hessian's agreement across step sizes;
the profile interval against the Wald interval; and repeated-sample coverage of both
intervals at 12, 32 and 120 groups and at variance components of 0.35 and zero, each with a
binomial interval on the coverage estimate.

**Invariant tests.** `tests/test_analysis_invariants.py` checks what the pipeline assumes
rather than what it prints: that the cell-collapsed binomial fit reproduces a record-level
logistic fit to 10⁻⁸; that fitting the record-level model once is exactly a refit after every
regrouping, which is why the partition does not refit it per replication; that the
composition-preserving partition reproduces every state's size and elderly count exactly
while the unrestricted partition collapses the between-group elderly spread; that the
matrix-based S12 draw and the matrix-based paired municipality contrast equal brute-force
refits on duplicated states; that all six ladder models receive the same state multiplicities
in draw *b*; that sentinel codes and codes outside the catalogue carry no site, and that a
municipality is keyed within its state; that an invalid onset date leaves the calendar month
missing so models 5 and 6 share a cohort; that a grouping with no exposure variation, a
failed fit and a raised exception are each counted as a failed draw rather than aborting a
run; that too few valid draws suppress an interval entirely; that `write_result` and
`write_table` archive rather than overwrite; that the quadrature objective returns the same
value at the same parameters whatever the optimizer evaluated before it; that the fit is
stable from 7 to 25 nodes and agrees across its three variance starts; that a profile which
cannot bracket the cutoff reports no interval; and that a failed numerical gate leaves H6
`evaluable: false` with the estimate still reported.

## 12. What changes in the numbers (2026-09-26)

Every entry below is a methodological correction rather than a preference, and each one moves
a number relative to the September 24 outputs kept in `results/superseded/`.

| Change | What moves |
|---|---|
| Named `SeedSequence` substreams, one per analysis | every bootstrap interval and Monte Carlo error |
| Separate streams per municipality threshold, and a new stream for the unthresholded analysis | each threshold's interval |
| 2,874 records with no municipality and 20 with no sector excluded | the municipality and sector cohorts, and their point estimates |
| Model-implied rates projected with `var_weights` | the binomial-scale implied slope and difference |
| A state-clustered bootstrap on the binomial scale for every valid municipality | an interval that did not exist |
| H6 on the centered model-5 spline, fitted by quadrature | the contextual estimate, which had no reported value before |
| Profile rather than Wald interval for the contextual coefficient | H6's interval, at an unchanged point estimate |
| Unknown-rate covariate unrounded, with a clustered interval | the nuisance coefficient and its interval |
| Leave-one-out deletion change per sector, and the all-municipality analysis | results that did not exist before |

Three further inputs can move a number without being visible as a change of method, so each
result file now records them: the package and interpreter versions, the hash of every
analysis source file, and the optimizer settings — quadrature nodes, variance starts,
finite-difference steps, boundary tolerance and gate thresholds.

The snapshot contains no residence state outside 1–32, no municipality code between 571 and
996, and no unparseable or out-of-range onset date, so enforcing the validity domain changes
no number in this dataset; it makes the cohort a property of the loader rather than of the
file.
