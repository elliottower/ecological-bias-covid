# Does the ecological discrepancy depend on how sites are defined, and how much of it do patient composition and calendar time account for?

**Status:** FROZEN at commit `a38196b`, tag `registration-s11-s12`, as `ANALYSIS_PROTOCOL.md` was frozen at `97f7094`.
**Date drafted:** 2026-09-24
**Extends:** `ANALYSIS_PROTOCOL.md` (frozen 2026-07-07 at `97f7094`) and `PROTOCOL_CORRECTION_ADDENDUM.md`
**Data:** the 3 January 2022 Mexico snapshot, `220103COVID19MEXICO.csv`, sha256 `25ccc890d190bf66a90aae98507f78f6bfe8b0e9be10767936a0e929512dbc6d`, loaded by `mexico_confirmed_cases.py`: 3,993,464 confirmed-case records, 299,581 deaths.

**Unit.** One row is one confirmed case. The data carry a record identifier, not a person
identifier, and no de-duplication across records is possible, so this document says **records**
and never patients.

## Status vocabulary

Three labels, and every analysis in the paper carries one.

- **Registered** — specified in `ANALYSIS_PROTOCOL.md` before the analysis was run. S1–S10 only.
- **Prospectively specified extension after prior results** — specified and frozen in this file
  before S11 or S12 was run, after the corrected S1–S10 results and the state-level discrepancy
  were known. S11 and S12. Shortened to "prospectively specified extension" after first mention.
- **Post-hoc** — specified after its own result was seen. Nothing here is post-hoc; anything that
  becomes so is labeled where it appears.

This file answers new questions prompted by known results. It does not repair the first
registration, and nothing in it is confirmatory of the results that prompted it.

## Foreknowledge

- The corrected S3 run (2026-09-24) gives, with sites defined as the state of the treating medical
  unit: observed ecological slope +1.311, model-implied slope +0.454, difference +0.857 (bootstrap
  CI over states 0.478 to 1.285; residual-regression CI 0.396 to 1.318), ratio 2.89. The patient
  model carries age 70+, sex and any of nine comorbidities.
- The corrected S9 run gives per-state odds ratios from 7.99 to 18.09, I² 97.4%, and agreement
  among every record-level estimator: pooled 11.26, one-stage random intercept 11.28, state fixed
  effects 11.28, two-stage common effect 11.28, DerSimonian–Laird 11.73, one-stage random slope
  11.74.
- The corrected S2 run records the registered between-state coefficient as not identified, gives a
  within-state odds ratio of 11.27, and, in a correlated-random-effects model carrying no other
  covariates, a state-level association of 1.20 per percentage point of elderly share.
- The CDC results of `ANALYSIS_PROTOCOL.md` are known: the 9-group ecological regression gives
  p = 0.12, the 7-group gives p = 0.02, and the S5 sex control fails at p = 0.99.
- **No result below has been computed.** No grouping other than the treating unit's state has been
  run on these data, and no record-level model other than the three-covariate one has been fitted.

## Randomness, frozen

| Quantity | Value |
|---|---|
| RNG | `numpy.random.default_rng` (PCG64) |
| Partition seed | 20260924 |
| Bootstrap seed | 20260925 |
| Partition replications | 500 per scheme |
| Bootstrap draws | 2,000 per grouping |
| Interval method | percentile (2.5th, 97.5th) |
| Record-level model | refitted inside every bootstrap draw and every partition replication |

Monte Carlo standard errors are reported for every quantity estimated by resampling.

## S11: every defensible site definition, scored under one rule

The same snapshot admits several definitions of a site. All are listed before any is scored, and
every one is reported whatever it gives. **No definition is promoted on the basis of its result.**

| Definition | Field | Sites expected |
|---|---|---|
| State of the treating medical unit | `ENTIDAD_UM` | 32 |
| State of residence | `ENTIDAD_RES` | 32 |
| Municipality of residence | `ENTIDAD_RES` + `MUNICIPIO_RES` | ~2,400 before the size rule |
| Health-care sector | `SECTOR` | ~13 |
| Composition-preserving random partition | assigned at random under quotas | 32, 500 replications |
| Size-matched unrestricted random partition | assigned at random under size quotas only | 32, 500 replications |

For each definition the analysis computes what the corrected S3 computes: the observed ecological
slope of site mortality on site elderly share, the slope implied by the record-level model
averaged over each site's own records, their difference, and the ratio.

**Primary quantity: the slope difference.** The ratio is secondary and is reported only where the
registered resampling distribution of the model-implied slope excludes zero and yields a bounded
95% interval. Otherwise it is recorded as unstable and carries no interpretation.

**Specification.** Unweighted ordinary least squares across sites is primary; its estimand is the
relationship across sites treated as equal units. A binomial regression on site deaths and
survivors, and a population-weighted least-squares fit, are reported beside it for every
definition; their estimand is weighted toward the largest sites.

### The two random partitions

These are outcome-blind, composition-preserving regrouping experiments, not formal negative
controls: nothing here guarantees the absence of a causal relation or a matched bias structure.

**Composition-preserving partition.** For each of 500 replications:

1. Copy each treating-unit state's observed record count and elderly count, giving 32 pairs
   (n_s, e_s).
2. Permute all elderly records without replacement and allocate exactly e_s to synthetic group s.
3. Permute all non-elderly records without replacement and allocate exactly n_s − e_s to group s.
4. Use nothing else in the assignment: not death, sex, comorbidity, residence, municipality,
   sector, dates, or the original state.
5. Fit the record-level model and compute observed and model-implied group mortality under the
   synthetic assignment.
6. Save the seed, the quota table, the realized checks and the per-replication estimates.

This reproduces the exact empirical support of site size and elderly share while severing the link
between geography and outcome except through individual attributes. Because the randomization is
constrained, the constraint is part of the analysis and is reported with the result.

**Size-matched unrestricted partition.** The same 32 group sizes, no elderly quota. It isolates
the collapse of exposure variation without confounding it with unequal group sizes.

### Cohort

Sites with fewer than 1,000 records are excluded, as in the primary analysis, so the retained
cohort differs between definitions. The paper does not claim these analyses use the same records.
Each definition reports: valid sites, retained records, retained deaths, median site size, the
range and standard deviation of elderly share, the states represented, and every record excluded
by the size rule or by an invalid or missing code.

For municipalities the size rule is registered at 1,000 records as primary, with 250 and 500 as
sensitivity thresholds, and with an all-valid-municipality grouped-binomial analysis reported
beside them. Each threshold reports how many municipalities, states, records and deaths it
retains, and which states disappear.

### Uncertainty

State and sector definitions use a site bootstrap. The municipality definition uses a bootstrap
clustered on state: each draw samples 32 states with replacement, keeps every eligible
municipality within each sampled state, gives duplicated clusters distinct identifiers, refits the
record-level model, and recomputes both the municipality discrepancy and the state discrepancy on
the same draw.

### Handling

Sites with no elderly records or no deaths, records with invalid residence, municipality or sector
codes, and cells too sparse to fit are counted and reported, never dropped silently. A replication
or draw whose model fails to converge is recorded, excluded from the summary, and reported as a
count; if more than 1% of replications fail, the analysis is reported as unstable.

## S12: what the discrepancy is made of

The record-level model is enriched in six prespecified steps, with sites defined as the treating
unit's state.

1. age 70+ alone
2. age 70+, sex, any of nine comorbidities *(the model behind the known +0.857)*
3. age 70+, sex, the nine comorbidities entered separately
4. age in ten-year bands, sex, the nine comorbidities separately
5. age as a restricted cubic spline with five knots at the 5th, 27.5th, 50th, 72.5th and 95th
   percentiles of the age distribution among confirmed cases, sex, the nine comorbidities
   separately
6. model 5 plus calendar month of symptom onset as fixed effects

Model 5 is the **most flexible prespecified composition model**; it is not assumed to fit best.
Model 6 adds time, which indexes treatment, vaccination, variant, testing and ascertainment as
well as composition, and is therefore reported as a separate contrast rather than folded into the
composition ladder.

- **Composition contrast: model 2 against model 5.**
- **Temporal contrast: model 5 against model 6.**

Each step reports the model-implied slope, the observed-minus-implied difference, the mean
absolute calibration error across states, and the change from the previous step. No monotonic
ordering is required or predicted.

The correlated-random-effects within-between model of `PROTOCOL_CORRECTION_ADDENDUM.md` §3 is
refitted with the model-5 covariates, which the corrected S2 model does not carry.

**Calendar time.** The date field is `FECHA_SINTOMAS`, the date of symptom onset. Valid dates run
from 2020-01-01 to 2022-01-03; records outside that range or unparseable are counted and excluded
from models 5 and 6 alike, so the two are fitted on identical records. Month is coded as
year-month with the earliest valid month as reference. No age-by-time interaction is fitted.
If more than 5% of records have an invalid onset date, admission date `FECHA_INGRESO` is used
instead and the substitution is reported.

**Outcome opportunity.** Records with symptom onset within 28 days of the snapshot have had less
time for a death to be recorded. A prespecified sensitivity analysis excludes them, for every step
of the ladder. Calendar adjustment does not repair unequal follow-up, and the paper does not claim
it does.

**Age validity.** 24 records carry an age above 120. The primary analysis keeps them, as the
primary analysis did; a sensitivity analysis excludes ages above 110.

**Unknown comorbidity.** 7,080 records have all nine flags unknown and fall in the reference
group, as in the primary analysis. The per-state rate of unknown flags is reported, and a
sensitivity analysis adds a state-level unknown-rate covariate.

## Hypotheses

**H1.** Under composition-preserving random partition the slope difference is close to zero: the
discrepancy across real sites comes from sites differing in ways the record-level model does not
carry, not from aggregation itself.

**H2.** Grouping records by state of residence rather than by the state that treated them leaves a
positive slope difference.

**H3.** The municipality discrepancy differs from the state discrepancy computed on the same
records. No direction is predicted: smaller areas mix fewer kinds of record, which shrinks the
discrepancy, and sort records more strongly by context, which widens it.

**H4.** The slope difference falls from model 2 to model 5 and remains positive: a more flexible
composition model accounts for part of the discrepancy and not all of it.

**H5.** The slope difference falls further from model 5 to model 6 and remains positive: calendar
time accounts for part of what composition does not, and not all of it.

**H6.** Under the model-5 covariates, a state's elderly share retains an association with the
death of an individual record.

H1 carries the design. Without it, a positive difference under any real grouping is consistent
with aggregation arithmetic alone, and the comparison across definitions says nothing about how
sites differ.

The sector grouping carries **no hypothesis**. With about 13 heterogeneous categories it is
reported descriptively: point estimate, interval, leave-one-sector-out range, site sizes and
leverage.

## Inference criteria

| Hypothesis | Holds when |
|---|---|
| H1 | the randomization distribution of the slope difference across 500 composition-preserving partitions has its 97.5th percentile below 0.171, that is below 20% of the +0.857 observed across treating states |
| H2 | the residence-state slope difference has a bootstrap CI excluding 0 |
| H3 | the bootstrap CI of the paired difference (municipality discrepancy minus state discrepancy, both computed on the municipality-retained records within the same draw) excludes 0 |
| H4 | the model-5 slope difference is smaller than the model-2 difference and its bootstrap CI excludes 0 |
| H5 | the model-6 slope difference is smaller than the model-5 difference and its bootstrap CI excludes 0 |
| H6 | the per-standard-deviation state-level odds ratio under model 5 has a CI excluding 1 |

The H1 margin is 20% of the discrepancy already measured across treating states, not a round
number chosen for its own sake: a regrouping that leaves a fifth of the observed discrepancy has
not reproduced it. Beside the criterion, H1 reports the mean and median synthetic difference, the
2.5th and 97.5th percentiles of the randomization distribution, Monte Carlo standard errors, and
the attenuation 1 − |median synthetic difference| / 0.857.

**All hypotheses are void if the loader's checks fail**, that is if the snapshot no longer
reproduces 3,993,464 confirmed-case records, 299,581 deaths and the primary analysis's per-site
shares.

**H3 is void if fewer than 100 municipalities retain 1,000 or more records at the primary
threshold.**

**Any model that fails to converge voids the hypotheses that depend on it.** A failure is
reported, not worked around.

## What would strengthen aggregate sharing

If H1 fails, so that composition-preserving partition also produces a discrepancy, the effect is
arithmetic rather than compositional and the paper's practical advice loses its point. If the
difference reaches zero at model 5 or model 6, so that a flexible model closes the gap, the honest
conclusion is that a network sharing enough covariate detail can reconstruct the record-level
relationship from aggregates, which is a finding in favor of aggregate sharing and will be
reported as such.

## Maximum claim under this registration

The paper may claim: that the discrepancy between the observed ecological slope and the slope a
record-level model implies depends on how sites are defined, with a measured value for each of six
definitions on one snapshot; that a partition preserving site size and elderly composition but
breaking geography does or does not produce it; and that more flexible composition modeling and
calendar-time adjustment each account for a measured share of it, with the residual state-level
association reported. It may not claim that the pattern holds in another country, another disease,
or for any outcome other than recorded death; that the residual reflects any particular mechanism,
since context, ascertainment, coding, follow-up and misspecification are not separated here; that
any of these definitions is how a specific federated network draws its sites; or that the
regrouping experiments establish that geography causes the discrepancy.

## Loader changes this registration requires

`mexico_confirmed_cases.py` currently reads the treating unit's state only. Before S11 and S12 run
it must also read `ENTIDAD_RES`, `MUNICIPIO_RES`, `SECTOR` and `FECHA_SINTOMAS`, and its audit must
record, for each: every observed code, the counts of missing and sentinel values (including
municipality code 999, "no especificado"), the date range and the count of invalid dates, and the
number of records excluded. It must also report whether the record identifier `ID_REGISTRO` is
unique, so the paper can say what one row is. The per-site invariant check against the primary
analysis stays as it is.

## Analyses not registered here

The registered S1–S10 analyses stand as in `ANALYSIS_PROTOCOL.md`, corrected as recorded in
`PROTOCOL_CORRECTION_ADDENDUM.md`. Reporting choices for the manuscript — which figures, which
sections, the treatment of the CDC and 4CE material — are not analyses and are not registered.

## Commit SHA

**a38196b15db16a4d497e5aa4d9e0a07f1c3834b3**
Frozen and pushed 2026-09-24, tagged `registration-s11-s12`.
Repository: github.com/elliottower/ecological-bias-covid (public)
