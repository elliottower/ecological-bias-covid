# Novelty search, 2026-09-30

What this paper may claim, and what it may not. Searched across ecological-bias theory,
aggregate-data and hierarchical related regression, hybrid ecological/individual designs,
`ecoreg`, the modifiable areal unit problem and scale sensitivity, distributed and
federated regression (DataSHIELD, OHDSI-style networks, JAMIA), and Mexican COVID-19
open-data studies. These are search-based nulls, not proof of absence, so the manuscript
says "we identified no prior study" and never claims priority outright.

## Verdict on each result

| Result | Verdict | Closest prior work |
|---|---|---|
| Coefficient pooling agrees with one-stage record-level modeling | **Established** | @wolfson2010datashield; @elemam2013secure |
| Observed against model-implied ecological slope on the same units | **Partly established** | @prentice1995aggregate; @wakefield2001statistical; @haneuse2011designs |
| Composition-preserving pseudo-site partition | **No precedent found** | — |
| Finer geography shrinks the discrepancy | **Established** | @best2001ecological; @lancaster2006reducing; @elliott2008spatial |
| A model ladder decomposing the discrepancy | **Partly established** | @lancaster2002deprivation; @lasserre2000biases |

The two papers a skeptical reviewer is most likely to raise are @lancaster2006reducing,
which compares ecological, stratified, aggregate-data and individual analyses on large
census data at two spatial scales, and @jackson2008hierarchical, which derives aggregate
likelihoods from an individual model and shares coefficients across both.

## What the paper must not say

- That it is the first to show ecological and individual associations differ.
- That it introduces model-implied aggregate outcomes or slopes; averaging a fitted
  individual model over each area's covariate distribution is standard in aggregate-data
  and hierarchical related regression.
- That smaller units reducing ecological bias is a new finding.
- That distributed coefficients reproducing a pooled regression is a new finding.
- That the ladder decomposes the *causes* of the discrepancy; the increments are
  descriptive and depend on the order in which terms enter.
- That the pseudo-site partition is a negative control, or that it proves site context
  causes what remains.

## One distinction that constrains a claim

"Coefficient pooling" covers different operations. Distributed score and information
aggregation targets the coefficient of a pooled-data model and can equal it;
inverse-variance meta-analysis of separately fitted coefficients targets a
precision-weighted common-effect summary and need not equal a pooled maximum-likelihood
estimate in a nonlinear model. This paper's two-stage estimators are the latter, so the
agreement observed here is an empirical result on this dataset, not a preservation
theorem, and the manuscript says so.

## What the paper may claim

The contribution is a diagnostic design applied to one complete record set, not five
separate inventions:

- No prior study identified compares the ecological slope through observed site outcomes
  with the slope through outcomes predicted for the identical records and sites by a fitted
  record-level model, while contrasting coefficient-sharing with marginal-rate-sharing.
- No prior study identified constructs pseudo-sites preserving every real site's size and
  focal exposure count exactly, randomizing record identities without using outcomes, and
  recomputes the discrepancy on them. This is the strongest of the five.
- Extending scale sensitivity beyond nested residential geographies to matched
  residence-based and care-delivery-based site definitions is distinctive, though the
  general dependence on areal units is not.

## Mexican COVID-19 open data

The dataset and the age–mortality topic are not novel: @antoniovilla2021unequal models
mortality in older Mexican adults from the individual records with municipality random
effects, and state-level ecological analyses of the same source exist. No located paper
regresses state case mortality on the share of records aged 70 or older and compares it
with a slope induced from a fitted individual model, nor uses the pseudo-site partition,
the shared-object contrast, the matched municipality comparison, or the ladder.
