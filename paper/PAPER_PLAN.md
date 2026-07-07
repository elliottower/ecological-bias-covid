# Paper Plan: Geometric Validity Testing for Multi-Site Causal Discovery in EHR Data

## The nugget

Scalar heterogeneity tests (Cochran's Q, I^2) and per-edge CI tests miss
structural inconsistencies that geometric methods detect. Sheaf cohomology
on comorbidity correlations finds massive collective inconsistency (z = 25)
invisible to pairwise Q tests (0/1305 significant). The Lie bracket norm
on the effect surface identifies a specific age-by-mortality interaction
(p = 0.021-0.037) that amplifies over time. These geometric tests
complement existing causal discovery tools (CD-NOD, RCIT) and provide
cross-site validity diagnostics that scalar statistics cannot.

## The contrast (motivating observation)

On 4CE COVID neurological data (21 sites, 6 countries):

| Method | Tests | Result |
|---|---|---|
| Cochran's Q per pair | Individual correlation heterogeneity | 0/1305 significant |
| Sheaf H^1 (comorbidity) | Collective correlation consistency | z = 25, p < 0.0001 |
| Sheaf H^1 (ICD counts) | Outcome count consistency | Null (p = 0.74) |
| CD-NOD | Causal graph from cross-site shifts | neuro-severity, neuro-age (5 edges) |
| RCIT on real data | DAG conditional independence | 42% consistency |
| Lie bracket | Effect surface non-commutativity | age x mortality, strengthening over time |
| Meta-analysis (IV/DL/IF) | Pooled effect + heterogeneity | I^2 = 99.8%, methods diverge 350x |

The contrast: the sheaf sees something when coverage is heterogeneous (comorbidity)
and sees nothing when coverage is homogeneous (ICD counts). This is a principled
distinction, not a failure -- it reveals when geometric methods earn their complexity.

## Companion to the iMSMS paper

The iMSMS (Xia) paper applies the same toolkit to paired microbiome data.
The cross-dataset comparison is the meta-finding: which methods work on which
data structures, and why.

| Method | iMSMS (paired, homogeneous) | 4CE (unpaired, heterogeneous) |
|---|---|---|
| Bracket norm | 60 deg rotation, p=0.002 | N/A (no paired design) |
| Sheaf (correlations) | Null (homogeneous coverage) | z=25 (heterogeneous coverage) |
| Meta-analysis | I^2=0% (homogeneous) | I^2=99.8% (heterogeneous) |
| CD-NOD | MS->EDSS, PCs->EDSS | neuro<->mortality |
| DAG CI | 64% consistent | 42% consistent |
| Lie bracket | Null (7 sites, low power) | age x mortality significant |
| Effect heterogeneity | 0/3 moderators | 4/4 moderators |

## Datasets

### Development: 4CE COVID (Phase2.1NeuroAnalysis)
- 21 healthcare sites, 6 countries
- Neurological phenotype counts (20 ICD-10 categories)
- Site demographics (age, sex, severity, mortality, readmission)
- Elixhauser comorbidity matrices (30 categories, 435 pairwise correlations)
- PMI (proportional morbidity index) effect estimates by site
- Heterogeneous coverage (stalk sizes 36 to 1305)

### Validation: BCD/NTZ synthetic data
- 1,175 synthetic patients calibrated to published Table 1
- 19-edge assumed DAG with 324 CI implications
- Allows ground-truth validation of CI tests

## Already done (experiments 0-7)

- [x] CI test calibration (Fisher's z + RCIT, 6/6 correct)
- [x] DAG assumption testing (222/324 Fisher, 237/324 RCIT)
- [x] PC algorithm DAG discovery (F1=0.18, inter-confounder edges)
- [x] Sheaf H^1 on ICD counts (null, homogeneous coverage)
- [x] CD-NOD on 4CE site data (5 edges, neuro<->mortality)
- [x] Meta-analysis IV/DL/IF (divergence = the finding)
- [x] Effect heterogeneity meta-regression (4/4 moderators significant)
- [x] Lie bracket on effect surface (age x mortality, strengthening)
- [x] Sheaf H^1 on comorbidity correlations (z=25, sheaf wins)
- [x] RCIT on real 4CE structure (42% DAG consistency)

## Still needed (priority order)

### Must-do for paper

1. **E-values for meta-analysis estimates** (same fix as iMSMS)
   Compute E-values for the pooled PMI estimates. How strong would an unmeasured
   confounder need to be to nullify the CNS->severity effect?

2. **CD-NOD bootstrap stability** (same fix as iMSMS)
   Sample 10 of 20 sites 100 times. Report edge frequency across bootstraps.
   With 20 sites (vs 7 in iMSMS), this should be much more stable.

3. **Temporal analysis of Lie bracket**
   The age x mortality interaction strengthens from 30 to 90 days
   (beta 32.5 -> 38.3). Fit a linear model to the temporal trend and test
   whether the strengthening is significant.

4. **Cross-dataset comparison table + figure**
   The green/red pattern across methods x datasets (Table above). This is
   the meta-contribution: a practical guide for choosing tools based on
   data properties.

5. **Negative control for sheaf**
   Permute site labels within the comorbidity data. The sheaf should return
   to null. This validates that z=25 is genuine cross-site inconsistency,
   not an artifact of the test statistic.

### Should-do

6. **Clinical interpretation of sheaf-detected inconsistency**
   Which comorbidity pairs are most inconsistent across sites? The z=25
   says something is collectively wrong, but what specifically?
   Already have HIV-Lymphoma, Paralysis-WeightLoss, Alcohol-Drugs
   as top-Q pairs. Cluster the inconsistency pattern.

7. **Sensitivity to alpha for CD-NOD**
   Run CD-NOD at alpha = 0.01, 0.05, 0.10. Report edge stability.

8. **Connection to Visweswaran's Markov boundary work**
   The Markov boundary (2016 paper) provides the variable selection for
   CI testing. Frame the DAG CI testing as empirically validating the
   Markov boundary implied by the assumed DAG.

### Nice-to-have

9. **Simulation study**: generate data from a known DAG, apply all methods,
   report power/FPR for each. Validates the toolkit on ground truth.

10. **MechVal leg-scoring** of each DAG edge (requires semantic annotation)

## Paper structure

### Section 1 -- Introduction
Causal discovery from multi-site observational data. Standard tools
(Cochran's Q, I^2) test heterogeneity one variable at a time. New tools
(sheaf cohomology, Lie bracket) test structural consistency across the
full covariate space. When do geometric methods add value? When coverage
is heterogeneous. We demonstrate on 4CE COVID data across 21 sites.

### Section 2 -- Methods
- CD-NOD for causal graph discovery with distribution shifts
- RCIT (Strobl & Visweswaran) for nonparametric CI testing
- Meta-analysis (IV/DL/IF) and meta-regression
- Lie bracket on the effect surface
- Sheaf H^1 for cross-site structural consistency
- E-values for sensitivity analysis

### Section 3 -- Causal graph discovery
CD-NOD + bootstrap stability. RCIT DAG validation (42% consistency).
DAG CI testing with Fisher's z (69%) and RCIT (73%).

### Section 4 -- Cross-site consistency
Sheaf H^1: null on ICD counts (homogeneous), z=25 on comorbidities
(heterogeneous). Cochran's Q comparison. Why the sheaf wins when
coverage is heterogeneous.

### Section 5 -- Effect surface geometry
Meta-analysis divergence. Lie bracket: age x mortality interaction,
temporal strengthening. E-values.

### Section 6 -- Cross-dataset comparison with iMSMS
The comparison table. Each dataset reveals what the other cannot.
Practical recommendations: use sheaf when coverage is heterogeneous,
bracket norm when design is paired, CD-NOD when domains >7.

### Section 7 -- Discussion
When geometric methods earn their complexity. Connection to
Visweswaran's prior work (Markov boundary, kernel CIT, FDR control).
Limitations (aggregate data, linear synthetic generation).

## Venue

**Primary target**: Journal of Causal Inference (Visweswaran publishes here)
- Methods-focused, causal inference audience
- Sheaf + Lie bracket + CD-NOD = novel toolkit paper

**Alternative**: JAMIA (informatics + clinical data infrastructure)

## Instructions for other chat

The other chat should:

1. Copy `experiments/visweswaran/dag_validation/results/` to understand
   existing results
2. Read `experiments/visweswaran/dag_validation/RESULTS.md` for all
   experiment details
3. Implement experiments #1-5 above (E-values, CD-NOD bootstrap,
   temporal Lie bracket, cross-dataset comparison, sheaf negative control)
4. Write results.tex in `experiments/visweswaran/paper/`
5. The CD-NOD bootstrap can run locally (CPU, ~10 min with 20 sites)
   or on Modal
6. The cross-dataset comparison needs access to iMSMS results at
   `experiments/xia/imsms/results/tables/`

Key code locations:
- `experiments/visweswaran/dag_validation/src/causal_pipeline.py` (CD-NOD, meta-analysis)
- `experiments/visweswaran/dag_validation/src/sheaf_comorbidity.py` (sheaf H^1)
- `experiments/visweswaran/dag_validation/src/lie_bracket.py` (Lie bracket)
- `experiments/visweswaran/dag_validation/src/rcit_real_4ce.py` (RCIT)
- All results JSON files in `experiments/visweswaran/dag_validation/results/tables/`
