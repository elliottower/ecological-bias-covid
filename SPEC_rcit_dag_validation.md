# SPEC: Causal Graph Validation Using Visweswaran's RCIT on Xia's BCD/NTZ DAG

**Project:** Visweswaran collaboration pilot
**Status:** ready to build
**Compute:** laptop-scale (R + Python, no GPU)
**Data:** PUBLIC — OpenGWAS summary statistics + 4CE aggregate counts + published BCD/NTZ DAG

---

## 0. One-paragraph thesis

Xia's BCD/NTZ paper (medRxiv 2025) estimates the causal effect of B-cell depletion vs
natalizumab on MS outcomes using a doubly-robust estimator — but never tests whether
the assumed DAG is correct. We use Visweswaran's own RCIT package (kernel conditional
independence tests, J Causal Inference 2019) to test the conditional independence
assumptions implied by the published DAG, and his PCp algorithm (ACM TIST 2019) to
discover whether the data-driven graph matches the assumed one. Where they disagree,
we apply MechVal leg-scoring to identify which causal legs are supported vs suspect.
This directly bridges the two labs: Visweswaran's causal discovery tools applied to
Xia's clinical neurology question.

---

## 1. Data and tools

### Visweswaran's tools (all public)
- **RCIT** (R): `devtools::install_github("ericstrobl/RCIT")` — kernel CIT for
  nonparametric conditional independence testing
- **PCp** (MATLAB): `github.com/ericstrobl/PCp` — PC algorithm with edge-specific
  FDR control
- **MarkovBlanket** (MATLAB): `github.com/ericstrobl/MarkovBlanket` — Markov boundary
  discovery

### Data sources (all public, no IRB)
- **BCD/NTZ DAG structure:** extracted from the medRxiv preprint (DOI 10.1101/2025.01.24.25321100)
  — the assumed confounders, treatment, and outcome variables are published.
  Code at github.com/xialab2016/BCD_NTZ_SemiSupervisedCausal.
- **OpenGWAS summary statistics:** public MR instruments for the DAG variables
  (MS risk, BMI, vitamin D, smoking, immune markers). Used for two-sample MR
  to test edge directions.
- **4CE COVID aggregate data:** Figshare DOI 10.6084/m9.figshare.12152976.v1 —
  multi-site neurological outcome counts. Used for the sheaf consistency experiment
  where heterogeneous stalks should genuinely beat Cochran's Q.

---

## 2. Repo layout

```
experiments/visweswaran/dag_validation/
  config.yaml
  src/
    extract_dag.py           # parse BCD/NTZ DAG from paper/code
    test_ci.R                # RCIT conditional independence tests on DAG edges
    discover_dag.R           # PCp data-driven DAG discovery
    compare_dags.py          # assumed vs discovered, edge-by-edge
    mr_edge_test.py          # two-sample MR for edge directions
    mechval_scoring.py       # MechVal leg-scoring on each DAG edge
    sheaf_4ce.py             # sheaf consistency on 4CE multi-site data
    baselines.py             # standard partial-correlation DAG checks
    figures.py
    run_all.py
  results/
    tables/
    figures/
  tests/
    test_ci_calibration.py   # verify RCIT is calibrated on synthetic known-truth
```

---

## 3. The three experiments

### 3.1 DAG assumption testing with RCIT

The BCD/NTZ paper assumes a specific DAG for the backdoor adjustment set. Each
conditional independence implied by the DAG is a testable prediction.

**Method:**
- Extract all conditional independence implications from the published DAG
  (using d-separation).
- For each implication, test with RCIT (nonparametric kernel CIT).
- Report: which implications hold, which are violated, with edge-specific p-values.
- Apply Benjamini-Yekutieli FDR control (Visweswaran's PCp contribution).

**What we learn:**
- If all implications hold: the DAG is consistent with the data. The doubly-robust
  estimate is on firmer ground.
- If some fail: specific edges are suspect. MechVal leg-scoring identifies which
  causal legs are weakest and what evidence would fix them.

**Data for this experiment:**
- We cannot access the patient-level BCD/NTZ data. Instead, we:
  (a) Use OpenGWAS summary statistics for the same variables (MS risk, disability,
      BMI, vitamin D, immune markers) as a public proxy.
  (b) Construct a synthetic dataset matching the published summary statistics
      (means, SDs, correlation structure from Table 1 of the preprint) and test
      the DAG on that. Label this clearly as "consistency check on published
      summaries, not a replication."

**Baseline:** standard partial-correlation-based CI testing (Fisher's z-test).
RCIT must add value for nonlinear dependencies that Fisher's z misses, or it's
equivalent (and we report that honestly).

### 3.2 Data-driven DAG discovery with PCp

- Run PCp on the same data to discover the DAG from scratch (no assumed structure).
- Compare edge-by-edge: assumed DAG vs discovered DAG.
- For every edge where they disagree, report the PCp p-value and the RCIT test
  for the corresponding CI.

**What we learn:**
- Which assumed edges are data-supported vs assumed-but-untestable.
- Whether the backdoor adjustment set is correct, too large, or missing a variable.

### 3.3 Sheaf consistency on 4CE multi-site COVID data

This is the experiment where sheaf methods should genuinely beat Cochran's Q:
- 4CE has heterogeneous sites (different countries, different EHR systems,
  different variable subsets measured).
- The site graph is NOT complete (not all sites measure all variables).
- Restriction maps are non-trivial (ICD-10 mapping differences across countries).

**Method:**
- Download 4CE aggregate neurological phenotype counts from Figshare.
- Build sheaf over the site graph with heterogeneous stalks (sites measure
  different subsets of neurological outcomes).
- Compute sheaf Laplacian and H^1.
- Compare against Cochran's Q (which requires homogeneous measurements).

**Prediction:** H^1 detects inconsistencies that Q cannot because Q requires
all sites to measure the same thing. If H^1 = Cochran's Q again (as in the
batch1 simulation), we report that and concede the sheaf adds nothing here either.

**Baseline:** Cochran's Q on the subset of sites that share measurements.
The sheaf must either (a) use MORE sites than Q can (because it handles
heterogeneous stalks) or (b) detect structure Q misses. Otherwise null result.

---

## 4. Mandatory baselines and null models

### 4.1 Baselines
- Fisher's z-test partial correlations (for 3.1, the standard CI test)
- Cochran's Q (for 3.3)
- Simple logistic regression DAG check (for 3.2)

### 4.2 Calibration check (REQUIRED before real experiments)
- Generate synthetic data from a KNOWN DAG (ground truth).
- Run RCIT and PCp on it. Verify: Type I error at nominal level, power > 80%
  at planted effect sizes matching the BCD/NTZ published estimates.
- If calibration fails, fix before proceeding. Do not run on real data with
  an uncalibrated tool.

### 4.3 Permutation nulls
- Permute treatment labels in the synthetic data; verify all tests go null.
- Permute site labels in 4CE; verify sheaf H^1 goes null.

---

## 5. Deliverables

### Tables (CSV)
- T1_dag_edges.csv — every edge in the assumed DAG, RCIT p-value, FDR-adjusted,
  verdict (consistent / violated)
- T2_dag_comparison.csv — assumed vs PCp-discovered, edge-by-edge agreement
- T3_mechval_scoring.csv — MechVal leg-scores for each causal leg
- T4_sheaf_vs_cochran.csv — H^1 vs Q on 4CE data, with site coverage comparison
- T5_calibration.csv — Type I error and power on synthetic known-truth

### Figures (PNG)
- F1_dag_annotated.png — the BCD/NTZ DAG with edges colored by RCIT verdict
- F2_dag_comparison.png — assumed vs discovered DAG side-by-side
- F3_mechval_heatmap.png — MechVal leg-scores across edges
- F4_sheaf_4ce.png — sheaf consistency across 4CE sites
- F5_calibration.png — RCIT calibration on synthetic data

---

## 6. Acceptance criteria

1. RCIT calibration verified on synthetic known-truth BEFORE running on real data.
2. All DAG edges tested with both RCIT and Fisher's z; comparison reported.
3. PCp-discovered DAG compared edge-by-edge with assumed DAG.
4. MechVal scoring applied to every edge with a clear verdict.
5. 4CE sheaf experiment run with Cochran's Q baseline; reported honestly.
6. Null result clause: if RCIT = Fisher's z on this data, or sheaf = Q on 4CE,
   those are valid findings about when the fancy methods do/don't add value.

---

## 7. Why Visweswaran would care

- It uses HIS tools (RCIT, PCp) on a real clinical question from his co-author's lab.
- It demonstrates the exact use case he designed them for: validating DAG assumptions
  in observational clinical research.
- The MechVal layer adds something his tools don't currently provide: semantic scoring
  of causal legs (not just statistical edge testing, but "what kind of evidence supports
  this edge and what's missing").
- The 4CE experiment tests sheaf methods on his own consortium's data, where the
  heterogeneous-stalk case should genuinely arise.
- If his tools find problems in Xia's DAG, that's a conversation starter, not a criticism —
  it's exactly the kind of methodological contribution he'd want to make.

---

## 8. First commands

1. Install RCIT: `Rscript -e 'devtools::install_github("ericstrobl/RCIT")'`
2. Extract DAG: `python src/extract_dag.py` (parse from preprint + code)
3. Calibration: `Rscript src/test_ci.R --synthetic --known-truth`
4. RCIT on real: `Rscript src/test_ci.R --dag bcd_ntz`
5. PCp discovery: `Rscript src/discover_dag.R`
6. Compare: `python src/compare_dags.py`
7. MechVal: `python src/mechval_scoring.py`
8. 4CE sheaf: `python src/sheaf_4ce.py`
9. Full: `python src/run_all.py`
