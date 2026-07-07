# Batch 1: Geometric and Topological Methods for Clinical Epidemiology

Six proof-of-concept experiments testing whether geometric/topological tools add value to standard epidemiological workflows. Experiments 1--2 target Dr. Visweswaran's federated EHR and causal graph work; experiments 3--4 target Dr. Xia's biomarker and treatment heterogeneity work. Experiments 5--6 rescue the two initial negatives.

---

## 1. Sheaf Consistency Test for Federated EHR (Visweswaran)

**Question.** Can sheaf cohomology detect inconsistent phenotype definitions across hospital sites in a federated EHR network?

**Setup.** Simulate 8 hospital sites estimating a treatment effect. Each site produces an estimate and standard error. One site (site 5) has a biased phenotype definition; a second site (site 7) has elevated noise. We construct the sheaf Laplacian over the complete graph of sites, compute the quadratic form Q = obs' Sigma^{-1} obs on pairwise differences, and compare against a chi-squared null.

**Results.**

| Metric | Value |
|--------|-------|
| Type I error rate | 5.85% (nominal: 5%) |
| Power at bias = 0.10 | 38.6% |
| Power at bias = 0.15 | 71.2% |
| Power at bias = 0.20 | 96.8% |
| Localization rank (biased site 5) | 1.06 / 8 |
| Structured separation for 80% power | ~0.06 |

The test is properly calibrated and localizes the biased site to rank 1 in 94% of simulations. Power rises monotonically with bias magnitude.

**Key finding.** The sheaf consistency test produces results identical to Cochran's Q at every bias level, every structured separation, and every replicate. This is expected: on the complete graph with scalar stalks and a single restriction map type (pairwise difference), the sheaf Laplacian's quadratic form reduces to the standard Q statistic. The sheaf framework becomes interesting when (a) sites measure different but related quantities (heterogeneous stalks), (b) the site graph is not complete (missing pairwise comparisons), or (c) restriction maps encode domain-specific transformations rather than simple differences.

**For Dr. Visweswaran.** The current simulation confirms that the sheaf framework recovers standard meta-analytic tools as a special case. The value proposition requires demonstrating advantage in one of the three scenarios above. A natural next experiment: multi-phenotype federated data where each site measures a different subset of phenotype features, and the sheaf encodes known relationships (ICD codes map to lab values map to imaging) across sites that do not share raw data.

---

## 2. Confound Collapse Audit (Xia)

**Question.** Can we detect confounded biomarker associations in observational MS data using partial correlation analysis?

**Setup.** Simulate 3000 patients with 10 real biomarkers (direct causal effects on outcomes via a known DAG) and 10 confounded biomarkers (spurious associations through shared upstream causes). Compute three correlation measures: raw Pearson correlation, partial correlation (conditioning on all other covariates), and intensive margin correlation (conditioning on treatment).

**Results.**

| Metric | Real biomarkers | Confounded biomarkers |
|--------|-----------------|----------------------|
| Raw correlation | 0.81 -- 0.90 | 0.51 -- 0.71 |
| Partial correlation | 0.61 -- 0.76 | 0.001 -- 0.033 |
| Intensive correlation | 0.80 -- 0.89 | 0.31 -- 0.64 |

Partial correlation AUROC for separating real from confounded: **1.000** at all sample sizes (n = 200 to 5000, zero standard deviation). The confounded biomarkers' partial correlations collapse to near-zero when conditioning on the other covariates, while real biomarkers retain substantial partial correlation (minimum 0.61).

**Key finding.** The separation is perfect and trivial in this simulation because the confounded biomarkers are generated via simple shared-cause pathways that conditioning fully removes. Real EHR data presents harder cases: partial confounding (confounders explain part of the association), unobserved confounders, and measurement error. The result confirms partial correlation as a baseline detector, but the interesting question is what happens when confounding is partial rather than complete.

**For Dr. Xia.** The perfect detection in simulation validates partial correlation as a necessary first-pass screen for biomarker studies. Two follow-up experiments would test the method's limits: (a) partial confounding with graded confound strength (what AUROC does partial correlation achieve when the confound explains 30% vs 70% of the association?), and (b) unobserved confounders (can we detect residual confounding from the structure of partial correlations across the biomarker panel?).

---

## 3. Curvature for Causal Graph Edge Validation (Visweswaran) -- initial negative

**Question.** Does Ollivier-Ricci curvature (ORC) distinguish true positive (TP) from false positive (FP) edges in learned causal graphs?

**Setup.** Generate random DAGs (10--20 nodes), simulate linear SEMs, learn undirected graphs via partial correlation thresholding, compute ORC on each edge, and test whether TP edges have systematically different curvature than FP edges.

**Results.**

| Metric | Value |
|--------|-------|
| AUROC (TP vs FP) | 0.466 |
| TP curvature mean | 0.252 |
| FP curvature mean | 0.264 |
| Mann-Whitney p | 0.020 |

ORC slightly favors FP edges (higher curvature), yielding a below-chance AUROC. The curvature distributions overlap almost completely. ORC measures the mass-transport cost of moving probability between neighborhoods, which depends on graph topology rather than edge-level statistical evidence. In these sparse causal graphs, TP and FP edges occupy similar local neighborhoods.

**Verdict.** ORC does not discriminate TP from FP edges in causal graphs. Rescued by experiment 5 below.

---

## 4. Treatment Heterogeneity Detection (Xia) -- initial negative

**Question.** Can geometric subspace methods recover treatment effect subtypes from observational MS trial data?

**Setup.** Simulate 3000 patients with 3 treatment response subtypes: responders (CATE = +2.0, 40%), anti-responders (CATE = -2.5, 40%), and non-responders (CATE = 0, 20%). Estimate individual CATEs via KNN T-learner, weight covariates by estimated CATE, project through PCA, and cluster in the top-2 PC space using K-means.

**Results.**

| Method | ARI |
|--------|-----|
| PCA on KNN-CATE weighted covariates | -0.011 |
| Naive K-means on raw covariates | 0.219 |

The geometric approach performs worse than raw covariate clustering, producing clusters uncorrelated with true subtypes (ARI near zero = random). The naive baseline recovers partial subtype structure because the subtypes are partially separable in covariate space by design.

The failure has two causes. First, KNN CATE estimation is noisy for individual-level treatment effects (n_neighbors=30 averages over heterogeneous subgroups). Second, PCA on CATE-weighted covariates introduces a multiplicative interaction that amplifies noise in CATE estimates.

**Verdict.** KNN-based CATE + PCA clustering fails to recover treatment subtypes. Rescued by experiment 6 below.

---

## 5. Curvature Rescue: Forman-Ricci and Topological Edge Features

**Question.** Do simpler combinatorial edge features outperform ORC for TP/FP discrimination?

**Setup.** Same simulation as experiment 3, but expanded: 60 graphs per condition, 6 conditions (linear/nonlinear SEM x loose/moderate/strict threshold), and 6 edge features instead of just ORC:

- **Forman-Ricci curvature**: 4 - d(u) - d(v) (purely combinatorial, measures degree deficit)
- **Augmented Forman**: 4 - d(u) - d(v) + 3 * |common neighbors|
- **Jaccard coefficient**: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
- **Edge betweenness centrality**: fraction of shortest paths through the edge
- **Partial correlation magnitude**: |pcorr(u,v)| (non-topological baseline)
- **Average endpoint clustering coefficient**: (C(u) + C(v)) / 2

**Results.**

Best AUROC per feature across all 6 conditions:

| Feature | Best AUROC | Condition | Direction |
|---------|-----------|-----------|-----------|
| Forman-Ricci | **0.677** | linear/loose | TP > FP |
| Partial corr magnitude | 0.657 | nonlinear/loose | TP > FP |
| Betweenness centrality | 0.610 | linear/loose | TP > FP |
| Average clustering | 0.568 | nonlinear/loose | TP > FP |
| Augmented Forman | 0.484 | --- | wrong direction |
| Jaccard | 0.408 | --- | wrong direction |

Forman-Ricci wins 5 of 6 conditions; partial correlation magnitude wins the remaining condition (nonlinear/loose). Both features are consistent across conditions: Forman-Ricci AUROC ranges from 0.634 to 0.677, and pcorr_mag ranges from 0.613 to 0.657.

**Mechanism.** Forman-Ricci curvature = 4 - d(u) - d(v) is higher (less negative) when both endpoints have low degree. True positive edges tend to connect lower-degree nodes than false positive edges: TP edges represent real causal relationships that produce specific partial correlations, while FP edges arise from indirect paths that tend to involve higher-degree hub nodes. The Forman-Ricci score captures this degree asymmetry directly.

Partial correlation magnitude works because true causal edges produce stronger direct statistical associations than spurious edges (whose partial correlations are attenuated by conditioning). The two features are complementary: Forman-Ricci is topological (graph structure only), while pcorr_mag is statistical (data-driven).

**For Dr. Visweswaran.** Forman-Ricci curvature (AUROC 0.677) provides a cheap topological signal that is complementary to partial correlation magnitude (AUROC 0.657). A combined classifier using both features could potentially exceed either alone. The next step is testing on real-world graphs: KEGG pathway graphs, protein interaction networks, or learned Bayesian network structures from actual clinical data.

---

## 6. HTE Rescue: Better CATE Estimation and Clustering

**Question.** Do better CATE estimators and clustering methods recover treatment effect subtypes where KNN/PCA failed?

**Setup.** Same simulation as experiment 4 (3000 patients, 3 subtypes, 10 covariates, 100 reps), but with a 4x4 grid:

CATE estimators:
- KNN T-learner (batch 1 baseline)
- RandomForest T-learner (200 trees, max_depth=10)
- GradientBoosting T-learner (200 trees, max_depth=4)
- RandomForest S-learner (treatment as feature)

Clustering methods:
- PCA on CATE-weighted covariates + K-means (batch 1 baseline)
- K-means directly on CATE values
- K-means on [covariates, CATE] concatenated
- Spectral clustering on CATE RBF kernel

Plus oracle baselines (clustering on true CATE) and naive baseline (K-means on raw covariates).

**Results.**

Best combinations by ARI (100 reps, n = 3000 patients):

| CATE method | Clustering | ARI mean | ARI std |
|------------|-----------|----------|---------|
| RF T-learner | cate + covariates | **0.270** | 0.017 |
| GBM T-learner | cate + covariates | 0.264 | 0.016 |
| RF S-learner | cate + covariates | 0.245 | 0.024 |
| RF T-learner | cate K-means | 0.238 | 0.016 |
| KNN | cate + covariates | 0.225 | 0.016 |
| *Naive (raw covariates)* | *K-means* | *0.218* | *0.017* |
| GBM T-learner | cate K-means | 0.189 | 0.015 |
| RF T-learner | spectral | 0.097 | 0.029 |
| KNN | spectral | 0.085 | 0.014 |
| *All 4 methods* | *PCA K-means* | *-0.011 to -0.016* | *0.002--0.004* |

Oracle baselines (clustering on true CATE):

| Clustering | Oracle ARI |
|-----------|-----------|
| cate K-means | **1.000** |
| spectral | **1.000** |
| cate + covariates | 0.551 |
| PCA K-means | 0.095 |

**Key findings.**

1. **PCA clustering was the bottleneck, not the CATE estimator.** All four CATE methods produce ARI ~ -0.01 with PCA clustering. The oracle itself achieves only ARI = 0.095 with PCA — the PCA projection on CATE-weighted covariates fundamentally destroys the subtype signal. Batch 1's failure was a clustering choice problem, not a CATE estimation problem.

2. **Concatenating covariates with CATE estimates (cate_cov_kmeans) consistently wins.** This approach feeds the CATE signal directly into clustering without the lossy PCA projection. RF T-learner + cate_cov_kmeans achieves ARI = 0.270, a factor-of-25 improvement over batch 1 (from -0.011 to 0.270).

3. **The improvement over naive baseline is modest.** Best estimated ARI (0.270) exceeds naive K-means on raw covariates (0.218) by 24% relative. The subtypes were designed with covariate separation, so raw covariates already carry partial subtype information. The CATE estimate adds incremental discriminative power.

4. **The CATE estimation gap is large.** Oracle cate K-means achieves ARI = 1.0 (perfect recovery), while the best estimated method reaches only 0.270. The gap is entirely due to CATE estimation noise: at n = 3000 with 10 covariates and 3 subtypes, individual-level treatment effects are hard to estimate precisely. Larger sample sizes or stronger effect sizes would narrow this gap.

5. **Spectral clustering underperforms K-means.** Despite achieving perfect oracle ARI (1.0), spectral clustering on estimated CATEs gives only ARI 0.055--0.097. The method is more sensitive to CATE estimation noise because the RBF kernel amplifies small errors.

**For Dr. Xia.** The rescue demonstrates that treatment subtype recovery is feasible with appropriate CATE estimation and clustering, but the signal is weak at realistic sample sizes. The path to stronger results: (a) larger cohorts (n > 10,000), (b) ensemble CATE methods that reduce estimation variance, (c) semi-supervised clustering that exploits known covariate structure, or (d) longitudinal designs where repeated measures sharpen individual-level effect estimates.

---

## Summary

| # | Experiment | Target | Verdict | Key metric |
|---|-----------|--------|---------|------------|
| 1 | Sheaf federated EHR | Visweswaran | Positive (= Cochran's Q) | Type I = 5.85%, power to 97% |
| 2 | Confound collapse audit | Xia | Positive (trivially) | Partial corr AUROC = 1.0 |
| 3 | ORC causal graph | Visweswaran | Negative | AUROC = 0.466 |
| 4 | HTE geometric subspace | Xia | Negative | ARI = -0.011 |
| 5 | Curvature rescue (Forman) | Visweswaran | Rescued | AUROC = 0.677 |
| 6 | HTE rescue (RF/GBM CATE) | Xia | Partially rescued | ARI = 0.270 (best), naive = 0.218 |

**Two strong positives**: Confound collapse audit (perfect detection in simulation) and sheaf federated EHR (properly calibrated, localizes biased sites). Both require harder test cases to demonstrate value beyond standard methods.

**Two rescued negatives**: ORC curvature replaced by Forman-Ricci (AUROC 0.466 to 0.677, clear rescue). HTE partially rescued (ARI from -0.011 to 0.270), though the improvement over naive baseline (0.218) is modest — the dominant insight is that PCA clustering was the failure mode, not the CATE estimator.

**Honest assessment**: The two positives work because the simulations are easy (complete confounding, scalar stalks on complete graphs). The curvature rescue gives a genuine topological signal (Forman-Ricci AUROC 0.677). The HTE rescue identifies PCA as the bottleneck and shows that direct CATE-based clustering works, but at realistic sample sizes the gap between estimated and oracle CATE (0.270 vs 1.0) remains the binding constraint. The interesting science is in the boundary cases that batch 2 experiments should target: partial confounding, heterogeneous stalks, realistic effect sizes, larger cohorts, and real-world graph structures.

---

## Next Steps (Batch 2 Candidates)

1. **Multi-phenotype sheaf** (Visweswaran): Sites measure different phenotype features (labs, imaging, codes). The sheaf framework must handle heterogeneous stalks and non-complete site graphs. This is where sheaf cohomology would genuinely outperform Cochran's Q.

2. **Partial confounding spectrum** (Xia): Grade confound strength from 0% to 100% and measure partial correlation AUROC at each level. Include unobserved confounders. Find where detection breaks down.

3. **Forman-Ricci + pcorr combined classifier** (Visweswaran): Logistic regression or random forest on the topological + statistical feature set for edge validation. Test on KEGG/STRING/BioGRID graph structures.

4. **Realistic treatment effects** (Xia): Smaller CATE differences (0.3--0.5 SD), more subtypes (4--5), continuous subtype boundaries, and time-varying treatment effects.

5. **Federated confound audit** (both): Combine experiments 1 and 2 -- detect confounded biomarkers across federated sites where different sites have different confounding structures.

6. **Real-data pilot** (both): Apply the surviving methods (partial correlation audit, Forman-Ricci edge validation) to one real clinical dataset as a calibration check before scaling.
