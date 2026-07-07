"""
Causal inference pipeline on 4CE multi-site neurological data.

Stage 1: CD-NOD (Causal Discovery from Nonstationary/heterogeneous Data)
  — discovers causal graph among clinical variables using cross-site
    heterogeneity as the identifying variation.

Stage 2: Double ML / TMLE-style meta-analysis
  — pools site-specific PMI estimates using influence-function-based
    methods, properly handling heterogeneity and small-sample separation.

Stage 3: Effect heterogeneity testing
  — tests whether the CNS→severity effect varies by site characteristics.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data" / "4ce_neuro"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_demographics() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "site_demographics.csv")


def load_pmi_effects() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "site_pmi_effects.csv")


def load_clinical() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "site_clinical.csv")


def build_site_level_dataset(demo: pd.DataFrame, clin: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (site, neuro_type) with all features for causal discovery.

    Merges demographics and clinical data. Encodes neuro_type as ordinal
    (None=0, Peripheral=1, Central=2) for continuous CI tests.
    """
    merged = demo.merge(clin, on=["site", "neuro_type"], how="left")

    neuro_map = {"None": 0, "Peripheral": 1, "Central": 2}
    merged["neuro_severity"] = merged["neuro_type"].map(neuro_map)

    site_ids = {s: i for i, s in enumerate(sorted(merged["site"].unique()))}
    merged["site_id"] = merged["site"].map(site_ids)

    merged = merged.dropna(subset=["prop_severe", "prop_deceased", "n_patients"])
    merged = merged[merged["n_patients"] >= 5]

    return merged


def run_cdnod(data: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Run CD-NOD causal discovery treating each site as a domain.

    CD-NOD (Zhang et al.) exploits distribution shifts across domains
    to identify causal directions beyond the Markov equivalence class.
    """
    from causallearn.search.ConstraintBased.CDNOD import cdnod

    feature_cols = [
        "neuro_severity", "prop_severe", "prop_deceased",
        "prop_readmitted", "prop_female",
        "prop_age_50to69", "prop_age_70to79", "prop_age_80plus",
    ]

    available = [c for c in feature_cols if c in data.columns]
    sub = data[available + ["site_id"]].dropna()

    if len(sub) < 10:
        print(f"  Warning: only {len(sub)} observations, CD-NOD may be unreliable")

    X = sub[available].values
    c_indx = sub["site_id"].values.reshape(-1, 1)

    print(f"  Running CD-NOD: {X.shape[0]} observations, {X.shape[1]} variables, "
          f"{len(np.unique(c_indx))} domains")
    print(f"  Variables: {available}")

    cg = cdnod(X, c_indx, alpha=alpha)

    n_vars = len(available)
    edges = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            edge_type = cg.G.graph[i, j]
            if edge_type == -1 and cg.G.graph[j, i] == 1:
                edges.append({
                    "source": available[i],
                    "target": available[j],
                    "type": "directed",
                    "edge_mark": f"{available[i]} -> {available[j]}",
                })
            elif edge_type == -1 and cg.G.graph[j, i] == -1 and i < j:
                edges.append({
                    "source": available[i],
                    "target": available[j],
                    "type": "undirected",
                    "edge_mark": f"{available[i]} -- {available[j]}",
                })

    print(f"  Discovered {len(edges)} edges:")
    for e in edges:
        print(f"    {e['edge_mark']}")

    return {
        "edges": edges,
        "variables": available,
        "n_observations": len(sub),
        "n_domains": len(np.unique(c_indx)),
        "alpha": alpha,
        "graph_matrix": cg.G.graph.tolist(),
    }


def filter_valid_pmi(pmi: pd.DataFrame, max_se: float = 2.0) -> pd.DataFrame:
    """Filter out PMI estimates with separation artifacts.

    Sites with very small cell counts produce extreme PMIs (10^6+) with
    huge SEs. We keep only estimates with SE < max_se on the log scale,
    which corresponds to a reasonable confidence interval width.
    """
    valid = pmi[pmi["se"] < max_se].copy()
    n_dropped = len(pmi) - len(valid)
    if n_dropped > 0:
        print(f"  Filtered {n_dropped}/{len(pmi)} estimates with SE >= {max_se}")
    return valid


def inverse_variance_meta(log_pmi: np.ndarray, se: np.ndarray) -> dict:
    """Standard inverse-variance weighted fixed-effects meta-analysis."""
    w = 1.0 / (se ** 2)
    pooled = np.sum(w * log_pmi) / np.sum(w)
    pooled_se = 1.0 / np.sqrt(np.sum(w))
    z = pooled / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    Q = np.sum(w * (log_pmi - pooled) ** 2)
    k = len(log_pmi)
    Q_p = 1 - stats.chi2.cdf(Q, df=k - 1) if k > 1 else np.nan
    I2 = max(0, (Q - (k - 1)) / Q) if Q > 0 else 0

    return {
        "method": "inverse_variance",
        "pooled_log_pmi": float(pooled),
        "pooled_se": float(pooled_se),
        "pooled_pmi": float(np.exp(pooled)),
        "pooled_lower": float(np.exp(pooled - 1.96 * pooled_se)),
        "pooled_upper": float(np.exp(pooled + 1.96 * pooled_se)),
        "z": float(z),
        "p": float(p),
        "Q": float(Q),
        "Q_p": float(Q_p),
        "I2": float(I2),
        "k": k,
    }


def dersimonian_laird_meta(log_pmi: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects meta-analysis."""
    w = 1.0 / (se ** 2)
    pooled_fe = np.sum(w * log_pmi) / np.sum(w)

    Q = np.sum(w * (log_pmi - pooled_fe) ** 2)
    k = len(log_pmi)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0, (Q - (k - 1)) / c)

    w_re = 1.0 / (se ** 2 + tau2)
    pooled = np.sum(w_re * log_pmi) / np.sum(w_re)
    pooled_se = 1.0 / np.sqrt(np.sum(w_re))
    z = pooled / pooled_se
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "method": "dersimonian_laird",
        "pooled_log_pmi": float(pooled),
        "pooled_se": float(pooled_se),
        "pooled_pmi": float(np.exp(pooled)),
        "pooled_lower": float(np.exp(pooled - 1.96 * pooled_se)),
        "pooled_upper": float(np.exp(pooled + 1.96 * pooled_se)),
        "z": float(z),
        "p": float(p),
        "tau2": float(tau2),
        "k": k,
    }


def influence_function_meta(
    log_pmi: np.ndarray, se: np.ndarray
) -> dict:
    """TMLE-inspired influence-function-based meta-analysis.

    Uses the efficient influence function for the pooled log-PMI under
    a nonparametric model. The IF-based variance estimate is valid even
    when the between-study variance is misspecified, unlike DL which
    assumes a specific random-effects model.

    The EIF for the pooled effect psi under a random-effects model is:
      IF_i = (theta_i - psi) + (sigma_i^2 + tau^2)^{-1} * (theta_i - psi)
    where we use the empirical variance to estimate tau^2 adaptively.
    """
    k = len(log_pmi)
    tau2_hat = max(0, np.var(log_pmi, ddof=1) - np.mean(se ** 2))

    w = 1.0 / (se ** 2 + tau2_hat)
    psi = np.sum(w * log_pmi) / np.sum(w)

    if_values = w * (log_pmi - psi) / np.sum(w)
    var_if = np.sum(if_values ** 2)
    se_if = np.sqrt(var_if)

    z = psi / se_if if se_if > 0 else 0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "method": "influence_function",
        "pooled_log_pmi": float(psi),
        "pooled_se": float(se_if),
        "pooled_pmi": float(np.exp(psi)),
        "pooled_lower": float(np.exp(psi - 1.96 * se_if)),
        "pooled_upper": float(np.exp(psi + 1.96 * se_if)),
        "z": float(z),
        "p": float(p),
        "tau2_hat": float(tau2_hat),
        "k": k,
    }


def run_meta_analysis(pmi: pd.DataFrame) -> pd.DataFrame:
    """Run all three meta-analysis methods on each (effect, timepoint) combo."""
    results = []

    for effect_name in pmi["effect_name"].unique():
        for tp in pmi["timepoint"].unique():
            sub = pmi[(pmi["effect_name"] == effect_name) & (pmi["timepoint"] == tp)]
            if len(sub) < 3:
                continue

            log_pmi = sub["log_pmi"].values
            se = sub["se"].values

            for method_fn in [inverse_variance_meta, dersimonian_laird_meta,
                              influence_function_meta]:
                result = method_fn(log_pmi, se)
                result["effect_name"] = effect_name
                result["timepoint"] = tp
                results.append(result)

    return pd.DataFrame(results)


def test_effect_heterogeneity(
    pmi: pd.DataFrame, demo: pd.DataFrame
) -> pd.DataFrame:
    """Test whether CNS→severity effect varies by site characteristics.

    Uses meta-regression: log_PMI ~ site_characteristic, weighted by
    inverse variance. Significant slope means the effect is modified
    by that characteristic.
    """
    cns_30 = pmi[(pmi["effect_name"] == "pmi.cns") & (pmi["timepoint"] == 30)].copy()

    none_demo = demo[demo["neuro_type"].isna()].copy()
    none_demo = none_demo.rename(columns={
        "prop_female": "site_prop_female",
        "prop_age_80plus": "site_prop_elderly",
        "prop_severe": "site_baseline_severity",
        "n_patients": "site_n_patients",
    })

    merged = cns_30.merge(
        none_demo[["site", "site_prop_female", "site_prop_elderly",
                    "site_baseline_severity", "site_n_patients"]],
        on="site", how="inner",
    )

    if len(merged) < 5:
        return pd.DataFrame()

    results = []
    moderators = ["site_prop_female", "site_prop_elderly",
                  "site_baseline_severity", "site_n_patients"]

    for mod in moderators:
        sub = merged.dropna(subset=[mod, "log_pmi", "se"])
        if len(sub) < 5:
            continue

        w = 1.0 / (sub["se"].values ** 2)
        x = sub[mod].values
        y = sub["log_pmi"].values

        x_mean = np.average(x, weights=w)
        y_mean = np.average(y, weights=w)
        ss_xy = np.sum(w * (x - x_mean) * (y - y_mean))
        ss_xx = np.sum(w * (x - x_mean) ** 2)

        if ss_xx < 1e-10:
            continue

        beta = ss_xy / ss_xx
        se_beta = 1.0 / np.sqrt(ss_xx)
        z = beta / se_beta
        p = 2 * (1 - stats.norm.cdf(abs(z)))

        results.append({
            "moderator": mod,
            "beta": float(beta),
            "se": float(se_beta),
            "z": float(z),
            "p": float(p),
            "n_sites": len(sub),
            "moderator_range": f"{sub[mod].min():.3f} - {sub[mod].max():.3f}",
        })

    return pd.DataFrame(results)


def run_full_pipeline(alpha: float = 0.1) -> dict:
    """Run the complete causal inference pipeline."""
    print(f"\n{'='*60}")
    print("CAUSAL INFERENCE PIPELINE ON 4CE DATA")
    print(f"{'='*60}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    demo = load_demographics()
    pmi_raw = load_pmi_effects()
    clin = load_clinical()
    print(f"\nLoaded: {demo.site.nunique()} sites, {len(pmi_raw)} PMI estimates")

    # Stage 1: CD-NOD
    print(f"\n{'='*60}")
    print("STAGE 1: CD-NOD CAUSAL DISCOVERY")
    print(f"{'='*60}")

    site_data = build_site_level_dataset(demo, clin)
    print(f"  Site-level dataset: {len(site_data)} rows, "
          f"{site_data.site.nunique()} sites")

    cdnod_result = run_cdnod(site_data, alpha=alpha)

    cdnod_df = pd.DataFrame(cdnod_result["edges"])
    if len(cdnod_df) > 0:
        cdnod_df.to_csv(tables_dir / "T6_cdnod_edges.csv", index=False)
        print(f"  Saved T6_cdnod_edges.csv")

    # Stage 2: Meta-analysis comparison
    print(f"\n{'='*60}")
    print("STAGE 2: META-ANALYSIS (IV vs DL vs IF)")
    print(f"{'='*60}")

    pmi = filter_valid_pmi(pmi_raw, max_se=2.0)
    print(f"  Valid PMI estimates: {len(pmi)} across {pmi.site.nunique()} sites")

    meta_results = run_meta_analysis(pmi)
    meta_results.to_csv(tables_dir / "T7_meta_analysis.csv", index=False)
    print(f"  Saved T7_meta_analysis.csv")

    # Print key comparison
    cns_30 = meta_results[
        (meta_results["effect_name"] == "pmi.cns") &
        (meta_results["timepoint"] == 30)
    ]
    if len(cns_30) > 0:
        print(f"\n  CNS -> Severity (30-day) pooled estimates:")
        for _, row in cns_30.iterrows():
            print(f"    {row['method']:25s}: PMI = {row['pooled_pmi']:.3f} "
                  f"[{row['pooled_lower']:.3f}, {row['pooled_upper']:.3f}], "
                  f"p = {row['p']:.2e}")

    # Stage 3: Effect heterogeneity
    print(f"\n{'='*60}")
    print("STAGE 3: EFFECT HETEROGENEITY TESTING")
    print(f"{'='*60}")

    hetero = test_effect_heterogeneity(pmi, demo)
    if len(hetero) > 0:
        hetero.to_csv(tables_dir / "T8_heterogeneity.csv", index=False)
        print(f"  Saved T8_heterogeneity.csv")
        print(f"\n  CNS -> Severity moderation by site characteristics:")
        for _, row in hetero.iterrows():
            sig = "*" if row["p"] < 0.05 else ""
            print(f"    {row['moderator']:30s}: beta = {row['beta']:+.3f}, "
                  f"p = {row['p']:.4f}{sig}")
    else:
        print("  Insufficient data for heterogeneity testing")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")

    return {
        "cdnod": cdnod_result,
        "meta": meta_results.to_dict("records"),
        "heterogeneity": hetero.to_dict("records") if len(hetero) > 0 else [],
    }


if __name__ == "__main__":
    run_full_pipeline()
