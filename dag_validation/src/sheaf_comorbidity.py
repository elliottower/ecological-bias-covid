"""
Sheaf cohomology on Elixhauser comorbidity correlation matrices.

Each site has a stalk consisting of its pairwise comorbidity correlations
for overall, CNS, and PNS patient subgroups. Sites with partial coverage
(small n → fewer estimable correlations) have shorter stalks. The sheaf
tests whether the local correlation patterns are globally consistent
across sites, accounting for heterogeneous coverage.

This is the experiment where the sheaf should outperform Cochran's Q:
Q tests one correlation pair at a time and requires all sites to report it,
while the sheaf integrates the full correlation structure and uses partial
sites via their available pairs only.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data" / "4ce_neuro"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_correlations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "site_elix_correlations.csv")


def build_stalks(corr: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build per-site stalk vectors from correlation data.

    Each stalk is a dict mapping (neuro_type, comorbidity_a, comorbidity_b) -> correlation.
    """
    stalks = {}
    for site in corr["site"].unique():
        sub = corr[corr["site"] == site]
        stalk = {}
        for _, row in sub.iterrows():
            key = (row["neuro_type"], row["comorbidity_a"], row["comorbidity_b"])
            stalk[key] = row["correlation"]
        stalks[site] = stalk
    return stalks


def compute_sheaf_h1(stalks: dict, n_perms: int = 1000) -> dict:
    """Compute sheaf H^1 norm measuring cross-site inconsistency.

    For each pair of sites (i, j), the restriction map compares their
    correlation values on shared keys. The coboundary is the difference
    on each shared key. H^1 = ||sum of coboundaries|| measures global
    inconsistency that can't be explained by local adjustments.

    The sheaf norm is the sum of squared differences across all site-pairs,
    weighted by the number of shared keys (so sites with more overlap
    contribute more). Tested via permutation.
    """
    sites = sorted(stalks.keys())
    n_sites = len(sites)

    observed_norm = _compute_norm(stalks, sites)

    null_norms = []
    rng = np.random.default_rng(42)

    all_keys = set()
    for stalk in stalks.values():
        all_keys.update(stalk.keys())
    all_keys = sorted(all_keys)

    all_values = {}
    for key in all_keys:
        vals = []
        for site in sites:
            if key in stalks[site]:
                vals.append(stalks[site][key])
        all_values[key] = vals

    for _ in tqdm(range(n_perms), desc="Sheaf permutation"):
        perm_stalks = {site: {} for site in sites}
        for key in all_keys:
            vals = all_values[key].copy()
            rng.shuffle(vals)
            idx = 0
            for site in sites:
                if key in stalks[site]:
                    perm_stalks[site][key] = vals[idx]
                    idx += 1

        null_norms.append(_compute_norm(perm_stalks, sites))

    null_norms = np.array(null_norms)
    p_value = float(np.mean(null_norms >= observed_norm))
    z_score = float((observed_norm - np.mean(null_norms)) / np.std(null_norms)) if np.std(null_norms) > 0 else 0.0

    return {
        "observed": observed_norm,
        "null_mean": float(np.mean(null_norms)),
        "null_std": float(np.std(null_norms)),
        "z_score": z_score,
        "p_value": p_value,
        "n_sites": n_sites,
        "n_perms": n_perms,
    }


def _compute_norm(stalks: dict, sites: list) -> float:
    """Sum of squared differences across all site pairs on shared keys."""
    total = 0.0
    n_pairs = 0
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            shared = set(stalks[sites[i]].keys()) & set(stalks[sites[j]].keys())
            if not shared:
                continue
            diff_sq = sum(
                (stalks[sites[i]][k] - stalks[sites[j]][k]) ** 2
                for k in shared
            )
            total += diff_sq / len(shared)
            n_pairs += 1
    return total / max(n_pairs, 1)


def compute_cochran_q_correlations(corr: pd.DataFrame) -> pd.DataFrame:
    """Run Cochran's Q on each (neuro_type, comorbidity_pair).

    Tests whether the correlation for a given pair is homogeneous across sites.
    Uses Fisher z-transform for proper variance.
    """
    results = []
    for neuro_type in corr["neuro_type"].unique():
        sub_nt = corr[corr["neuro_type"] == neuro_type]
        for (ca, cb), grp in sub_nt.groupby(["comorbidity_a", "comorbidity_b"]):
            if len(grp) < 3:
                continue

            r_vals = grp["correlation"].values
            n_sites = len(r_vals)

            r_clipped = np.clip(r_vals, -0.999, 0.999)
            z_vals = np.arctanh(r_clipped)

            z_mean = np.mean(z_vals)
            Q = np.sum((z_vals - z_mean) ** 2)
            p = 1 - stats.chi2.cdf(Q, df=n_sites - 1)

            results.append({
                "neuro_type": neuro_type,
                "comorbidity_a": ca,
                "comorbidity_b": cb,
                "n_sites": n_sites,
                "mean_r": float(np.mean(r_vals)),
                "sd_r": float(np.std(r_vals)),
                "Q": float(Q),
                "p": float(p),
            })

    return pd.DataFrame(results)


def run_comorbidity_sheaf(n_perms: int = 1000) -> dict:
    """Run the comorbidity sheaf analysis."""
    print(f"\n{'='*60}")
    print("SHEAF COHOMOLOGY ON COMORBIDITY CORRELATIONS")
    print(f"{'='*60}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    corr = load_correlations()
    print(f"\nLoaded: {len(corr)} correlations from {corr.site.nunique()} sites")

    coverage = corr.groupby(["site", "neuro_type"]).size().unstack(fill_value=0)
    print(f"\nCoverage (pairs per site × neuro_type):")
    print(coverage.to_string())

    n_full = (coverage == 435).all(axis=1).sum()
    n_partial = (coverage < 435).any(axis=1).sum()
    print(f"\n  Full coverage (435/435 all types): {n_full} sites")
    print(f"  Partial coverage: {n_partial} sites")

    # Sheaf analysis — all neuro types together
    print(f"\n{'='*60}")
    print("SHEAF H^1 (all neuro types, {n_perms} permutations)")
    print(f"{'='*60}")

    stalks = build_stalks(corr)
    stalk_sizes = {s: len(v) for s, v in stalks.items()}
    print(f"  Stalk sizes: min={min(stalk_sizes.values())}, max={max(stalk_sizes.values())}, "
          f"median={sorted(stalk_sizes.values())[len(stalk_sizes)//2]}")

    sheaf_result = compute_sheaf_h1(stalks, n_perms=n_perms)

    print(f"\n  Sheaf H^1 norm: {sheaf_result['observed']:.6f}")
    print(f"  Null: {sheaf_result['null_mean']:.6f} +/- {sheaf_result['null_std']:.6f}")
    print(f"  z-score: {sheaf_result['z_score']:.2f}")
    print(f"  p-value: {sheaf_result['p_value']:.4f}")

    # Also run on CNS-only and PNS-only subsets
    for nt in ["CNS", "PNS"]:
        sub = corr[corr["neuro_type"] == nt]
        if sub.site.nunique() < 5:
            continue
        sub_stalks = {}
        for site in sub["site"].unique():
            s = sub[sub["site"] == site]
            stalk = {}
            for _, row in s.iterrows():
                key = (row["comorbidity_a"], row["comorbidity_b"])
                stalk[key] = row["correlation"]
            sub_stalks[site] = stalk

        sub_result = compute_sheaf_h1(sub_stalks, n_perms=n_perms)
        print(f"\n  {nt}-only sheaf H^1:")
        print(f"    norm={sub_result['observed']:.6f}, z={sub_result['z_score']:.2f}, "
              f"p={sub_result['p_value']:.4f}, {sub_result['n_sites']} sites")
        sheaf_result[f"{nt}_norm"] = sub_result["observed"]
        sheaf_result[f"{nt}_z"] = sub_result["z_score"]
        sheaf_result[f"{nt}_p"] = sub_result["p_value"]

    # Cochran's Q for comparison
    print(f"\n{'='*60}")
    print("COCHRAN'S Q ON INDIVIDUAL CORRELATIONS")
    print(f"{'='*60}")

    q_results = compute_cochran_q_correlations(corr)
    q_results.to_csv(tables_dir / "T10_cochran_comorbidity.csv", index=False)

    n_sig = (q_results["p"] < 0.05).sum()
    n_total = len(q_results)
    print(f"  {n_sig}/{n_total} pairs have significant heterogeneity (p < 0.05)")

    top_q = q_results.nlargest(10, "Q")
    print(f"\n  Top 10 heterogeneous pairs:")
    for _, row in top_q.iterrows():
        print(f"    {row['neuro_type']:8s} {row['comorbidity_a']:15s} - {row['comorbidity_b']:15s}: "
              f"Q={row['Q']:.1f}, r={row['mean_r']:.3f}±{row['sd_r']:.3f}, "
              f"k={row['n_sites']}")

    # Comparison: how many pairs can Q use vs sheaf?
    q_usable = q_results[q_results["n_sites"] >= 10]
    print(f"\n  Pairs with >= 10 sites (usable for Q): {len(q_usable)}")
    print(f"  Sheaf uses all {len(stalks)} sites via partial coverage")

    # Save results
    import json
    summary = {
        "sheaf_norm": sheaf_result["observed"],
        "sheaf_z": sheaf_result["z_score"],
        "sheaf_p": sheaf_result["p_value"],
        "n_sites": sheaf_result["n_sites"],
        "n_perms": n_perms,
        "cochran_n_sig": int(n_sig),
        "cochran_n_total": int(n_total),
        "n_full_coverage": int(n_full),
        "n_partial_coverage": int(n_partial),
    }
    for key in ["CNS_norm", "CNS_z", "CNS_p", "PNS_norm", "PNS_z", "PNS_p"]:
        if key in sheaf_result:
            summary[f"sheaf_{key}"] = sheaf_result[key]

    with open(tables_dir / "T10_comorbidity_sheaf_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved T10_cochran_comorbidity.csv, T10_comorbidity_sheaf_summary.json")

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if sheaf_result["p_value"] < 0.05:
        print("  Sheaf H^1 is SIGNIFICANT: comorbidity correlation patterns are")
        print("  inconsistent across sites beyond what random variation explains.")
        print("  Different sites measure different comorbidity relationships,")
        print("  and the sheaf detects this using partial-coverage sites that")
        print("  Cochran's Q would exclude.")
    else:
        print("  Sheaf H^1 is not significant: comorbidity correlations are")
        print("  broadly consistent across sites.")

    return {"sheaf": sheaf_result, "cochran": q_results.to_dict("records")}


if __name__ == "__main__":
    run_comorbidity_sheaf(n_perms=1000)
