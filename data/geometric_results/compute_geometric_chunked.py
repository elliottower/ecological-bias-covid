"""
Compute geometric statistics from CDC data using chunked reading.

Uses age_70plus (not 80+) to avoid zero-variation groups.
Reads in 2M-row chunks so we never hold the full 12GB in memory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

CSV_PATH = Path(__file__).parent.parent / "cdc_case_surveillance" / "cdc_full.csv"
RESULTS_DIR = Path(__file__).parent
USECOLS = ["age_group", "sex", "race_ethnicity_combined",
           "hosp_yn", "icu_yn", "death_yn", "medcond_yn"]
CHUNK_SIZE = 2_000_000


def process_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()
    chunk["died"] = (chunk["death_yn"] == "Yes").astype(np.int8)
    chunk["hospitalized"] = (chunk["hosp_yn"] == "Yes").astype(np.int8)
    chunk["icu"] = (chunk["icu_yn"] == "Yes").astype(np.int8)
    chunk["has_comorbidity"] = (chunk["medcond_yn"] == "Yes").astype(np.int8)
    chunk["age_70plus"] = chunk["age_group"].isin(
        ["70 - 79 Years", "80+ Years"]).astype(np.int8)
    chunk["age_80plus"] = (chunk["age_group"] == "80+ Years").astype(np.int8)
    chunk["male"] = (chunk["sex"] == "Male").astype(np.int8)
    chunk["site"] = chunk["race_ethnicity_combined"].str[:25]
    complete = chunk.dropna(subset=["age_group", "death_yn", "sex", "race_ethnicity_combined"])
    complete = complete[complete["death_yn"].isin(["Yes", "No"])]
    complete = complete[complete["sex"].isin(["Male", "Female"])]
    return complete


def main():
    print(f"Chunked geometric analysis — {datetime.now(timezone.utc).isoformat()}")

    binary_vars = ["died", "hospitalized", "icu", "has_comorbidity", "male", "age_70plus"]

    # Phase 1: Accumulate per-site sufficient statistics via chunks
    site_sums = {}      # site -> {var: sum}
    site_cross = {}     # site -> {(v1,v2): sum of products}
    site_counts = {}    # site -> n

    total_rows = 0
    reader = pd.read_csv(CSV_PATH, usecols=USECOLS, chunksize=CHUNK_SIZE,
                         low_memory=False, encoding="utf-8-sig")

    for chunk_idx, raw_chunk in enumerate(tqdm(reader, desc="Reading chunks")):
        chunk = process_chunk(raw_chunk)
        total_rows += len(chunk)

        for site, sub in chunk.groupby("site"):
            if site not in site_sums:
                site_sums[site] = {v: 0 for v in binary_vars}
                site_cross[site] = {}
                for i, v1 in enumerate(binary_vars):
                    for j, v2 in enumerate(binary_vars):
                        if j > i:
                            site_cross[site][(v1, v2)] = 0
                site_counts[site] = 0

            n = len(sub)
            site_counts[site] += n
            for v in binary_vars:
                site_sums[site][v] += int(sub[v].sum())
            for i, v1 in enumerate(binary_vars):
                for j, v2 in enumerate(binary_vars):
                    if j > i:
                        site_cross[site][(v1, v2)] += int((sub[v1] * sub[v2]).sum())

    print(f"\nTotal rows processed: {total_rows:,}")

    # Filter to sites with n >= 1000
    valid_sites = sorted([s for s, n in site_counts.items() if n >= 1000])
    print(f"Valid sites (n >= 1000): {len(valid_sites)}")
    for s in valid_sites:
        print(f"  {s:<30s}: n={site_counts[s]:>10,}")

    # Phase 2: Compute per-site correlation matrices from sufficient statistics
    site_corrs = {}
    n_vars = len(binary_vars)

    for site in valid_sites:
        n = site_counts[site]
        means = {v: site_sums[site][v] / n for v in binary_vars}

        corr = np.zeros((n_vars, n_vars))
        np.fill_diagonal(corr, 1.0)

        all_valid = True
        for i, v1 in enumerate(binary_vars):
            for j, v2 in enumerate(binary_vars):
                if j <= i:
                    continue
                key = (v1, v2)
                e_xy = site_cross[site][key] / n
                cov = e_xy - means[v1] * means[v2]
                std1 = np.sqrt(means[v1] * (1 - means[v1]))
                std2 = np.sqrt(means[v2] * (1 - means[v2]))
                if std1 < 1e-10 or std2 < 1e-10:
                    all_valid = False
                    break
                r = cov / (std1 * std2)
                corr[i, j] = r
                corr[j, i] = r
            if not all_valid:
                break

        if all_valid:
            site_corrs[site] = {
                "corr_matrix": corr,
                "n_patients": n,
                "means": means,
            }
        else:
            print(f"  SKIPPING {site}: zero variance in at least one variable")

    print(f"\nSites with valid correlations: {len(site_corrs)}")

    # Print key correlations
    print(f"\n  Per-site correlation: died × age_70plus")
    died_idx = binary_vars.index("died")
    age_idx = binary_vars.index("age_70plus")
    for site in sorted(site_corrs.keys()):
        r = site_corrs[site]["corr_matrix"][died_idx, age_idx]
        print(f"    {site:<30s}: r = {r:+.4f}  (n={site_corrs[site]['n_patients']:,})")

    # Phase 3: Sheaf (multivariate consistency) test
    print(f"\n  Running consistency test (200 permutations)...")
    sites = sorted(site_corrs.keys())
    n_sites = len(sites)
    tri_idx = np.triu_indices(n_vars, k=1)
    stalk_dim = len(tri_idx[0])

    site_vectors = {}
    for site in sites:
        site_vectors[site] = site_corrs[site]["corr_matrix"][tri_idx]

    def sheaf_norm(vecs, site_list):
        total = 0.0
        n_pairs = 0
        for i in range(len(site_list)):
            for j in range(i + 1, len(site_list)):
                diff = vecs[site_list[i]] - vecs[site_list[j]]
                total += np.sum(diff ** 2)
                n_pairs += 1
        return total / max(n_pairs, 1)

    observed = sheaf_norm(site_vectors, sites)

    rng = np.random.default_rng(42)
    all_vecs = np.array([site_vectors[s] for s in sites])
    null_norms = []
    for _ in tqdm(range(200), desc="Consistency permutation"):
        perm_vecs = {}
        for dim in range(stalk_dim):
            col = all_vecs[:, dim].copy()
            rng.shuffle(col)
            for i, site in enumerate(sites):
                if site not in perm_vecs:
                    perm_vecs[site] = np.zeros(stalk_dim)
                perm_vecs[site][dim] = col[i]
        null_norms.append(sheaf_norm(perm_vecs, sites))

    null_norms = np.array(null_norms)
    sheaf_z = float((observed - null_norms.mean()) / max(null_norms.std(), 1e-10))
    sheaf_p = float(np.mean(null_norms >= observed))

    print(f"\n  CONSISTENCY STATISTIC:")
    print(f"    Observed: {observed:.6f}")
    print(f"    Null: {null_norms.mean():.6f} ± {null_norms.std():.6f}")
    print(f"    z = {sheaf_z:.2f}")
    print(f"    p = {sheaf_p:.4f}")

    # Phase 4: Cochran's Q on each correlation pair
    print(f"\n  Cochran's Q on individual correlation pairs:")
    q_results = {}
    best_q = {"Q": 0, "pair": "", "p": 1}
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            r_vals = []
            for site in sites:
                r_vals.append(site_corrs[site]["corr_matrix"][i, j])
            r_vals = np.array(r_vals)
            r_clipped = np.clip(r_vals, -0.999, 0.999)
            z_vals = np.arctanh(r_clipped)
            z_mean = np.mean(z_vals)
            Q = float(np.sum((z_vals - z_mean) ** 2))
            p = float(1 - stats.chi2.cdf(Q, df=n_sites - 1))
            pair_name = f"{binary_vars[i]} × {binary_vars[j]}"
            q_results[pair_name] = {"Q": Q, "p": p, "mean_r": float(np.mean(r_vals)),
                                     "sd_r": float(np.std(r_vals))}
            sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
            print(f"    {pair_name:<35s}: Q={Q:7.2f}, p={p:.4f}{sig}")
            if Q > best_q["Q"]:
                best_q = {"Q": Q, "pair": pair_name, "p": p}

    # Phase 5: Interaction heterogeneity via chunked logistic regression
    # Need to re-read data for per-site logistic regressions
    print(f"\n  Computing per-site interaction coefficients (chunked)...")

    # Accumulate per-site data for logistic regression
    # For interaction coefficients we need the raw data, not just sufficient statistics
    # Use a simpler approach: accumulate contingency tables per site
    # Actually, we need logistic regression. Let's do a second pass but only keep
    # the columns we need and only for valid sites
    site_data = {s: [] for s in valid_sites if s in site_corrs}

    reader2 = pd.read_csv(CSV_PATH, usecols=USECOLS, chunksize=CHUNK_SIZE,
                          low_memory=False, encoding="utf-8-sig")

    for raw_chunk in tqdm(reader2, desc="Reading for interactions"):
        chunk = process_chunk(raw_chunk)
        for site in site_data:
            sub = chunk[chunk["site"] == site][["died", "age_70plus", "has_comorbidity", "male"]]
            if len(sub) > 0:
                site_data[site].append(sub)

    # Fit per-site logistic regressions
    import statsmodels.api as sm

    site_interactions = {}
    for site in tqdm(sorted(site_data.keys()), desc="Fitting interactions"):
        if not site_data[site]:
            continue
        df_site = pd.concat(site_data[site], ignore_index=True).dropna()
        if len(df_site) < 100 or df_site["died"].nunique() < 2:
            continue

        try:
            # age × comorbidity interaction
            df_site["age_x_comor"] = df_site["age_70plus"] * df_site["has_comorbidity"]
            X = sm.add_constant(df_site[["age_70plus", "has_comorbidity", "male", "age_x_comor"]])
            y = df_site["died"]
            m = sm.Logit(y, X).fit(disp=0, maxiter=50)

            site_interactions[site] = {
                "age_x_comorbidity_beta": float(m.params["age_x_comor"]),
                "age_x_comorbidity_se": float(m.bse["age_x_comor"]),
                "age_x_comorbidity_p": float(m.pvalues["age_x_comor"]),
                "age_x_comorbidity_or": float(np.exp(m.params["age_x_comor"])),
                "n": len(df_site),
            }
        except Exception as e:
            print(f"    {site}: logistic regression failed ({e})")

    print(f"\n  Per-site age × comorbidity interaction:")
    betas = []
    for site in sorted(site_interactions.keys()):
        info = site_interactions[site]
        sig = " *" if info["age_x_comorbidity_p"] < 0.05 else ""
        print(f"    {site:<30s}: OR = {info['age_x_comorbidity_or']:.3f}, "
              f"β = {info['age_x_comorbidity_beta']:+.4f}, "
              f"p = {info['age_x_comorbidity_p']:.4f}{sig}")
        betas.append(info["age_x_comorbidity_beta"])

    # Interaction heterogeneity norm
    betas = np.array(betas)
    observed_bracket = float(np.sqrt(np.sum((betas - betas.mean()) ** 2)))

    null_brackets = []
    rng2 = np.random.default_rng(42)
    for _ in range(200):
        perm = betas.copy()
        rng2.shuffle(perm)
        null_brackets.append(float(np.sqrt(np.sum((perm - perm.mean()) ** 2))))

    null_brackets = np.array(null_brackets)
    bracket_z = float((observed_bracket - null_brackets.mean()) / max(null_brackets.std(), 1e-10))
    bracket_p = float(np.mean(null_brackets >= observed_bracket))

    print(f"\n  INTERACTION HETEROGENEITY:")
    print(f"    Observed norm: {observed_bracket:.6f}")
    print(f"    Null: {null_brackets.mean():.6f} ± {null_brackets.std():.6f}")
    print(f"    z = {bracket_z:.2f}")
    print(f"    p = {bracket_p:.4f}")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "CDC Case Surveillance (chunked, age_70plus)",
        "n_total_rows": total_rows,
        "n_valid_sites": len(site_corrs),
        "sites": {s: {"n": site_counts[s]} for s in sorted(site_corrs.keys())},
        "consistency_statistic": {
            "observed": float(observed),
            "null_mean": float(null_norms.mean()),
            "null_std": float(null_norms.std()),
            "z_score": sheaf_z,
            "p_value": sheaf_p,
            "n_sites": n_sites,
            "n_perms": 200,
        },
        "best_cochran_q": best_q,
        "cochran_q_all": q_results,
        "interaction_heterogeneity": {
            "observed_norm": observed_bracket,
            "null_mean": float(null_brackets.mean()),
            "null_std": float(null_brackets.std()),
            "z_score": bracket_z,
            "p_value": bracket_p,
            "n_sites": len(site_interactions),
            "per_site": site_interactions,
        },
    }

    out_path = RESULTS_DIR / "cdc_geometric_full.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print summary for paper placeholders
    print(f"\n{'='*70}")
    print("PAPER PLACEHOLDER VALUES:")
    print(f"  SHEAF_Z  = {sheaf_z:.1f}")
    print(f"  SHEAF_P  = {sheaf_p}")
    print(f"  BRACKET_Z = {bracket_z:.1f}")
    print(f"  BRACKET_P = {bracket_p}")
    print(f"  Q_BEST   = {best_q['Q']:.1f} ({best_q['pair']})")
    print(f"  Q_P      = {best_q['p']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
