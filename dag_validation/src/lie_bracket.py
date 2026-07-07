"""
Lie bracket norm on cross-site effect transport maps.

The effect surface theta(s) maps site characteristics s to effect vectors
theta = (log_pmi_cns, log_pmi_pns). The Lie bracket [dtheta/ds_a, dtheta/ds_b]
of two characteristic directions measures non-commutativity of adjusting for
those characteristics sequentially. For smooth functions the mixed partials
commute (Schwarz's theorem), so the empirical Lie bracket is zero iff the
effect surface is additive in the characteristics. Nonzero bracket norm
indicates interaction/effect modification: the order of covariate adjustment
matters, a geometric obstruction to naive pooling.

In practice this reduces to testing interaction terms beta_{ab} in the
weighted meta-regression theta = alpha_0 + alpha_a * x_a + alpha_b * x_b
+ beta_{ab} * x_a * x_b. The Lie bracket norm is the Frobenius norm of the
interaction matrix B = [beta_{ab}], tested via permutation.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data" / "4ce_neuro"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_site_vectors(max_se: float = 0.5) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build site-level effect and characteristic vectors.

    Returns (df, effect_cols, char_cols) where df has one row per site with
    both effect estimates and site characteristics.
    """
    pmi = pd.read_csv(DATA_DIR / "site_pmi_effects.csv")
    demo = pd.read_csv(DATA_DIR / "site_demographics.csv")

    effects = []
    for eff in ["pmi.cns", "pmi.pns"]:
        for tp in [30, 60, 90]:
            sub = pmi[(pmi["effect_name"] == eff) & (pmi["timepoint"] == tp) & (pmi["se"] < max_se)]
            col = f"{eff.replace('.', '_')}_{tp}"
            sub = sub[["site", "log_pmi"]].rename(columns={"log_pmi": col})
            effects.append(sub)

    eff_df = effects[0]
    for e in effects[1:]:
        eff_df = eff_df.merge(e, on="site", how="outer")

    none_demo = demo[demo["neuro_type"].isna()].copy()
    char_cols = ["prop_female", "prop_age_80plus", "prop_severe", "prop_deceased"]
    available_chars = [c for c in char_cols if c in none_demo.columns]

    site_df = eff_df.merge(none_demo[["site"] + available_chars], on="site", how="inner")

    effect_cols = [c for c in site_df.columns if c.startswith("pmi_")]
    site_df = site_df.dropna(subset=effect_cols[:2] + available_chars[:2])

    return site_df, effect_cols, available_chars


def weighted_interaction_regression(
    y: np.ndarray, x1: np.ndarray, x2: np.ndarray, w: np.ndarray
) -> dict:
    """Fit weighted regression y = a0 + a1*x1 + a2*x2 + b12*x1*x2 + e.

    Returns the interaction coefficient b12 and its standard error.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x1, x2, x1 * x2])
    W = np.diag(w)

    XtWX = X.T @ W @ X
    try:
        inv_XtWX = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return {"beta_interaction": 0.0, "se": np.inf, "t": 0.0, "p": 1.0}

    beta = inv_XtWX @ (X.T @ W @ y)
    resid = y - X @ beta
    sigma2 = np.sum(w * resid**2) / max(n - 4, 1)
    cov_beta = sigma2 * inv_XtWX

    b12 = beta[3]
    se_b12 = np.sqrt(max(0, cov_beta[3, 3]))
    t_stat = b12 / se_b12 if se_b12 > 0 else 0.0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - 4, 1)))

    return {"beta_interaction": float(b12), "se": float(se_b12), "t": float(t_stat), "p": float(p_val)}


def compute_lie_bracket_norm(
    site_df: pd.DataFrame, effect_cols: list[str], char_cols: list[str]
) -> dict:
    """Compute the Lie bracket norm as the Frobenius norm of the interaction matrix.

    For each (effect, char_pair) combination, fits the weighted interaction
    regression and collects beta_{ab}. The Lie bracket norm is ||B||_F.
    """
    pmi = pd.read_csv(DATA_DIR / "site_pmi_effects.csv")

    results = []
    B_matrix = []

    for eff_col in effect_cols:
        sub = site_df.dropna(subset=[eff_col])
        if len(sub) < 6:
            continue

        parts = eff_col.split("_")
        eff_name = f"{parts[0]}.{parts[1]}"
        tp = int(parts[2])
        pmi_sub = pmi[(pmi["effect_name"] == eff_name) & (pmi["timepoint"] == tp)]
        se_map = dict(zip(pmi_sub["site"], pmi_sub["se"]))
        se_vals = sub["site"].map(se_map).fillna(1.0).values
        w = 1.0 / (se_vals**2)

        y = sub[eff_col].values

        for ca, cb in combinations(char_cols, 2):
            sub_clean = sub.dropna(subset=[ca, cb, eff_col])
            if len(sub_clean) < 6:
                continue

            x1 = sub_clean[ca].values
            x2 = sub_clean[cb].values
            y_clean = sub_clean[eff_col].values
            se_clean = sub_clean["site"].map(se_map).fillna(1.0).values
            w_clean = 1.0 / (se_clean**2)

            reg = weighted_interaction_regression(y_clean, x1, x2, w_clean)
            reg["effect"] = eff_col
            reg["char_a"] = ca
            reg["char_b"] = cb
            reg["n_sites"] = len(sub_clean)
            results.append(reg)
            B_matrix.append(reg["beta_interaction"])

    B_norm = np.sqrt(np.sum(np.array(B_matrix)**2)) if B_matrix else 0.0

    return {
        "interactions": results,
        "bracket_norm": float(B_norm),
        "n_interactions": len(B_matrix),
    }


def permutation_test(
    site_df: pd.DataFrame, effect_cols: list[str], char_cols: list[str],
    n_perms: int = 1000
) -> dict:
    """Test whether the Lie bracket norm exceeds chance by permutation.

    Permutes site labels (shuffles the mapping between characteristics and effects),
    re-computes the bracket norm, and builds a null distribution.
    """
    observed = compute_lie_bracket_norm(site_df, effect_cols, char_cols)
    obs_norm = observed["bracket_norm"]

    null_norms = []
    rng = np.random.default_rng(42)

    for _ in tqdm(range(n_perms), desc="Permutation test"):
        perm_df = site_df.copy()
        perm_idx = rng.permutation(len(perm_df))
        for cc in char_cols:
            perm_df[cc] = perm_df[cc].values[perm_idx]

        perm_result = compute_lie_bracket_norm(perm_df, effect_cols, char_cols)
        null_norms.append(perm_result["bracket_norm"])

    null_norms = np.array(null_norms)
    p_value = float(np.mean(null_norms >= obs_norm))
    z_score = float((obs_norm - np.mean(null_norms)) / np.std(null_norms)) if np.std(null_norms) > 0 else 0.0

    return {
        "observed_norm": obs_norm,
        "null_mean": float(np.mean(null_norms)),
        "null_std": float(np.std(null_norms)),
        "z_score": z_score,
        "p_value": p_value,
        "n_perms": n_perms,
        "interactions": observed["interactions"],
    }


def run_lie_bracket_analysis(n_perms: int = 1000) -> dict:
    """Run the full Lie bracket analysis."""
    print(f"\n{'='*60}")
    print("LIE BRACKET NORM ON CROSS-SITE EFFECT TRANSPORT")
    print(f"{'='*60}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    site_df, effect_cols, char_cols = build_site_vectors(max_se=0.5)
    print(f"\nSites with valid effects: {len(site_df)}")
    print(f"Effect dimensions: {effect_cols}")
    print(f"Characteristic dimensions: {char_cols}")

    print(f"\nRunning permutation test ({n_perms} permutations)...")
    result = permutation_test(site_df, effect_cols, char_cols, n_perms=n_perms)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Observed Lie bracket norm: {result['observed_norm']:.4f}")
    print(f"  Null distribution: {result['null_mean']:.4f} +/- {result['null_std']:.4f}")
    print(f"  z-score: {result['z_score']:.2f}")
    print(f"  p-value: {result['p_value']:.4f}")

    print(f"\n  Interaction terms (components of bracket):")
    for r in result["interactions"]:
        sig = "*" if r["p"] < 0.05 else ""
        print(f"    {r['effect']:15s} x ({r['char_a']:15s}, {r['char_b']:15s}): "
              f"beta = {r['beta_interaction']:+.4f}, p = {r['p']:.4f}{sig}")

    interactions_df = pd.DataFrame(result["interactions"])
    interactions_df.to_csv(tables_dir / "T9_lie_bracket.csv", index=False)
    print(f"\n  Saved T9_lie_bracket.csv")

    summary = {
        "bracket_norm": result["observed_norm"],
        "z_score": result["z_score"],
        "p_value": result["p_value"],
        "null_mean": result["null_mean"],
        "null_std": result["null_std"],
        "n_perms": n_perms,
        "n_sites": len(site_df),
        "n_interactions": len(result["interactions"]),
    }

    import json
    with open(tables_dir / "T9_lie_bracket_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    if result["p_value"] < 0.05:
        print("  The Lie bracket norm is SIGNIFICANT (p < 0.05).")
        print("  Effect transport between sites is non-commutative: the order")
        print("  of covariate adjustment matters. This is a geometric signature")
        print("  of effect modification that scalar heterogeneity tests miss.")
    else:
        print("  The Lie bracket norm is NOT significant (p >= 0.05).")
        print("  Effect transport is approximately commutative: the effect surface")
        print("  is additive in site characteristics, and pooling order does not matter.")
        print("  Standard meta-analysis assumptions are geometrically justified.")

    return result


if __name__ == "__main__":
    run_lie_bracket_analysis(n_perms=1000)
