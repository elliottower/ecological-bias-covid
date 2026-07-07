"""
Conditional independence testing using kernel CIT (RCIT) and Fisher's z.

RCIT (Strobl et al., J Causal Inference 2019) uses random Fourier features
to approximate a kernel-based conditional independence test. When R + RCIT
are unavailable, we fall back to causal-learn's kernel CI test (KCI).

Fisher's z-test is the standard linear partial correlation CI test — the
baseline that RCIT must beat on nonlinear dependencies.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def fishers_z_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: list[str],
) -> dict:
    """Fisher's z-test for conditional independence via partial correlation.

    Tests H0: X _||_ Y | Z using partial correlation + Fisher's z transform.
    """
    n = len(data)
    k = len(conditioning_set)

    if k == 0:
        r = data[x].corr(data[y])
    else:
        from sklearn.linear_model import LinearRegression
        Z = data[conditioning_set].values
        reg_x = LinearRegression().fit(Z, data[x].values)
        reg_y = LinearRegression().fit(Z, data[y].values)
        resid_x = data[x].values - reg_x.predict(Z)
        resid_y = data[y].values - reg_y.predict(Z)
        r = np.corrcoef(resid_x, resid_y)[0, 1]

    r = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - k - 3)
    z_stat = abs(z) / se
    p_value = 2 * (1 - stats.norm.cdf(z_stat))

    return {
        "method": "fisher_z",
        "statistic": z_stat,
        "p_value": p_value,
        "partial_corr": r,
        "n": n,
        "df": n - k - 3,
    }


def kci_test(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: list[str],
) -> dict:
    """Kernel conditional independence test via causal-learn.

    This is the Python fallback when R/RCIT is unavailable. Uses the same
    kernel approach as RCIT but the causal-learn implementation.
    """
    from causallearn.utils.cit import CIT

    all_vars = [x, y] + conditioning_set
    sub = data[all_vars].values
    var_names = list(range(len(all_vars)))

    cit = CIT(sub, method="kci")
    p_value = cit(0, 1, list(range(2, len(all_vars))))

    return {
        "method": "kci",
        "statistic": float("nan"),
        "p_value": float(p_value),
        "n": len(data),
    }


def rcit_test_via_r(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: list[str],
) -> dict | None:
    """Run RCIT via R subprocess. Returns None if R/RCIT unavailable."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(RCIT)"],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        cols = [x, y] + conditioning_set
        data[cols].to_csv(f.name, index=False)
        csv_path = f.name

    n_cond = len(conditioning_set)
    r_script = f"""
    library(RCIT)
    d <- read.csv("{csv_path}")
    x_col <- d[,1]
    y_col <- d[,2]
    if ({n_cond} > 0) {{
        z_cols <- as.matrix(d[,3:ncol(d)])
        result <- RCIT(x_col, y_col, z_cols)
    }} else {{
        result <- RIT(x_col, y_col)
    }}
    cat(result$p, "\\n")
    cat(result$Sta, "\\n")
    """

    try:
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        p_value = float(lines[0].split()[-1])
        statistic = float(lines[1].split()[-1]) if len(lines) > 1 else float("nan")
        return {
            "method": "rcit",
            "statistic": statistic,
            "p_value": p_value,
            "n": len(data),
        }
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None
    finally:
        Path(csv_path).unlink(missing_ok=True)


def test_all_implications(
    data: pd.DataFrame,
    implications: list[dict],
    alpha: float = 0.05,
    use_rcit: bool = True,
) -> pd.DataFrame:
    """Test all CI implications with both Fisher's z and kernel CIT.

    Returns a DataFrame with one row per implication, columns for each method's
    p-value and verdict.
    """
    results = []
    for imp in tqdm(implications, desc="Testing CI implications"):
        x, y = imp["X"], imp["Y"]
        cond = imp["conditioning_set"]

        fisher = fishers_z_test(data, x, y, cond)

        kernel = None
        if use_rcit:
            kernel = rcit_test_via_r(data, x, y, cond)
        if kernel is None:
            try:
                kernel = kci_test(data, x, y, cond)
            except Exception:
                kernel = {"method": "kci_failed", "p_value": float("nan")}

        results.append({
            "X": x,
            "Y": y,
            "conditioning_set": "|".join(sorted(cond)) if cond else "{}",
            "type": imp.get("type", ""),
            "fisher_z_pval": fisher["p_value"],
            "fisher_z_stat": fisher["statistic"],
            "fisher_partial_corr": fisher.get("partial_corr", float("nan")),
            "fisher_verdict": "independent" if fisher["p_value"] > alpha else "dependent",
            "kernel_method": kernel["method"],
            "kernel_pval": kernel["p_value"],
            "kernel_verdict": (
                "independent" if kernel["p_value"] > alpha else "dependent"
            ),
            "methods_agree": (
                (fisher["p_value"] > alpha) == (kernel["p_value"] > alpha)
            ),
        })

    return pd.DataFrame(results)


def apply_fdr_correction(
    results: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """Apply Benjamini-Yekutieli FDR correction (Visweswaran's PCp contribution)."""
    from scipy.stats import false_discovery_control

    results = results.copy()

    for col in ["fisher_z_pval", "kernel_pval"]:
        pvals = results[col].values
        valid = ~np.isnan(pvals)
        if valid.sum() == 0:
            results[f"{col}_fdr"] = np.nan
            results[f"{col.replace('pval', 'verdict')}_fdr"] = "na"
            continue

        adjusted = np.full_like(pvals, np.nan)
        adjusted[valid] = false_discovery_control(pvals[valid], method="by")
        results[f"{col}_fdr"] = adjusted
        verdict_col = col.replace("pval", "verdict") + "_fdr"
        results[verdict_col] = np.where(
            np.isnan(adjusted), "na",
            np.where(adjusted > alpha, "independent", "dependent"),
        )

    return results


if __name__ == "__main__":
    from src.extract_dag import build_assumed_dag, get_testable_implications
    from src.generate_synthetic import generate_known_truth

    print("=== CI Testing on Known-Truth Data ===")
    df, ground_truth = generate_known_truth(n_samples=2000, seed=42)
    G = build_assumed_dag()
    implications = get_testable_implications(G)

    print(f"Testing {len(implications)} implications...")
    results = test_all_implications(df, implications[:20], use_rcit=True)
    results = apply_fdr_correction(results)
    print(results[["X", "Y", "conditioning_set", "fisher_verdict", "kernel_verdict"]].to_string())
