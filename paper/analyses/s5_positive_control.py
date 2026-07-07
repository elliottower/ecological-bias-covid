"""
Supplementary Analysis S5: Reduced-Confounding Positive Control

Runs ecological regression with sex ratio (proportion male) as predictor
of site-level mortality on both CDC and Mexico. Sex has a known
individual-level effect (male OR=1.7) but its cross-site confounding
structure differs from age.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_ecological_regression(x, y, dataset_name, predictor_name):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "dataset": dataset_name,
        "predictor": predictor_name,
        "n_sites": len(x),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "se": float(std_err),
        "significant": bool(p_value < 0.05),
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S5: Reduced-Confounding Positive Control  [{ts}]")
    print("=" * 70)

    with open(DATA_DIR / "cdc_case_surveillance/results/cdc_ecological_fallacy.json") as f:
        cdc = json.load(f)
    with open(DATA_DIR / "mexico_covid/mexico_ecological_results.json") as f:
        mexico = json.load(f)

    cdc_sites = cdc["site_data"]
    cdc_male = [s["prop_male"] for s in cdc_sites]
    cdc_mort = [s["prop_died"] for s in cdc_sites]
    cdc_elderly = [s["prop_80plus"] for s in cdc_sites]

    mx_sites = mexico["ecological_regression"]["site_details"]
    mx_male = [v["prop_male"] for v in mx_sites.values()]
    mx_mort = [v["mortality_rate"] for v in mx_sites.values()]
    mx_elderly = [v["prop_elderly"] for v in mx_sites.values()]

    results = {}

    print("\n--- CDC ---")
    r = run_ecological_regression(cdc_male, cdc_mort, "CDC", "sex_ratio")
    results["cdc_sex"] = r
    print(f"  Sex → mortality: slope={r['slope']:+.4f}, p={r['p_value']:.4f}, "
          f"R²={r['r_squared']:.3f} {'*' if r['significant'] else 'ns'}")

    r = run_ecological_regression(cdc_elderly, cdc_mort, "CDC", "elderly_proportion")
    results["cdc_elderly"] = r
    print(f"  Age → mortality: slope={r['slope']:+.4f}, p={r['p_value']:.4f}, "
          f"R²={r['r_squared']:.3f} {'*' if r['significant'] else 'ns'}")

    print("\n--- Mexico ---")
    r = run_ecological_regression(mx_male, mx_mort, "Mexico", "sex_ratio")
    results["mexico_sex"] = r
    print(f"  Sex → mortality: slope={r['slope']:+.4f}, p={r['p_value']:.4f}, "
          f"R²={r['r_squared']:.3f} {'*' if r['significant'] else 'ns'}")

    r = run_ecological_regression(mx_elderly, mx_mort, "Mexico", "elderly_proportion")
    results["mexico_elderly"] = r
    print(f"  Age → mortality: slope={r['slope']:+.4f}, p={r['p_value']:.4f}, "
          f"R²={r['r_squared']:.3f} {'*' if r['significant'] else 'ns'}")

    print("\n--- Cross-site correlations (CDC) ---")
    corr_em = np.corrcoef(cdc_elderly, cdc_male)[0, 1]
    corr_ec = np.corrcoef(cdc_elderly,
                           [s["prop_comorbidity"] for s in cdc_sites])[0, 1]
    corr_mc = np.corrcoef(cdc_male,
                           [s["prop_comorbidity"] for s in cdc_sites])[0, 1]
    print(f"  rho(elderly, male) = {corr_em:+.3f}")
    print(f"  rho(elderly, comorbidity) = {corr_ec:+.3f}")
    print(f"  rho(male, comorbidity) = {corr_mc:+.3f}")

    cdc_sex_sig = results["cdc_sex"]["significant"]
    cdc_age_sig = results["cdc_elderly"]["significant"]

    if cdc_sex_sig and not cdc_age_sig:
        conclusion = ("Sex-mortality ecological regression succeeds where "
                       "age-mortality fails on CDC. The failure is specific "
                       "to age's confounding structure.")
    elif not cdc_sex_sig and not cdc_age_sig:
        conclusion = ("FALSIFICATION: Sex-mortality ecological regression "
                       "also fails on CDC. The pseudo-site construction "
                       "may be too noisy for any ecological regression.")
    elif cdc_sex_sig and cdc_age_sig:
        conclusion = ("Both sex and age ecological regressions are "
                       "significant on CDC. No differential failure.")
    else:
        conclusion = ("Age significant but sex not — unexpected pattern.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "results": results,
        "cross_site_correlations_cdc": {
            "elderly_male": float(corr_em),
            "elderly_comorbidity": float(corr_ec),
            "male_comorbidity": float(corr_mc),
        },
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s5_positive_control.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
