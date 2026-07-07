"""
Individual-level analysis of CDC Case Surveillance COVID-19 data.

6.9M cases with age group, sex, race, hospitalization, ICU, death,
and comorbidity flag. No geography in this version, but state-level
grouping could come from the geo version.

This demonstrates the ecological fallacy analysis at scale:
- Individual-level age × mortality logistic regression
- Comparison with what site-level ecological analysis would find

Usage:
    cd experiments/visweswaran/data/cdc_case_surveillance
    uv run python analyze_cdc.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import statsmodels.formula.api as smf

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    print(f"  Loading CDC data (this may take a moment)...")
    df = pd.read_csv(
        Path(__file__).parent / "cdc_full.csv",
        low_memory=False,
        encoding="utf-8-sig",
    )
    df.columns = df.columns.str.strip()

    print(f"  Loaded {len(df):,} rows")
    print(f"  Columns: {list(df.columns)}")

    df["died"] = (df["death_yn"] == "Yes").astype(int)
    df["hospitalized"] = (df["hosp_yn"] == "Yes").astype(int)
    df["icu"] = (df["icu_yn"] == "Yes").astype(int)
    df["has_comorbidity"] = (df["medcond_yn"] == "Yes").astype(int)

    age_map = {
        "0 - 9 Years": 5, "10 - 19 Years": 15, "20 - 29 Years": 25,
        "30 - 39 Years": 35, "40 - 49 Years": 45, "50 - 59 Years": 55,
        "60 - 69 Years": 65, "70 - 79 Years": 75, "80+ Years": 85,
    }
    df["age_midpoint"] = df["age_group"].map(age_map)
    df["age_80plus"] = (df["age_group"] == "80+ Years").astype(int)
    df["age_70plus"] = df["age_group"].isin(["70 - 79 Years", "80+ Years"]).astype(int)
    df["male"] = (df["sex"] == "Male").astype(int)

    return df


def descriptive_stats(df):
    print(f"\n{'='*60}")
    print("DESCRIPTIVE STATISTICS")
    print(f"{'='*60}")

    complete = df.dropna(subset=["age_midpoint", "death_yn"])
    print(f"\n  Total cases: {len(df):,}")
    print(f"  With age + death info: {len(complete):,}")
    print(f"  Overall mortality (among known): {complete['died'].mean()*100:.2f}%")

    print(f"\n  Age distribution:")
    for ag in sorted(df["age_group"].dropna().unique()):
        sub = df[df["age_group"] == ag]
        died_known = sub[sub["death_yn"].isin(["Yes", "No"])]
        mort = died_known["died"].mean() * 100 if len(died_known) > 0 else float("nan")
        print(f"    {ag:20s}: n={len(sub):>9,}  mortality={mort:5.2f}%")

    print(f"\n  Sex distribution:")
    for s in ["Male", "Female"]:
        sub = df[df["sex"] == s]
        died_known = sub[sub["death_yn"].isin(["Yes", "No"])]
        mort = died_known["died"].mean() * 100 if len(died_known) > 0 else float("nan")
        print(f"    {s:20s}: n={len(sub):>9,}  mortality={mort:5.2f}%")

    print(f"\n  Comorbidity:")
    for c in ["Yes", "No"]:
        sub = df[df["medcond_yn"] == c]
        died_known = sub[sub["death_yn"].isin(["Yes", "No"])]
        mort = died_known["died"].mean() * 100 if len(died_known) > 0 else float("nan")
        print(f"    medcond={c:3s}: n={len(sub):>9,}  mortality={mort:5.2f}%")


def individual_level_analysis(df):
    """Run individual-level logistic regression on the full dataset."""
    print(f"\n{'='*60}")
    print("INDIVIDUAL-LEVEL LOGISTIC REGRESSION")
    print(f"{'='*60}")

    complete = df.dropna(subset=["age_midpoint", "death_yn", "sex"]).copy()
    complete = complete[complete["death_yn"].isin(["Yes", "No"])]
    complete = complete[complete["sex"].isin(["Male", "Female"])]
    print(f"\n  Complete cases for regression: {len(complete):,}")

    print(f"\n  --- Model 1: died ~ age ---")
    m1 = smf.logit("died ~ age_midpoint", data=complete).fit(disp=0)
    or_age = np.exp(m1.params["age_midpoint"] * 10)
    print(f"    Age OR per decade: {or_age:.2f} (p={m1.pvalues['age_midpoint']:.2e})")

    print(f"\n  --- Model 2: died ~ age + sex + comorbidity ---")
    m2 = smf.logit("died ~ age_midpoint + male + has_comorbidity", data=complete).fit(disp=0)
    print(m2.summary2().tables[1].to_string())
    print(f"\n    Age OR per decade: {np.exp(m2.params['age_midpoint']*10):.2f}")
    print(f"    Male OR: {np.exp(m2.params['male']):.2f}")
    print(f"    Comorbidity OR: {np.exp(m2.params['has_comorbidity']):.2f}")

    print(f"\n  --- Model 3: died ~ age * comorbidity + sex ---")
    m3 = smf.logit("died ~ age_midpoint * has_comorbidity + male", data=complete).fit(disp=0)
    print(m3.summary2().tables[1].to_string())

    int_p = m3.pvalues.get("age_midpoint:has_comorbidity", 1.0)
    int_or = np.exp(m3.params.get("age_midpoint:has_comorbidity", 0) * 10)
    print(f"\n    Age × comorbidity interaction OR (per decade): {int_or:.3f} (p={int_p:.4f})")

    results = {
        "n_complete": len(complete),
        "age_or_per_decade_unadjusted": float(or_age),
        "age_or_per_decade_adjusted": float(np.exp(m2.params["age_midpoint"] * 10)),
        "male_or": float(np.exp(m2.params["male"])),
        "comorbidity_or": float(np.exp(m2.params["has_comorbidity"])),
        "age_x_comorbidity_interaction_or": float(int_or),
        "age_x_comorbidity_interaction_p": float(int_p),
        "model2_aic": float(m2.aic),
        "model3_aic": float(m3.aic),
    }

    with open(RESULTS_DIR / "cdc_individual_regression.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved cdc_individual_regression.json")

    return results


def ecological_simulation(df):
    """Simulate the ecological fallacy by aggregating to 'sites' and comparing.

    Use age_group as a natural grouping variable to create pseudo-sites,
    then show how site-level regression differs from individual-level.
    """
    print(f"\n{'='*60}")
    print("ECOLOGICAL FALLACY DEMONSTRATION")
    print(f"{'='*60}")

    complete = df.dropna(subset=["age_midpoint", "death_yn", "sex", "race_ethnicity_combined"]).copy()
    complete = complete[complete["death_yn"].isin(["Yes", "No"])]
    complete = complete[complete["sex"].isin(["Male", "Female"])]

    complete["race_site"] = complete["race_ethnicity_combined"].str[:20]
    site_groups = complete.groupby("race_site")
    valid_sites = [name for name, g in site_groups if len(g) >= 1000]

    print(f"\n  Using race/ethnicity as pseudo-sites ({len(valid_sites)} groups with n >= 1000)")

    eco_data = []
    for site in valid_sites:
        sub = complete[complete["race_site"] == site]
        eco_data.append({
            "site": site,
            "n": len(sub),
            "prop_died": sub["died"].mean(),
            "prop_80plus": sub["age_80plus"].mean(),
            "prop_70plus": sub["age_70plus"].mean(),
            "prop_male": sub["male"].mean(),
            "prop_comorbidity": sub["has_comorbidity"].mean(),
            "mean_age": sub["age_midpoint"].mean(),
        })

    eco_df = pd.DataFrame(eco_data)

    print(f"\n  Site-level summary:")
    print(f"  {'Site':<25s} {'n':>8s} {'mort%':>7s} {'80+%':>7s} {'male%':>7s} {'comor%':>7s}")
    print(f"  {'-'*62}")
    for _, row in eco_df.sort_values("prop_died", ascending=False).iterrows():
        print(f"  {row['site']:<25s} {row['n']:8,} {row['prop_died']*100:6.2f}% "
              f"{row['prop_80plus']*100:6.2f}% {row['prop_male']*100:6.2f}% "
              f"{row['prop_comorbidity']*100:6.2f}%")

    print(f"\n  --- Ecological regression: mortality ~ age_80plus proportion ---")
    if len(eco_df) >= 3:
        slope, intercept, r, p, se = stats.linregress(eco_df["prop_80plus"], eco_df["prop_died"])
        print(f"    Ecological slope: {slope:+.3f} (p={p:.4f}, R²={r**2:.3f})")
        print(f"    Interpretation: each 10pp increase in 80+ proportion → {slope*0.1*100:+.1f}pp mortality increase")

    print(f"\n  --- Individual-level regression (same data) ---")
    m_ind = smf.logit("died ~ age_80plus + male + has_comorbidity", data=complete).fit(disp=0)
    or_80 = np.exp(m_ind.params["age_80plus"])
    print(f"    Individual age-80+ OR: {or_80:.2f} (p={m_ind.pvalues['age_80plus']:.2e})")

    print(f"\n  --- The comparison ---")
    print(f"    Ecological slope suggests each 10pp age-80+ increase → {slope*0.1*100:+.1f}pp mortality")
    print(f"    Individual OR suggests 80+ patients have {or_80:.1f}x the odds of death")
    print(f"    These measure DIFFERENT things:")
    print(f"      Ecological = compositional + contextual effects (confounded)")
    print(f"      Individual = patient-level effect (adjusted)")

    results = {
        "n_sites": len(eco_df),
        "ecological_slope": float(slope),
        "ecological_p": float(p),
        "ecological_r2": float(r**2),
        "individual_or_80plus": float(or_80),
        "individual_p": float(m_ind.pvalues["age_80plus"]),
        "site_data": eco_data,
    }

    with open(RESULTS_DIR / "cdc_ecological_fallacy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved cdc_ecological_fallacy.json")

    return results


if __name__ == "__main__":
    print(f"CDC Case Surveillance analysis — {datetime.now(timezone.utc).isoformat()}")

    df = load_data()
    descriptive_stats(df)
    individual_level_analysis(df)
    ecological_simulation(df)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'='*60}")
