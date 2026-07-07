"""
Individual-level analysis of Zenodo NY hospital COVID-19 data.

This is the prototype showing what the paper gains from individual-level data:
- Individual-level age × mortality interaction (vs 4CE ecological version)
- Cross-site heterogeneity with only 2 hospitals
- Comparison: ecological (site-level) vs individual-level estimates

Usage:
    cd experiments/visweswaran/data/zenodo_ny_hospitals
    uv run python analyze_individual.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(Path(__file__).parent / "demographics_both_hospitals.csv")
    df["died"] = (df["outcome"] == 1).astype(int)
    df["age_80plus"] = (df["age"] >= 80).astype(int)
    df["age_group"] = pd.cut(df["age"], bins=[0, 50, 69, 79, 120],
                              labels=["<50", "50-69", "70-79", "80+"])
    df["n_comorbidities"] = df[["hypertension", "hyperlipidemia", "diabetes",
                                 "coronary.artery.disease", "chf",
                                 "cerebrovascular.disease", "hepatitis",
                                 "endstage.renal.disease", "chronic.kidney.disease",
                                 "asthma", "copd", "dementia", "cancer"]].sum(axis=1)
    return df


def individual_level_analysis(df):
    """Run individual-level logistic regression: mortality ~ age + hospital + age×hospital."""
    from scipy.optimize import minimize

    print(f"\n{'='*60}")
    print("INDIVIDUAL-LEVEL ANALYSIS (n={})".format(len(df)))
    print(f"{'='*60}")

    print(f"\n  Hospitals: {df['hospital'].unique().tolist()}")
    print(f"  Patients per hospital:")
    for h, sub in df.groupby("hospital"):
        print(f"    {h}: n={len(sub)}, died={sub['died'].sum()} ({sub['died'].mean()*100:.1f}%)")

    print(f"\n  Mortality by age group:")
    for ag, sub in df.groupby("age_group", observed=True):
        print(f"    {ag}: n={len(sub)}, died={sub['died'].sum()} ({sub['died'].mean()*100:.1f}%)")

    print(f"\n  Age × mortality by hospital:")
    for h in df["hospital"].unique():
        sub = df[df["hospital"] == h]
        for ag, agsub in sub.groupby("age_group", observed=True):
            if len(agsub) > 0:
                print(f"    {h} / {ag}: n={len(agsub)}, died={agsub['died'].sum()} ({agsub['died'].mean()*100:.1f}%)")

    df_model = df.copy()
    df_model["male"] = (df_model["sex"] == "male").astype(int)
    df_model["is_suny"] = (df_model["hospital"] == "SUNY").astype(int)
    df_model["age_decade"] = df_model["age"] / 10

    print(f"\n  --- Logistic regression: died ~ age + sex + n_comorbidities + hospital ---")
    m1 = _fit_logistic("died ~ age_decade + male + n_comorbidities + is_suny", df_model)
    print(m1.summary2().tables[1].to_string())

    print(f"\n  --- With age × hospital interaction ---")
    m2 = _fit_logistic("died ~ age_decade * is_suny + male + n_comorbidities", df_model)
    print(m2.summary2().tables[1].to_string())

    int_row = m2.params.get("age_decade:is_suny", None)
    interaction_result = None
    if int_row is not None:
        interaction_result = {
            "beta": float(m2.params["age_decade:is_suny"]),
            "se": float(m2.bse["age_decade:is_suny"]),
            "p": float(m2.pvalues["age_decade:is_suny"]),
            "or": float(np.exp(m2.params["age_decade:is_suny"])),
        }
        print(f"\n  Age × hospital interaction: OR={interaction_result['or']:.3f}, p={interaction_result['p']:.4f}")

    return {
        "n_patients": len(df),
        "n_hospitals": df["hospital"].nunique(),
        "mortality_rate": float(df["died"].mean()),
        "age_interaction": interaction_result,
    }


def ecological_vs_individual(df):
    """Compare ecological (hospital-level) vs individual-level estimates."""
    print(f"\n{'='*60}")
    print("ECOLOGICAL vs INDIVIDUAL COMPARISON")
    print(f"{'='*60}")

    print(f"\n  --- Ecological (hospital-level, like 4CE) ---")
    eco = df.groupby("hospital").agg(
        prop_died=("died", "mean"),
        prop_80plus=("age_80plus", "mean"),
        prop_male=("sex", lambda x: (x == "male").mean()),
        mean_comorbidities=("n_comorbidities", "mean"),
        n=("died", "count"),
    ).reset_index()

    for _, row in eco.iterrows():
        print(f"    {row['hospital']}: died={row['prop_died']:.3f}, "
              f"80+={row['prop_80plus']:.3f}, male={row['prop_male']:.3f}, "
              f"comorbidities={row['mean_comorbidities']:.2f}, n={row['n']}")

    eco_diff = eco.iloc[1]["prop_died"] - eco.iloc[0]["prop_died"]
    age_diff = eco.iloc[1]["prop_80plus"] - eco.iloc[0]["prop_80plus"]
    print(f"\n    Ecological mortality difference: {eco_diff:+.3f}")
    print(f"    Ecological age-80+ difference: {age_diff:+.3f}")

    if age_diff != 0:
        eco_slope = eco_diff / age_diff
        print(f"    Naive ecological slope (mortality / age-80+): {eco_slope:+.3f}")

    print(f"\n  --- Individual-level ---")
    df_model = df.copy()
    df_model["male"] = (df_model["sex"] == "male").astype(int)

    m_unadj = _fit_logistic("died ~ age_80plus", df_model)
    or_age80 = np.exp(m_unadj.params["age_80plus"])
    p_unadj = m_unadj.pvalues["age_80plus"]
    print(f"    Individual-level age-80+ OR: {or_age80:.2f} (p={p_unadj:.4f})")

    m_adj = _fit_logistic("died ~ age_80plus + male + n_comorbidities", df_model)
    or_adj = np.exp(m_adj.params["age_80plus"])
    p_adj = m_adj.pvalues["age_80plus"]
    print(f"    Adjusted age-80+ OR: {or_adj:.2f} (p={p_adj:.4f})")

    result = {
        "ecological": {
            "mortality_diff": float(eco_diff),
            "age80_diff": float(age_diff),
        },
        "individual_unadjusted": {
            "or_age80": float(or_age80),
            "p": float(p_unadj),
        },
        "individual_adjusted": {
            "or_age80": float(or_adj),
            "p": float(p_adj),
        },
    }

    with open(RESULTS_DIR / "ecological_vs_individual.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved ecological_vs_individual.json")

    return result


def comorbidity_analysis(df):
    """Comorbidity-level analysis: which comorbidities differ between hospitals?"""
    print(f"\n{'='*60}")
    print("COMORBIDITY HETEROGENEITY ACROSS HOSPITALS")
    print(f"{'='*60}")

    comorbidity_cols = ["hypertension", "hyperlipidemia", "diabetes",
                        "coronary.artery.disease", "chf",
                        "cerebrovascular.disease", "hepatitis",
                        "endstage.renal.disease", "chronic.kidney.disease",
                        "asthma", "copd", "dementia", "cancer"]

    print(f"\n  {'Comorbidity':<30s} {'SUNY':>8s} {'Maimo':>8s} {'χ² p':>10s}")
    print(f"  {'-'*58}")

    for col in comorbidity_cols:
        suny = df[df["hospital"] == "SUNY"][col]
        maimo = df[df["hospital"] == "Maimonides"][col]
        ct = pd.crosstab(df["hospital"], df[col])
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
        else:
            p = 1.0
        sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
        print(f"  {col:<30s} {suny.mean():8.3f} {maimo.mean():8.3f} {p:10.4f}{sig}")


def _fit_logistic(formula, df):
    """Fit logistic regression using statsmodels."""
    import statsmodels.formula.api as smf
    model = smf.logit(formula, data=df).fit(disp=0)
    return model


if __name__ == "__main__":
    print(f"Zenodo NY hospitals individual-level analysis — {datetime.now(timezone.utc).isoformat()}")

    df = load_data()
    print(f"\nLoaded {len(df)} patients from {df['hospital'].nunique()} hospitals")
    print(f"  Age range: {df['age'].min()}-{df['age'].max()}")
    print(f"  Overall mortality: {df['died'].mean()*100:.1f}%")

    individual_level_analysis(df)
    ecological_vs_individual(df)
    comorbidity_analysis(df)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results in: {RESULTS_DIR}")
    print(f"{'='*60}")
