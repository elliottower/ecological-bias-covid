"""
Mexico COVID-19 Datos Abiertos: Ecological fallacy analysis.

Uses real hospital sites (ENTIDAD_UM = state of medical unit, 32 states)
to demonstrate ecological vs individual-level analysis — the same test
as the CDC pseudo-site analysis, but with genuine geographic sites.

Data dictionary (key columns):
  SEXO: 1=female, 2=male, 99=not specified
  EDAD: age in years
  TIPO_PACIENTE: 1=outpatient, 2=hospitalized
  INTUBADO: 1=yes, 2=no, 97/98/99=NA
  UCI: 1=yes, 2=no, 97/98/99=NA
  FECHA_DEF: death date (9999-99-99 = alive)
  ENTIDAD_UM: state code of medical unit (1-32)
  DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, CARDIOVASCULAR,
  OBESIDAD, RENAL_CRONICA, TABAQUISMO: 1=yes, 2=no, 98=unknown
  CLASIFICACION_FINAL: 1-3 = confirmed COVID-19
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
OUT_FILE = DATA_DIR / "mexico_ecological_results.json"

STATE_NAMES = {
    1: "Aguascalientes", 2: "Baja California", 3: "Baja California Sur",
    4: "Campeche", 5: "Coahuila", 6: "Colima", 7: "Chiapas",
    8: "Chihuahua", 9: "CDMX", 10: "Durango", 11: "Guanajuato",
    12: "Guerrero", 13: "Hidalgo", 14: "Jalisco", 15: "Mexico (state)",
    16: "Michoacan", 17: "Morelos", 18: "Nayarit", 19: "Nuevo Leon",
    20: "Oaxaca", 21: "Puebla", 22: "Queretaro", 23: "Quintana Roo",
    24: "San Luis Potosi", 25: "Sinaloa", 26: "Sonora", 27: "Tabasco",
    28: "Tamaulipas", 29: "Tlaxcala", 30: "Veracruz", 31: "Yucatan",
    32: "Zacatecas",
}


def load_data(csv_path: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Load and filter to confirmed COVID cases with known outcomes."""
    print(f"[{datetime.now().isoformat()}] Loading {csv_path} in chunks...")

    frames = []
    base_cols = [
        "SEXO", "EDAD", "TIPO_PACIENTE", "INTUBADO", "UCI",
        "FECHA_DEF", "ENTIDAD_UM", "DIABETES", "EPOC", "ASMA",
        "INMUSUPR", "HIPERTENSION", "CARDIOVASCULAR", "OBESIDAD",
        "RENAL_CRONICA", "TABAQUISMO",
    ]
    # Column name differs between 2022 and 2026 formats
    header = pd.read_csv(csv_path, nrows=0, encoding="latin-1").columns.tolist()
    if "CLASIFICACION_FINAL" in header:
        clasif_col = "CLASIFICACION_FINAL"
    else:
        clasif_col = "CLASIFICACION_FINAL_COVID"
    usecols = base_cols + [clasif_col]

    total_rows = 0
    kept_rows = 0

    for chunk in tqdm(
        pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize,
                     encoding="latin-1", low_memory=False),
        desc="Reading chunks"
    ):
        total_rows += len(chunk)
        confirmed = chunk[chunk[clasif_col].isin([1, 2, 3])].copy()
        kept_rows += len(confirmed)
        frames.append(confirmed)

    df = pd.concat(frames, ignore_index=True)
    print(f"[{datetime.now().isoformat()}] Total rows: {total_rows:,}, "
          f"confirmed COVID: {kept_rows:,}")
    return df


def prepare_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary analysis variables."""
    df["died"] = (df["FECHA_DEF"] != "9999-99-99").astype(int)
    df["male"] = (df["SEXO"] == 2).astype(int)
    df["age_70plus"] = (df["EDAD"] >= 70).astype(int)
    df["hospitalized"] = (df["TIPO_PACIENTE"] == 2).astype(int)
    df["icu"] = (df["UCI"] == 1).astype(int)

    # Comorbidities: 1 = yes, anything else = no
    comorbidity_cols = [
        "DIABETES", "HIPERTENSION", "OBESIDAD", "CARDIOVASCULAR",
        "RENAL_CRONICA", "EPOC", "ASMA", "INMUSUPR", "TABAQUISMO",
    ]
    for col in comorbidity_cols:
        df[col.lower()] = (df[col] == 1).astype(int)

    df["has_comorbidity"] = (
        df[[c.lower() for c in comorbidity_cols]].sum(axis=1) > 0
    ).astype(int)

    df["site"] = df["ENTIDAD_UM"].map(STATE_NAMES).fillna("Unknown")
    return df


def ecological_regression(df: pd.DataFrame) -> dict:
    """Site-level regression: proportion 70+ vs mortality rate."""
    site_stats = df.groupby("site").agg(
        n=("died", "size"),
        mortality_rate=("died", "mean"),
        prop_elderly=("age_70plus", "mean"),
        prop_male=("male", "mean"),
        prop_comorbidity=("has_comorbidity", "mean"),
    ).reset_index()

    # Only sites with n >= 1000
    site_stats = site_stats[site_stats["n"] >= 1000]
    print(f"\n  Sites with n >= 1000: {len(site_stats)}")

    slope, intercept, r, p, se = stats.linregress(
        site_stats["prop_elderly"], site_stats["mortality_rate"]
    )
    print(f"  Ecological regression: β = {slope:+.4f}, p = {p:.4f}, "
          f"R² = {r**2:.4f}")

    return {
        "n_sites": int(len(site_stats)),
        "beta": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r**2),
        "p_value": float(p),
        "se": float(se),
        "site_details": {
            row["site"]: {
                "n": int(row["n"]),
                "mortality_rate": float(row["mortality_rate"]),
                "prop_elderly": float(row["prop_elderly"]),
                "prop_male": float(row["prop_male"]),
                "prop_comorbidity": float(row["prop_comorbidity"]),
            }
            for _, row in site_stats.iterrows()
        },
    }


def individual_logistic(df: pd.DataFrame) -> dict:
    """Individual-level logistic regression: age → death."""
    import statsmodels.api as sm

    # Filter to complete cases
    cols = ["died", "age_70plus", "male", "has_comorbidity"]
    complete = df[cols].dropna()
    print(f"\n  Complete cases for logistic regression: {len(complete):,}")

    # Model 1: age only
    X1 = sm.add_constant(complete[["age_70plus"]])
    model1 = sm.Logit(complete["died"], X1).fit(disp=0)
    or_age = np.exp(model1.params["age_70plus"])
    p_age = model1.pvalues["age_70plus"]
    print(f"  Age 70+ OR = {or_age:.2f}, p = {p_age:.2e}")

    # Model 2: age + sex + comorbidity
    X2 = sm.add_constant(complete[["age_70plus", "male", "has_comorbidity"]])
    model2 = sm.Logit(complete["died"], X2).fit(disp=0)
    print(f"  Adjusted model:")
    for var in ["age_70plus", "male", "has_comorbidity"]:
        print(f"    {var}: OR = {np.exp(model2.params[var]):.2f}, "
              f"p = {model2.pvalues[var]:.2e}")

    # Model 3: age × comorbidity interaction
    complete["age_x_comorbidity"] = (
        complete["age_70plus"] * complete["has_comorbidity"]
    )
    X3 = sm.add_constant(complete[[
        "age_70plus", "male", "has_comorbidity", "age_x_comorbidity"
    ]])
    model3 = sm.Logit(complete["died"], X3).fit(disp=0)
    or_interaction = np.exp(model3.params["age_x_comorbidity"])
    p_interaction = model3.pvalues["age_x_comorbidity"]
    print(f"  Interaction OR = {or_interaction:.4f}, p = {p_interaction:.2e}")

    return {
        "n_complete": int(len(complete)),
        "age_only": {
            "or": float(or_age),
            "p": float(p_age),
        },
        "adjusted": {
            var: {
                "or": float(np.exp(model2.params[var])),
                "p": float(model2.pvalues[var]),
            }
            for var in ["age_70plus", "male", "has_comorbidity"]
        },
        "interaction": {
            "or": float(or_interaction),
            "p": float(p_interaction),
        },
    }


def per_site_interactions(df: pd.DataFrame) -> dict:
    """Per-site logistic regression with age × comorbidity interaction."""
    import statsmodels.api as sm

    results = {}
    sites = df.groupby("site").size()
    valid_sites = sites[sites >= 5000].index.tolist()

    print(f"\n  Fitting per-site interactions ({len(valid_sites)} sites)...")
    for site in tqdm(valid_sites, desc="Per-site interactions"):
        sub = df[df["site"] == site].copy()
        sub["age_x_comorbidity"] = sub["age_70plus"] * sub["has_comorbidity"]
        cols = ["age_70plus", "male", "has_comorbidity", "age_x_comorbidity"]
        complete = sub[["died"] + cols].dropna()

        if len(complete) < 100 or complete["died"].nunique() < 2:
            continue

        try:
            X = sm.add_constant(complete[cols])
            model = sm.Logit(complete["died"], X).fit(disp=0, maxiter=50)
            results[site] = {
                "n": int(len(complete)),
                "interaction_or": float(np.exp(model.params["age_x_comorbidity"])),
                "interaction_beta": float(model.params["age_x_comorbidity"]),
                "interaction_p": float(model.pvalues["age_x_comorbidity"]),
            }
        except Exception as e:
            print(f"    {site}: failed ({e})")

    return results


def consistency_test(df: pd.DataFrame, n_perms: int = 500) -> dict:
    """Multivariate consistency test on per-site correlation matrices."""
    analysis_vars = [
        "died", "hospitalized", "icu", "has_comorbidity", "male", "age_70plus"
    ]

    sites = df.groupby("site").size()
    valid_sites = sites[sites >= 5000].index.tolist()

    # Per-site correlation matrices
    corr_matrices = {}
    for site in valid_sites:
        sub = df[df["site"] == site][analysis_vars]
        if sub.shape[0] >= 100:
            corr_matrices[site] = sub.corr().values

    site_names = list(corr_matrices.keys())
    n_vars = len(analysis_vars)
    n_pairs = n_vars * (n_vars - 1) // 2

    # Extract upper triangles
    triu_idx = np.triu_indices(n_vars, k=1)
    site_vectors = {}
    for site in site_names:
        site_vectors[site] = corr_matrices[site][triu_idx]

    # Observed inconsistency
    def compute_inconsistency(vectors):
        sites_list = list(vectors.keys())
        total = 0.0
        count = 0
        for i in range(len(sites_list)):
            for j in range(i + 1, len(sites_list)):
                diff = vectors[sites_list[i]] - vectors[sites_list[j]]
                total += np.mean(diff ** 2)
                count += 1
        return total / count if count > 0 else 0.0

    observed = compute_inconsistency(site_vectors)

    # Permutation null
    null_stats = []
    rng = np.random.default_rng(None)
    all_vectors = np.array([site_vectors[s] for s in site_names])

    for _ in tqdm(range(n_perms), desc="Consistency permutation"):
        perm_vectors = {}
        for pair_idx in range(n_pairs):
            perm = rng.permutation(len(site_names))
            for si, site in enumerate(site_names):
                if site not in perm_vectors:
                    perm_vectors[site] = np.zeros(n_pairs)
                perm_vectors[site][pair_idx] = all_vectors[perm[si], pair_idx]
        null_stats.append(compute_inconsistency(perm_vectors))

    null_mean = np.mean(null_stats)
    null_std = np.std(null_stats)
    z = (observed - null_mean) / null_std if null_std > 0 else 0.0
    p = np.mean(np.array(null_stats) >= observed)

    print(f"\n  Consistency statistic:")
    print(f"    Observed: {observed:.6f}")
    print(f"    Null: {null_mean:.6f} ± {null_std:.6f}")
    print(f"    z = {z:.2f}, p = {p:.4f}")

    return {
        "observed": float(observed),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
        "z_score": float(z),
        "p_value": float(p),
        "n_sites": len(site_names),
        "coverage": "homogeneous (all sites report all 15 pairs)",
    }


def main():
    # Find the CSV
    csv_candidates = list(DATA_DIR.glob("*.csv"))
    if not csv_candidates:
        print("No CSV found. Looking for ZIP...")
        import zipfile
        zip_candidates = list(DATA_DIR.glob("*.zip"))
        if not zip_candidates:
            print("No data files found. Download first.")
            sys.exit(1)
        print(f"Extracting {zip_candidates[0]}...")
        with zipfile.ZipFile(zip_candidates[0]) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                print("No CSV in ZIP")
                sys.exit(1)
            zf.extract(csv_names[0], DATA_DIR)
            csv_path = DATA_DIR / csv_names[0]
    else:
        csv_path = csv_candidates[0]

    print(f"Using: {csv_path}")
    df = load_data(str(csv_path))
    df = prepare_variables(df)

    print(f"\n{'='*70}")
    print("MEXICO COVID-19 ECOLOGICAL FALLACY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total confirmed cases: {len(df):,}")
    print(f"Deaths: {df['died'].sum():,} ({100*df['died'].mean():.1f}%)")
    print(f"Sites (states): {df['site'].nunique()}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "Mexico COVID-19 Datos Abiertos",
        "n_total": int(len(df)),
        "n_deaths": int(df["died"].sum()),
        "n_sites": int(df["site"].nunique()),
    }

    # 1. Ecological regression
    print(f"\n--- Ecological regression (site-level) ---")
    results["ecological_regression"] = ecological_regression(df)

    # 2. Individual-level logistic
    print(f"\n--- Individual-level logistic regression ---")
    results["individual_logistic"] = individual_logistic(df)

    # 3. Per-site interactions
    print(f"\n--- Per-site age × comorbidity interactions ---")
    results["per_site_interactions"] = per_site_interactions(df)

    # 4. Consistency test
    print(f"\n--- Consistency test (32 states, homogeneous coverage) ---")
    results["consistency_test"] = consistency_test(df)

    # Save
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{datetime.now().isoformat()}] Results saved to {OUT_FILE}")

    # Summary
    eco = results["ecological_regression"]
    ind = results["individual_logistic"]
    print(f"\n{'='*70}")
    print("ECOLOGICAL FALLACY SUMMARY (MEXICO — REAL HOSPITAL SITES)")
    print(f"{'='*70}")
    print(f"  Site-level:      β = {eco['beta']:+.4f}, p = {eco['p_value']:.4f}")
    print(f"  Individual-level: OR = {ind['age_only']['or']:.2f}, "
          f"p = {ind['age_only']['p']:.2e}")
    print(f"  Sites: {eco['n_sites']} states")

    interactions = results["per_site_interactions"]
    if interactions:
        ors = [v["interaction_or"] for v in interactions.values()]
        print(f"  Per-site interaction OR range: "
              f"{min(ors):.3f} — {max(ors):.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
