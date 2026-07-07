"""
RCIT on real 4CE patient-level structure.

Tests conditional independence implications of a causal DAG over 4CE
site-level variables using RCIT (kernel CI test with random Fourier features).

The assumed DAG for the 4CE neurological COVID study is:
  Age → Comorbidity burden → Severity
  Age → Neuro involvement → Severity
  Sex → Comorbidity burden
  Comorbidity burden → Mortality
  Neuro involvement → Mortality

This tests whether the real aggregate data from 20 sites is consistent
with these assumptions, using RCIT for nonlinear conditional independence.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.test_ci import rcit_test_via_r


DATA_DIR = Path(__file__).parent.parent / "data" / "4ce_neuro"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def build_4ce_dag_dataset() -> pd.DataFrame:
    """Build a site-level dataset for DAG testing.

    Each row is a (site, neuro_type) observation. Variables represent
    aggregate patient characteristics at each site.
    """
    demo = pd.read_csv(DATA_DIR / "site_demographics.csv")
    clin = pd.read_csv(DATA_DIR / "site_clinical.csv")
    pmi = pd.read_csv(DATA_DIR / "site_pmi_effects.csv")

    demo["neuro_type"] = demo["neuro_type"].fillna("None")

    merged = demo.merge(clin, on=["site", "neuro_type"], how="left")

    neuro_map = {"None": 0, "Peripheral": 1, "Central": 2}
    merged["neuro_severity"] = merged["neuro_type"].map(neuro_map)

    cns30 = pmi[(pmi["effect_name"] == "pmi.cns") & (pmi["timepoint"] == 30)]
    cns30 = cns30[["site", "log_pmi"]].rename(columns={"log_pmi": "cns_effect"})
    merged = merged.merge(cns30, on="site", how="left")

    merged = merged.dropna(subset=["n_patients", "prop_severe"])
    merged = merged[merged["n_patients"] >= 5]

    return merged


def define_4ce_dag() -> list[dict]:
    """Define CI implications of the assumed 4CE causal DAG.

    The DAG has edges:
      age → comorbidity_burden
      age → neuro_severity
      age → severity
      sex → comorbidity_burden
      comorbidity_burden → severity
      comorbidity_burden → mortality
      neuro_severity → severity
      neuro_severity → mortality
      severity → mortality
    """
    implications = []

    var_names = {
        "age": "prop_age_80plus",
        "sex": "prop_female",
        "elix": "mean_elixhauser",
        "neuro": "neuro_severity",
        "severity": "prop_severe",
        "mortality": "prop_deceased",
        "readmit": "prop_readmitted",
    }

    # Independence: sex _|_ neuro_severity | age
    implications.append({
        "x": var_names["sex"], "y": var_names["neuro"],
        "z": [var_names["age"]],
        "expected": "independent",
        "desc": "sex _|_ neuro | age",
    })

    # Independence: sex _|_ mortality | {elix, severity}
    implications.append({
        "x": var_names["sex"], "y": var_names["mortality"],
        "z": [var_names["elix"], var_names["severity"]],
        "expected": "independent",
        "desc": "sex _|_ mortality | {elix, severity}",
    })

    # Independence: sex _|_ severity | {age, elix}
    implications.append({
        "x": var_names["sex"], "y": var_names["severity"],
        "z": [var_names["age"], var_names["elix"]],
        "expected": "independent",
        "desc": "sex _|_ severity | {age, elix}",
    })

    # Dependence: age → severity (direct effect)
    implications.append({
        "x": var_names["age"], "y": var_names["severity"],
        "z": [],
        "expected": "dependent",
        "desc": "age → severity (marginal)",
    })

    # Dependence: neuro → severity
    implications.append({
        "x": var_names["neuro"], "y": var_names["severity"],
        "z": [],
        "expected": "dependent",
        "desc": "neuro → severity (marginal)",
    })

    # Dependence: neuro → mortality
    implications.append({
        "x": var_names["neuro"], "y": var_names["mortality"],
        "z": [],
        "expected": "dependent",
        "desc": "neuro → mortality (marginal)",
    })

    # Dependence: elix → severity
    implications.append({
        "x": var_names["elix"], "y": var_names["severity"],
        "z": [],
        "expected": "dependent",
        "desc": "elix → severity (marginal)",
    })

    # Dependence: elix → mortality
    implications.append({
        "x": var_names["elix"], "y": var_names["mortality"],
        "z": [],
        "expected": "dependent",
        "desc": "elix → mortality (marginal)",
    })

    # Independence: age _|_ readmit | {severity, elix}
    implications.append({
        "x": var_names["age"], "y": var_names["readmit"],
        "z": [var_names["severity"], var_names["elix"]],
        "expected": "independent",
        "desc": "age _|_ readmit | {severity, elix}",
    })

    # Dependence: neuro → readmit (via severity pathway)
    implications.append({
        "x": var_names["neuro"], "y": var_names["readmit"],
        "z": [],
        "expected": "dependent",
        "desc": "neuro → readmit (marginal)",
    })

    # Independence: sex _|_ readmit | {severity, elix}
    implications.append({
        "x": var_names["sex"], "y": var_names["readmit"],
        "z": [var_names["severity"], var_names["elix"]],
        "expected": "independent",
        "desc": "sex _|_ readmit | {severity, elix}",
    })

    # CNS effect should depend on age composition
    implications.append({
        "x": var_names["age"], "y": "cns_effect",
        "z": [],
        "expected": "dependent",
        "desc": "age → cns_effect (marginal)",
    })

    return implications


def run_rcit_4ce(alpha: float = 0.05) -> dict:
    """Run RCIT on real 4CE data to test DAG implications."""
    print(f"\n{'='*60}")
    print("RCIT ON REAL 4CE STRUCTURE")
    print(f"{'='*60}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = build_4ce_dag_dataset()
    print(f"\nDataset: {len(data)} rows, {data.site.nunique()} sites")

    implications = define_4ce_dag()
    print(f"Testing {len(implications)} CI implications")

    results = []
    for impl in tqdm(implications, desc="RCIT tests"):
        x_col = impl["x"]
        y_col = impl["y"]
        z_cols = impl["z"]

        sub = data.dropna(subset=[x_col, y_col] + z_cols)
        if len(sub) < 10:
            print(f"  Skipping {impl['desc']}: only {len(sub)} observations")
            continue

        try:
            result = rcit_test_via_r(sub, x_col, y_col, z_cols)
            p_value = result["p_value"] if result is not None else np.nan
        except Exception as e:
            print(f"  RCIT failed for {impl['desc']}: {e}")
            p_value = np.nan

        is_independent = p_value > alpha if not np.isnan(p_value) else None
        consistent = None
        if is_independent is not None:
            if impl["expected"] == "independent":
                consistent = is_independent
            else:
                consistent = not is_independent

        results.append({
            "description": impl["desc"],
            "x": x_col,
            "y": y_col,
            "z": "|".join(z_cols) if z_cols else "none",
            "expected": impl["expected"],
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "observed": "independent" if is_independent else "dependent" if is_independent is not None else None,
            "consistent": consistent,
            "n_observations": len(sub),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(tables_dir / "T11_rcit_4ce.csv", index=False)

    n_consistent = results_df["consistent"].sum()
    n_tested = results_df["consistent"].notna().sum()

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  {n_consistent}/{n_tested} implications consistent ({100*n_consistent/n_tested:.0f}%)")

    print(f"\n  Detailed results:")
    for _, row in results_df.iterrows():
        status = "OK" if row["consistent"] else "FAIL" if row["consistent"] is not None else "N/A"
        p_str = f"p={row['p_value']:.4f}" if row["p_value"] is not None else "p=N/A"
        print(f"    [{status:4s}] {row['description']:45s} {p_str:12s} "
              f"(expected {row['expected']}, observed {row['observed']})")

    print(f"\n  Saved T11_rcit_4ce.csv")

    import json
    summary = {
        "n_consistent": int(n_consistent),
        "n_tested": int(n_tested),
        "consistency_rate": float(n_consistent / n_tested) if n_tested > 0 else 0,
        "n_observations": int(len(data)),
        "n_sites": int(data.site.nunique()),
        "alpha": alpha,
    }
    with open(tables_dir / "T11_rcit_4ce_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return {"results": results_df.to_dict("records"), "summary": summary}


if __name__ == "__main__":
    run_rcit_4ce()
