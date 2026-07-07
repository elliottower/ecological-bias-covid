"""
Supplementary Analysis S3: Expected Ecological Coefficient Under No Bias

For each of Mexico's 32 states, predicts site-level mortality by
integrating the fitted individual-level logistic model over the state's
observed covariate distribution. Regresses predicted mortality on elderly
proportion to get the "expected ecological slope" absent aggregation bias.
"""

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MEXICO_CSV = DATA_DIR / "mexico_covid/COVID19MEXICO.csv"
AGE_THRESHOLD = 70


def load_state_data():
    """Load individual data and compute per-state covariate distributions."""
    print("  Loading Mexico individual-level data...")
    state_data = {}

    with open(MEXICO_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = int(row["EDAD"])
            except (ValueError, KeyError):
                continue

            state = row.get("ENTIDAD_RES", "").strip()
            if not state:
                continue

            fecha_def = row.get("FECHA_DEF", "").strip()
            died = 0 if fecha_def in ("", "9999-99-99") else 1
            elderly = 1 if age >= AGE_THRESHOLD else 0
            male = 1 if row.get("SEXO", "").strip() == "1" else 0

            has_comorbidity = 0
            for comorb_col in ["DIABETES", "EPOC", "ASMA", "INMUSUPR",
                               "HIPERTENSION", "CARDIOVASCULAR", "OBESIDAD",
                               "RENAL_CRONICA"]:
                if row.get(comorb_col, "").strip() == "1":
                    has_comorbidity = 1
                    break

            if state not in state_data:
                state_data[state] = {"elderly": [], "male": [], "comorb": [],
                                     "died": []}
            state_data[state]["elderly"].append(elderly)
            state_data[state]["male"].append(male)
            state_data[state]["comorb"].append(has_comorbidity)
            state_data[state]["died"].append(died)

    result = {}
    for state, d in state_data.items():
        n = len(d["died"])
        result[state] = {
            "n": n,
            "prop_elderly": sum(d["elderly"]) / n,
            "prop_male": sum(d["male"]) / n,
            "prop_comorb": sum(d["comorb"]) / n,
            "mortality_rate": sum(d["died"]) / n,
        }

    print(f"  Loaded {sum(v['n'] for v in result.values()):,} records "
          f"across {len(result)} states")
    return result


def fit_individual_model():
    """Fit individual-level logistic and return coefficients."""
    print("  Fitting individual-level logistic model...")
    records = []

    with open(MEXICO_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = int(row["EDAD"])
            except (ValueError, KeyError):
                continue

            state = row.get("ENTIDAD_RES", "").strip()
            if not state:
                continue

            fecha_def = row.get("FECHA_DEF", "").strip()
            died = 0 if fecha_def in ("", "9999-99-99") else 1
            elderly = 1 if age >= AGE_THRESHOLD else 0
            male = 1 if row.get("SEXO", "").strip() == "1" else 0

            has_comorbidity = 0
            for comorb_col in ["DIABETES", "EPOC", "ASMA", "INMUSUPR",
                               "HIPERTENSION", "CARDIOVASCULAR", "OBESIDAD",
                               "RENAL_CRONICA"]:
                if row.get(comorb_col, "").strip() == "1":
                    has_comorbidity = 1
                    break

            records.append((died, elderly, male, has_comorbidity))

    y = np.array([r[0] for r in records], dtype=np.float64)
    X = np.column_stack([
        np.ones(len(records)),
        np.array([r[1] for r in records], dtype=np.float64),
        np.array([r[2] for r in records], dtype=np.float64),
        np.array([r[3] for r in records], dtype=np.float64),
    ])

    from statsmodels.discrete.discrete_model import Logit
    model = Logit(y, X)
    result = model.fit(method="lbfgs", maxiter=100, disp=False)

    return {
        "intercept": float(result.params[0]),
        "beta_elderly": float(result.params[1]),
        "beta_male": float(result.params[2]),
        "beta_comorb": float(result.params[3]),
    }


def predict_state_mortality(state_info, coefficients):
    """Predict state-level mortality by integrating individual model
    over the state's covariate distribution."""
    intercept = coefficients["intercept"]
    b_e = coefficients["beta_elderly"]
    b_m = coefficients["beta_male"]
    b_c = coefficients["beta_comorb"]

    p_e = state_info["prop_elderly"]
    p_m = state_info["prop_male"]
    p_c = state_info["prop_comorb"]

    predicted = 0.0
    for elderly in [0, 1]:
        for male in [0, 1]:
            for comorb in [0, 1]:
                prob_cell = ((p_e if elderly else 1 - p_e) *
                             (p_m if male else 1 - p_m) *
                             (p_c if comorb else 1 - p_c))

                logit = intercept + b_e * elderly + b_m * male + b_c * comorb
                p_death = 1.0 / (1.0 + np.exp(-logit))
                predicted += prob_cell * p_death

    return predicted


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S3: Expected Ecological Coefficient  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    state_data = load_state_data()
    coefficients = fit_individual_model()

    print(f"\n  Individual-level model coefficients:")
    print(f"    intercept = {coefficients['intercept']:.4f}")
    print(f"    beta_elderly = {coefficients['beta_elderly']:.4f} "
          f"(OR = {np.exp(coefficients['beta_elderly']):.2f})")
    print(f"    beta_male = {coefficients['beta_male']:.4f} "
          f"(OR = {np.exp(coefficients['beta_male']):.2f})")
    print(f"    beta_comorb = {coefficients['beta_comorb']:.4f} "
          f"(OR = {np.exp(coefficients['beta_comorb']):.2f})")

    states = sorted(state_data.keys())
    elderly_props = []
    observed_mortalities = []
    predicted_mortalities = []

    for state in states:
        info = state_data[state]
        predicted = predict_state_mortality(info, coefficients)
        elderly_props.append(info["prop_elderly"])
        observed_mortalities.append(info["mortality_rate"])
        predicted_mortalities.append(predicted)

    observed_slope, obs_intercept, _, obs_p, obs_se = stats.linregress(
        elderly_props, observed_mortalities)
    expected_slope, exp_intercept, _, exp_p, exp_se = stats.linregress(
        elderly_props, predicted_mortalities)

    ratio = observed_slope / expected_slope if expected_slope != 0 else float("inf")

    print(f"\n  Observed ecological slope:  {observed_slope:+.4f} (p={obs_p:.2e})")
    print(f"  Expected ecological slope:  {expected_slope:+.4f} (p={exp_p:.2e})")
    print(f"  Ratio (observed/expected):  {ratio:.2f}")

    agreement = abs(1 - ratio) < 0.20

    if agreement:
        conclusion = ("FALSIFICATION: Observed and expected ecological slopes "
                       f"agree within 20% (ratio={ratio:.2f}). The ecological "
                       "regression is a valid summary of the individual-level "
                       "relationship.")
    else:
        direction = "amplified" if ratio > 1 else "attenuated"
        conclusion = (f"Ecological regression is {direction} relative to "
                       f"expectation (ratio={ratio:.2f}). Aggregation bias "
                       f"distorts the ecological estimate by {abs(1-ratio):.0%}.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "coefficients": coefficients,
        "n_states": len(states),
        "observed_ecological_slope": float(observed_slope),
        "observed_ecological_p": float(obs_p),
        "expected_ecological_slope": float(expected_slope),
        "expected_ecological_p": float(exp_p),
        "ratio_observed_expected": float(ratio),
        "within_20_percent": agreement,
        "per_state": {
            state: {
                "elderly_prop": elderly_props[i],
                "observed_mortality": observed_mortalities[i],
                "predicted_mortality": predicted_mortalities[i],
            }
            for i, state in enumerate(states)
        },
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s3_expected_ecological.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
