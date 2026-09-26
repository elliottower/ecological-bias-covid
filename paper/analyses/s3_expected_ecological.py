"""
Supplementary Analysis S3: Model-Implied Ecological Slope

For each of Mexico's 32 treating-unit states, the fitted patient-level logistic
model (age 70+, sex, any of nine comorbidities) gives every patient a predicted
probability of death; averaging those within a state gives the state mortality the
model implies for the state's observed covariate distribution. Regressing implied
state mortality on elderly share gives the model-implied ecological slope, beside
the observed one.

The gap between them is the observed-minus-model-implied ecological discrepancy.
It is not aggregation bias alone: state context, omitted patient variables,
differences in baseline mortality, ascertainment, coding and model misspecification
all land in it. The slope difference is the primary quantity, with a residual
regression and a bootstrap over states for uncertainty; the ratio, unstable across
32 units, is secondary. The product-of-marginals approximation, which treats age,
sex and comorbidity as independent within a state, is reported as a sensitivity
analysis. The registered 20% rule is applied as frozen.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import mexico_confirmed_cases
from paths import RESULTS

OUTPUT_DIR = RESULTS
AGE_THRESHOLD = mexico_confirmed_cases.AGE_THRESHOLD
COVARIATES = ["elderly", "male", "has_comorbidity"]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260922


def fit_patient_model(cells):
    """Binomial GLM on covariate cells; the covariates are binary, so this is the
    patient-level fit."""
    pooled = cells.groupby(COVARIATES, as_index=False).agg(
        deaths=("deaths", "sum"), alive=("alive", "sum"), n=("n", "sum"))
    X = np.column_stack([np.ones(len(pooled))] +
                        [pooled[c].to_numpy(dtype=float) for c in COVARIATES])
    fit = sm.GLM(pooled[["deaths", "alive"]].to_numpy(dtype=float), X,
                 family=sm.families.Binomial()).fit()
    if not fit.converged:
        raise RuntimeError("patient-level logistic regression did not converge")
    return {
        "intercept": float(fit.params[0]),
        "beta_elderly": float(fit.params[1]),
        "beta_male": float(fit.params[2]),
        "beta_comorb": float(fit.params[3]),
    }


def predicted_probability(cells, coefficients):
    logit = (coefficients["intercept"]
             + coefficients["beta_elderly"] * cells["elderly"]
             + coefficients["beta_male"] * cells["male"]
             + coefficients["beta_comorb"] * cells["has_comorbidity"])
    return 1.0 / (1.0 + np.exp(-logit))


def state_table(cells, coefficients):
    """Per-state size, observed mortality, model-implied mortality and covariate shares."""
    predicted = predicted_probability(cells, coefficients)
    weighted = cells.assign(predicted_deaths=predicted * cells["n"])
    table = weighted.groupby("site").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "observed_mortality": g["deaths"].sum() / g["n"].sum(),
            "model_implied_mortality": g["predicted_deaths"].sum() / g["n"].sum(),
            "prop_elderly": (g["n"] * g["elderly"]).sum() / g["n"].sum(),
            "prop_male": (g["n"] * g["male"]).sum() / g["n"].sum(),
            "prop_comorb": (g["n"] * g["has_comorbidity"]).sum() / g["n"].sum(),
        }), include_groups=False)
    return table


def marginal_prediction(row, coefficients):
    """Product-of-marginals approximation: the covariates treated as independent."""
    predicted = 0.0
    for elderly in (0, 1):
        for male in (0, 1):
            for comorb in (0, 1):
                weight = ((row["prop_elderly"] if elderly else 1 - row["prop_elderly"]) *
                          (row["prop_male"] if male else 1 - row["prop_male"]) *
                          (row["prop_comorb"] if comorb else 1 - row["prop_comorb"]))
                logit = (coefficients["intercept"] + coefficients["beta_elderly"] * elderly
                         + coefficients["beta_male"] * male + coefficients["beta_comorb"] * comorb)
                predicted += weight / (1.0 + np.exp(-logit))
    return predicted


def slopes(table):
    observed = stats.linregress(table["prop_elderly"], table["observed_mortality"])
    implied = stats.linregress(table["prop_elderly"], table["model_implied_mortality"])
    residual = stats.linregress(table["prop_elderly"],
                                table["observed_mortality"] - table["model_implied_mortality"])
    return observed, implied, residual


def bootstrap(cells, table, draws, seed):
    """Resample states with replacement, refitting the patient model each draw."""
    rng = np.random.default_rng(seed)
    sites = np.array(sorted(cells["site"].unique()))
    observed_slopes, implied_slopes, differences, ratios = [], [], [], []
    for _ in range(draws):
        drawn = rng.choice(sites, size=len(sites), replace=True)
        resampled = pd.concat([cells[cells["site"] == s].assign(site=f"{s}_{i}")
                               for i, s in enumerate(drawn)], ignore_index=True)
        coefficients = fit_patient_model(resampled)
        drawn_table = state_table(resampled, coefficients)
        observed = stats.linregress(drawn_table["prop_elderly"],
                                    drawn_table["observed_mortality"]).slope
        implied = stats.linregress(drawn_table["prop_elderly"],
                                   drawn_table["model_implied_mortality"]).slope
        observed_slopes.append(observed)
        implied_slopes.append(implied)
        differences.append(observed - implied)
        ratios.append(observed / implied if implied != 0 else np.nan)

    def interval(values):
        return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]

    return {
        "draws": draws,
        "seed": seed,
        "resampling_unit": "state",
        "observed_slope_ci": interval(observed_slopes),
        "model_implied_slope_ci": interval(implied_slopes),
        "slope_difference_ci": interval(differences),
        "ratio_ci": interval(ratios),
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S3: Model-Implied Ecological Slope  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    cells = mexico_confirmed_cases.collapsed_cells(df)
    coefficients = fit_patient_model(cells)
    print("\n  Patient-level model:")
    for name, key in [("elderly", "beta_elderly"), ("male", "beta_male"),
                      ("comorbidity", "beta_comorb")]:
        print(f"    {name}: OR = {np.exp(coefficients[key]):.2f}")

    table = state_table(cells, coefficients)
    table["model_implied_mortality_independent_marginals"] = table.apply(
        marginal_prediction, axis=1, coefficients=coefficients)

    observed, implied, residual = slopes(table)
    marginal = stats.linregress(table["prop_elderly"],
                                table["model_implied_mortality_independent_marginals"])
    difference = observed.slope - implied.slope
    ratio = observed.slope / implied.slope
    t_critical = stats.t.ppf(0.975, len(table) - 2)

    print(f"\n  Observed ecological slope:      {observed.slope:+.4f} (p={observed.pvalue:.2e})")
    print(f"  Model-implied ecological slope: {implied.slope:+.4f} (p={implied.pvalue:.2e})")
    print(f"  Difference:                     {difference:+.4f} "
          f"(residual-slope p={residual.pvalue:.2e})")
    print(f"  Ratio (observed/implied):       {ratio:.2f}")

    print(f"\n  Bootstrapping {BOOTSTRAP_DRAWS} state resamples...")
    intervals = bootstrap(cells, table, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    print(f"    slope difference 95% CI: [{intervals['slope_difference_ci'][0]:+.4f}, "
          f"{intervals['slope_difference_ci'][1]:+.4f}]")
    print(f"    ratio 95% CI:            [{intervals['ratio_ci'][0]:.2f}, "
          f"{intervals['ratio_ci'][1]:.2f}]")

    registered_agreement = bool(abs(1 - ratio) < 0.20)
    if registered_agreement:
        conclusion = ("FALSIFICATION under the registered 20% rule: the observed and "
                       f"model-implied ecological slopes agree within 20% (ratio={ratio:.2f}).")
    else:
        direction = "above" if ratio > 1 else "below"
        conclusion = (f"The observed ecological slope is {direction} the slope this patient "
                       f"model implies for the observed state compositions: "
                       f"{observed.slope:+.4f} against {implied.slope:+.4f}, a difference of "
                       f"{difference:+.4f} (ratio {ratio:.2f}). The registered 20% rule is not "
                       f"met, so the registered falsification does not fire.")
    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "coefficients": coefficients,
        "n_states": int(len(table)),
        "observed_ecological_slope": float(observed.slope),
        "observed_ecological_se": float(observed.stderr),
        "observed_ecological_p": float(observed.pvalue),
        "model_implied_ecological_slope": float(implied.slope),
        "model_implied_ecological_se": float(implied.stderr),
        "model_implied_ecological_p": float(implied.pvalue),
        "model_implied_slope_method": ("mean over each state's patients of the model's predicted "
                                        "probability, i.e. the observed joint covariate "
                                        "distribution"),
        "slope_difference": float(difference),
        "slope_difference_residual_regression": {
            "slope": float(residual.slope),
            "se": float(residual.stderr),
            "ci": [float(residual.slope - t_critical * residual.stderr),
                   float(residual.slope + t_critical * residual.stderr)],
            "p": float(residual.pvalue),
            "note": ("state residual, observed minus model-implied mortality, regressed on "
                      "elderly share; its slope is the difference of the two slopes"),
        },
        "ratio_observed_implied": float(ratio),
        "bootstrap": intervals,
        "sensitivity_independent_marginals": {
            "model_implied_slope": float(marginal.slope),
            "ratio_observed_implied": float(observed.slope / marginal.slope),
            "note": "covariates treated as independent within a state",
        },
        "registered_rule": {
            "criterion": ("observed and expected ecological slopes agree within 20% "
                           "(ANALYSIS_PROTOCOL.md, S3)"),
            "agreement_within_20_percent": registered_agreement,
            "note": ("a frozen decision rule, not an equivalence test; the interval on the "
                      "slope difference is the quantity to read"),
        },
        "per_state": json.loads(table.to_json(orient="index")),
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s3_expected_ecological.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
