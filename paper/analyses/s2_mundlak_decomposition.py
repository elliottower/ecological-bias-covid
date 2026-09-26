"""
Supplementary Analysis S2: Within/Between Effect Decomposition (Mundlak)

The registered specification carries the state mean of the age indicator beside
31 state dummies. A state mean is constant within a state, so it lies in the span
of the intercept and the dummies, and its coefficient is not identified: the
registered decomposition cannot be estimated. This script establishes that before
fitting anything, reports the within-state coefficient, which is identified, and
fits a correlated-random-effects model in its place as a labeled exploratory
correction (r/glmm_fits.R).
"""

import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm

import mexico_confirmed_cases
from paths import PROJECT_ROOT, RESULTS

OUTPUT_DIR = RESULTS
AGE_THRESHOLD = mexico_confirmed_cases.AGE_THRESHOLD
GLMM_SCRIPT = PROJECT_ROOT / "paper" / "analyses" / "r" / "glmm_fits.R"


def between_is_identified(state_means) -> bool:
    """False when the state-mean column lies in the span of the intercept and state dummies."""
    sites = sorted(state_means)
    means = np.array([state_means[s] for s in sites])
    design = np.column_stack([np.ones(len(sites)), np.eye(len(sites))[:, 1:]])
    coef, *_ = np.linalg.lstsq(design, means, rcond=None)
    return not bool(np.allclose(design @ coef, means))


def within_coefficient(cells):
    """Logistic regression of death on age and state fixed effects, fitted on cells.

    Every covariate is binary, so this cell fit carries the likelihood of the
    patient-level fit. Its age coefficient is the within-state coefficient of the
    registered specification, which the collinear between-state column leaves
    unchanged.
    """
    grouped = cells.groupby(["site", "elderly"], as_index=False).agg(
        deaths=("deaths", "sum"), alive=("alive", "sum"), n=("n", "sum"))
    sites = sorted(grouped["site"].unique())
    dummies = pd.get_dummies(pd.Categorical(grouped["site"], categories=sites),
                             drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(grouped)), grouped["elderly"].to_numpy(dtype=float), dummies])
    endog = grouped[["deaths", "alive"]].to_numpy(dtype=float)

    fit = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    if not fit.converged:
        raise RuntimeError("within-state logistic regression did not converge")

    coef, se = float(fit.params[1]), float(fit.bse[1])
    return {
        "coef": coef,
        "se": se,
        "or": float(np.exp(coef)),
        "ci": [float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se))],
        "p": float(fit.pvalues[1]),
        "n_cells": int(len(grouped)),
        "n_patients": int(grouped["n"].sum()),
    }


def exploratory_within_between(cells):
    """Correlated-random-effects within-between model, fitted in R."""
    cells_path = OUTPUT_DIR / "mexico_cells.csv"
    out_path = OUTPUT_DIR / "mexico_glmm_fits.json"
    cells.to_csv(cells_path, index=False)
    subprocess.run(["Rscript", str(GLMM_SCRIPT), str(cells_path), str(out_path)], check=True)
    fits = json.load(open(out_path))
    return fits["s2_within_between_random_intercept"], fits


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S2: Within/Between Effect Decomposition  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    cells = mexico_confirmed_cases.collapsed_cells(df)
    state_means = df.groupby("site")["elderly"].mean().to_dict()

    identified = between_is_identified(state_means)
    print(f"\n  Between-state coefficient identified under the registered specification: "
          f"{identified}")
    if identified:
        raise RuntimeError("the registered between-state column is not collinear here; "
                            "this script's reason for reporting it as non-estimable is gone")

    within = within_coefficient(cells)
    print(f"  Within-state coefficient: {within['coef']:+.4f} (OR={within['or']:.2f}, "
          f"95% CI [{within['ci'][0]:.2f}, {within['ci'][1]:.2f}], p={within['p']:.2e})")

    means = np.array([state_means[s] for s in sorted(state_means)])

    print("\n  Fitting the exploratory within-between random-intercept model in R...")
    cre, all_fits = exploratory_within_between(cells)
    print(f"    within:  OR={cre['elderly_within']['or']:.2f} "
          f"[{cre['elderly_within']['ci'][0]:.2f}, {cre['elderly_within']['ci'][1]:.2f}]")
    print(f"    between: OR={cre['elderly_between']['or']:.2f} "
          f"[{cre['elderly_between']['ci'][0]:.2f}, {cre['elderly_between']['ci'][1]:.2f}]")

    rescaled = {
        "state_elderly_share": {
            "min": float(means.min()), "max": float(means.max()),
            "sd": float(means.std(ddof=1)),
        },
        "note": ("the between-state coefficient is per unit change in a state's elderly share, "
                  "a contrast from 0 to 1 that no state approaches; these rescalings use "
                  "contrasts the data contain"),
    }
    for label, scale in [("per_percentage_point", 0.01),
                          ("per_sd_of_state_share", float(means.std(ddof=1))),
                          ("across_the_observed_range", float(means.max() - means.min()))]:
        log_or = cre["elderly_between"]["log_or"] * scale
        se = cre["elderly_between"]["se"] * scale
        rescaled[label] = {
            "scale": scale,
            "log_or": log_or,
            "se": se,
            "or": float(np.exp(log_or)),
            "ci": [float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se))],
        }
    print(f"    between, per percentage point: OR="
          f"{rescaled['per_percentage_point']['or']:.3f} "
          f"[{rescaled['per_percentage_point']['ci'][0]:.3f}, "
          f"{rescaled['per_percentage_point']['ci'][1]:.3f}]")

    conclusion = (
        "The registered specification cannot identify the between-state coefficient: the state "
        "mean of the age indicator is collinear with the state dummies. No between-state "
        "coefficient, no within-between difference and no aggregation-bias ratio follows from "
        "it. The within-state coefficient is identified and reported. The registered CI-overlap "
        "criterion does not apply to a quantity that cannot be estimated.")
    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "registered_analysis": {
            "estimator": "Mundlak fixed-effects approximation (logistic + state dummies)",
            "status": "not_identified",
            "completed": True,
            "reason": ("the between-state exposure is constant within each state and therefore "
                        "lies in the span of the intercept and the 31 state dummies"),
            "between_coefficient_reported": False,
            "within_state": within,
        },
        "exploratory_correction": {
            "estimator": ("correlated-random-effects within-between binomial model, "
                           "logit P(death) = a + u_state + bW (elderly - state mean) + "
                           "bB (state mean), fitted on site x elderly x sex x comorbidity "
                           "cells with lme4::glmer"),
            "within_state": cre["elderly_within"],
            "between_state": cre["elderly_between"],
            "between_state_rescaled": rescaled,
            "random_effects_sd": cre["random_effects_sd"],
            "converged": cre["converged"],
            "convergence_messages": cre["convergence_messages"],
            "software": {k: all_fits[k] for k in ["r_version", "lme4_version"]},
        },
        "n_obs": int(len(df)),
        "n_states": int(df["site"].nunique()),
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s2_mundlak_decomposition.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
