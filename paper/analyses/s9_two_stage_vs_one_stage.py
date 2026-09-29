"""
Supplementary Analysis S9: Two-Stage vs. One-Stage IPD on Mexico

Registered comparison on Mexico's 32 treating-unit states:
  (a) Ecological: state mortality rate on state elderly proportion
  (b) Two-stage IPD: per-state logistic, pooled by DerSimonian-Laird random effects
  (c) One-stage IPD: death ~ elderly + (1 | state)

(c) is the registered random-intercept model, fitted with lme4 on site x elderly
cells (r/glmm_fits.R); the covariates are binary, so the cell fit carries the
patient-level likelihood. Two estimators are aligned before they are compared: a
common-effect two-stage pooling belongs beside the random-intercept model, which
imposes one age coefficient on every state, and DerSimonian-Laird pooling belongs
beside a random-slope model, which lets the age coefficient vary. The registered
falsification criterion compares two-stage pooling with the pooled individual-level
odds ratio, which ignores state; that comparison is reported as registered, beside
the aligned ones.
"""

import json
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

import mexico_confirmed_cases
from paths import PROJECT_ROOT, RESULTS, write_result

OUTPUT_DIR = RESULTS
AGE_THRESHOLD = mexico_confirmed_cases.AGE_THRESHOLD
R_DIR = PROJECT_ROOT / "paper" / "analyses" / "r"


def by_site_and_age(cells):
    """Deaths and patients per site x elderly cell."""
    return cells.groupby(["site", "elderly"], as_index=False).agg(
        deaths=("deaths", "sum"), alive=("alive", "sum"), n=("n", "sum"))


def logistic_on_cells(table, covariate_columns):
    """Binomial GLM on aggregated cells; raises unless it converges."""
    X = np.column_stack([np.ones(len(table))] +
                        [table[c].to_numpy(dtype=float) for c in covariate_columns])
    endog = table[["deaths", "alive"]].to_numpy(dtype=float)
    fit = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    if not fit.converged:
        raise RuntimeError(f"binomial GLM on {covariate_columns} did not converge")
    return fit


def estimate(log_or, se, extra=None):
    out = {
        "log_or": float(log_or),
        "se": float(se),
        "or": float(np.exp(log_or)),
        "ci": [float(np.exp(log_or - 1.96 * se)), float(np.exp(log_or + 1.96 * se))],
    }
    if extra:
        out.update(extra)
    return out


def ecological_regression(site_age):
    """(a) Ecological: state mortality rate on state elderly proportion."""
    per_site = site_age.groupby("site").apply(
        lambda g: pd.Series({
            "prop_elderly": float((g["n"] * g["elderly"]).sum() / g["n"].sum()),
            "mortality_rate": float(g["deaths"].sum() / g["n"].sum()),
        }), include_groups=False)
    slope, intercept, r, p, se = scipy_stats.linregress(
        per_site["prop_elderly"], per_site["mortality_rate"])
    return {
        "method": "ecological",
        "beta": float(slope),
        "se": float(se),
        "p_value": float(p),
        "r_squared": float(r ** 2),
        "n_sites": int(len(per_site)),
        "note": "slope is in mortality-rate units per unit elderly-proportion",
    }


def per_state_estimates(site_age):
    """Per-state logistic of death on age, with separation and convergence flags.

    Each state contributes two cells and the model has two parameters, so the fit is
    saturated: statsmodels warns about zero residual degrees of freedom and perfect
    prediction. The coefficient is the 2x2 log odds ratio, which the check below
    confirms.
    """
    rows = []
    for site, g in site_age.groupby("site"):
        g = g.sort_values("elderly")
        counts = {(int(e), "deaths"): int(d) for e, d in zip(g["elderly"], g["deaths"])}
        counts.update({(int(e), "alive"): int(a) for e, a in zip(g["elderly"], g["alive"])})
        zero_cell = any(v == 0 for v in counts.values())
        closed_form = np.log(counts[(1, "deaths")] * counts[(0, "alive")] /
                             (counts[(1, "alive")] * counts[(0, "deaths")])) if not zero_cell else np.nan
        fit = logistic_on_cells(g, ["elderly"])
        if not zero_cell and not np.isclose(fit.params[1], closed_form, atol=1e-8):
            raise RuntimeError(f"state {site}: fitted log odds ratio {fit.params[1]} differs from "
                                f"the 2x2 value {closed_form}")
        rows.append({
            "site": int(site),
            "log_or": float(fit.params[1]),
            "se": float(fit.bse[1]),
            "or": float(np.exp(fit.params[1])),
            "n": int(g["n"].sum()),
            "deaths": int(g["deaths"].sum()),
            "zero_cell": zero_cell,
            "converged": bool(fit.converged),
        })
    return pd.DataFrame(rows)


def dersimonian_laird(per_state):
    """(b) Two-stage IPD as registered: per-state logistic pooled by DerSimonian-Laird."""
    log_ors = per_state["log_or"].to_numpy()
    variances = per_state["se"].to_numpy() ** 2
    weights_fe = 1.0 / variances
    w_sum = weights_fe.sum()
    fe_est = float((weights_fe * log_ors).sum() / w_sum)

    q = float((weights_fe * (log_ors - fe_est) ** 2).sum())
    k = len(log_ors)
    c = w_sum - (weights_fe ** 2).sum() / w_sum
    tau2 = max(0.0, (q - (k - 1)) / c)

    weights_re = 1.0 / (variances + tau2)
    re_est = float((weights_re * log_ors).sum() / weights_re.sum())
    re_se = float(1.0 / np.sqrt(weights_re.sum()))

    return estimate(re_est, re_se, {
        "method": "two_stage_ipd_dersimonian_laird",
        "tau2": float(tau2),
        "I2": float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0,
        "Q": q,
        "n_states_included": int(k),
    })


def run_r(script, table, out_name, **files):
    in_path = OUTPUT_DIR / files["in_name"]
    out_path = OUTPUT_DIR / out_name
    table.to_csv(in_path, index=False)
    subprocess.run(["Rscript", str(R_DIR / script), str(in_path), str(out_path)], check=True)
    return json.load(open(out_path))


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S9: Two-Stage vs. One-Stage IPD on Mexico  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    cells = mexico_confirmed_cases.collapsed_cells(df)
    site_age = by_site_and_age(cells)

    print("\n  (a) Ecological regression...")
    eco = ecological_regression(site_age)
    print(f"      beta = {eco['beta']:+.4f}, p = {eco['p_value']:.2e}, R2 = {eco['r_squared']:.3f}")

    print("\n  (b) Two-stage IPD...")
    per_state = per_state_estimates(site_age)
    two_stage = dersimonian_laird(per_state)
    print(f"      DerSimonian-Laird OR = {two_stage['or']:.2f} "
          f"[{two_stage['ci'][0]:.2f}, {two_stage['ci'][1]:.2f}], I2 = {two_stage['I2']:.3f}")
    meta = run_r("meta_pool.R", per_state[["site", "log_or", "se"]],
                 "mexico_meta_pool.json", in_name="mexico_per_state.csv")
    print(f"      common-effect OR = {np.exp(meta['common_effect']['log_or']):.2f}; "
          f"REML + Hartung-Knapp OR = {np.exp(meta['reml_hartung_knapp']['log_or']):.2f}")

    print("\n  (c) One-stage IPD...")
    glmm = run_r("glmm_fits.R", cells, "mexico_glmm_fits.json", in_name="mexico_cells.csv")
    one_stage_ri = glmm["s9_one_stage_random_intercept"]["elderly"]
    one_stage_rs = glmm["s9_one_stage_random_slope"]["elderly"]
    print(f"      random intercept (registered) OR = {one_stage_ri['or']:.2f} "
          f"[{one_stage_ri['ci'][0]:.2f}, {one_stage_ri['ci'][1]:.2f}]")
    print(f"      random slope OR = {one_stage_rs['or']:.2f} "
          f"[{one_stage_rs['ci'][0]:.2f}, {one_stage_rs['ci'][1]:.2f}]")

    fixed_effects = logistic_on_cells(
        site_age.assign(**{f"site_{s}": (site_age["site"] == s).astype(float)
                           for s in sorted(site_age["site"].unique())[1:]}),
        ["elderly"] + [f"site_{s}" for s in sorted(site_age["site"].unique())[1:]])
    one_stage_fe = estimate(fixed_effects.params[1], fixed_effects.bse[1],
                            {"method": "one_stage_state_fixed_effects"})
    print(f"      state fixed effects OR = {one_stage_fe['or']:.2f}")

    crude = logistic_on_cells(site_age.groupby("elderly", as_index=False).agg(
        deaths=("deaths", "sum"), alive=("alive", "sum"), n=("n", "sum")), ["elderly"])
    pooled = estimate(crude.params[1], crude.bse[1],
                      {"method": "pooled_individual_level_ignoring_state"})
    print(f"\n  Pooled individual-level OR (ignores state): {pooled['or']:.2f}")

    registered_difference = abs(two_stage["or"] - pooled["or"]) / pooled["or"]
    registered_fires = bool(registered_difference > 0.50)

    comparisons = {
        "common_effect_two_stage_vs_one_stage_random_intercept": {
            "two_stage_log_or": meta["common_effect"]["log_or"],
            "one_stage_log_or": one_stage_ri["log_or"],
            "log_or_difference": meta["common_effect"]["log_or"] - one_stage_ri["log_or"],
            "note": "both impose one age coefficient on every state",
        },
        "random_effects_two_stage_vs_one_stage_random_slope": {
            "two_stage_log_or": two_stage["log_or"],
            "one_stage_log_or": one_stage_rs["log_or"],
            "log_or_difference": two_stage["log_or"] - one_stage_rs["log_or"],
            "note": "both let the age coefficient vary across states",
        },
        "registered_falsification": {
            "criterion": ("two-stage IPD OR differs from the pooled individual-level OR by more "
                           "than 50% (ANALYSIS_PROTOCOL.md, S9)"),
            "two_stage_or": two_stage["or"],
            "pooled_individual_level_or": pooled["or"],
            "difference_fraction": float(registered_difference),
            "fires": registered_fires,
            "note": ("the registered comparator ignores state, so it carries between-state "
                      "confounding and noncollapsibility; a gap between it and a "
                      "state-conditioned estimate is not evidence that heterogeneity distorts "
                      "all pooling"),
        },
    }
    for key in ["common_effect_two_stage_vs_one_stage_random_intercept",
                "random_effects_two_stage_vs_one_stage_random_slope"]:
        d = comparisons[key]["log_or_difference"]
        print(f"  {key}: log-OR difference {d:+.4f}")
    print(f"  Registered falsification fires: {registered_fires} "
          f"({registered_difference:.0%} difference)")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "ecological": eco,
        "two_stage_ipd": two_stage,
        "two_stage_other_estimators": meta,
        "one_stage_random_intercept_registered": one_stage_ri,
        "one_stage_random_slope": one_stage_rs,
        "one_stage_state_fixed_effects": one_stage_fe,
        "pooled_individual_level": pooled,
        "per_state": json.loads(per_state.to_json(orient="records")),
        "per_state_flags": {
            "any_zero_cell": bool(per_state["zero_cell"].any()),
            "all_converged": bool(per_state["converged"].all()),
            "max_se": float(per_state["se"].max()),
        },
        "comparisons": comparisons,
        "software": {k: glmm[k] for k in ["r_version", "lme4_version"]} |
                    {"metafor_version": meta["metafor_version"]},
    }

    outpath = write_result(output, OUTPUT_DIR / "s9_two_stage_vs_one_stage.json")
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
