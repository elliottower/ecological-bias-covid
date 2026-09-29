"""
Extension S12: what the state-level discrepancy is made of

Registered in ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md, frozen at commit a38196b
(tag registration-s11-s12), before any record-level model other than the
three-covariate one had been fitted.

Six prespecified models, sites defined as the treating unit's state. Each step
reports the model-implied ecological slope, the observed-minus-implied difference,
the mean absolute calibration error across states, and the change from the step
before. The composition contrast is model 2 against model 5; the temporal contrast
is model 5 against model 6. No monotonic ordering is required or predicted.
"""

import gc
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy import stats

import glmm
import mexico_confirmed_cases
from paths import RESULTS, require_clean_tree, run_metadata, write_result, write_table

OUTPUT_DIR = RESULTS
COMORBIDITIES = [c.lower() for c in mexico_confirmed_cases.COMORBIDITY_COLS]
SPLINE_PERCENTILES = [5, 27.5, 50, 72.5, 95]
AGE_BAND_EDGES = list(range(0, 130, 10))
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 2000
RECENT_ONSET_DAYS = 28
SNAPSHOT_DATE = "2022-01-03"
IMPLAUSIBLE_AGE = 110
INSTABILITY_RATE = 0.01  # a bootstrap with more than this share of failed draws is unstable
MINIMUM_EXPOSURE_SD = 1e-6  # a draw with no elderly-share variation carries no slope
FIT_FAILURES = (PerfectSeparationError, np.linalg.LinAlgError, ValueError)
# Numerical acceptance thresholds for H6, fixed before the model is fitted. A fit that
# fails any of them leaves H6 not evaluable, with every diagnostic still reported.
H6_GATES = {
    "hessian_step_agreement": 1e-3,   # relative change in the errors across step sizes
    "condition_number": 1e10,
    "relative_gradient": 1e-6,        # largest |gradient| over |log-likelihood|
}
MAX_CONTEXTUAL_CELLS = 2_500_000  # beyond this the design matrix stops fitting in memory


def spline_term(knots):
    """Five knots at the registered percentiles: two boundary, three interior."""
    lower, *interior, upper = [float(k) for k in knots]
    return (f"cr(age, knots={interior}, lower_bound={lower}, upper_bound={upper}, "
            f"constraints='center')")


def model_specifications(knots):
    """The six registered steps, as patsy formulas over the cell table."""
    comorbidity_terms = " + ".join(COMORBIDITIES)
    spline = spline_term(knots)
    return [
        ("model_1_elderly_only", "elderly", False),
        ("model_2_elderly_sex_any_comorbidity", "elderly + male + has_comorbidity", False),
        ("model_3_nine_comorbidities", f"elderly + male + {comorbidity_terms}", False),
        ("model_4_age_bands", f"C(age_band) + male + {comorbidity_terms}", False),
        ("model_5_age_spline", f"{spline} + male + {comorbidity_terms}", True),
        ("model_6_age_spline_and_month",
         f"{spline} + male + {comorbidity_terms} + C(onset_month)", True),
    ]


def build_cells(df, covariates):
    """Deaths and records per state x covariate cell."""
    return df.groupby(["site"] + covariates, as_index=False).agg(
        deaths=("died", "sum"), n=("died", "size"))


def converged(fit) -> bool:
    """IRLS results carry .converged; the scipy optimizers report it in mle_retvals."""
    flag = getattr(fit, "converged", None)
    if flag is not None:
        return bool(flag)
    return bool(fit.mle_retvals.get("converged", False))


def fit_and_predict(cells, formula, return_fit=False):
    """Binomial GLM on the cells, returning predicted probability per cell row."""
    pooled = cells.groupby([c for c in cells.columns if c not in ("site", "deaths", "n")],
                           as_index=False).agg(deaths=("deaths", "sum"), n=("n", "sum"))
    design = patsy.dmatrix(formula, pooled, return_type="dataframe")
    exog_names = list(design.columns)
    endog = np.column_stack([pooled["deaths"], pooled["n"] - pooled["deaths"]]).astype(float)
    model = sm.GLM(endog, design.to_numpy(), family=sm.families.Binomial())
    model.exog_names[:] = exog_names
    fit = model.fit(maxiter=200)
    if not converged(fit):
        fit = model.fit(method="lbfgs", maxiter=1000)
        if not converged(fit):
            raise RuntimeError(f"{formula} did not converge under IRLS or lbfgs")
    predicted = pd.Series(fit.predict(design.to_numpy()), index=pooled.index)
    pooled = pooled.assign(predicted=predicted)
    key = [c for c in pooled.columns if c not in ("deaths", "n", "predicted")]
    merged = cells.merge(pooled[key + ["predicted"]], on=key, how="left")
    if return_fit:
        return merged, len(design.columns), fit
    return merged, len(design.columns)


def state_table(cells):
    """Observed and implied state mortality with the state's elderly share."""
    cells = cells.assign(predicted_deaths=cells["predicted"] * cells["n"])
    return cells.groupby("site").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "deaths": g["deaths"].sum(),
            "observed_mortality": g["deaths"].sum() / g["n"].sum(),
            "implied_mortality": g["predicted_deaths"].sum() / g["n"].sum(),
            "prop_elderly": (g["n"] * g["elderly"]).sum() / g["n"].sum(),
        }), include_groups=False)


def slopes(table):
    observed = stats.linregress(table["prop_elderly"], table["observed_mortality"])
    implied = stats.linregress(table["prop_elderly"], table["implied_mortality"])
    calibration = float(np.average(
        np.abs(table["observed_mortality"] - table["implied_mortality"]), weights=table["n"]))
    return {
        "observed_slope": float(observed.slope),
        "implied_slope": float(implied.slope),
        "slope_difference": float(observed.slope - implied.slope),
        "calibration_error": calibration,
    }


class BootstrapModel:
    """Everything a state bootstrap needs for one model, precomputed once.

    A draw is a multiset of states. Resampling changes only the per-cell counts, so
    the design matrix is built once and the fit is refitted per draw on reweighted
    counts, which is the registered "refitted inside every bootstrap draw".
    """

    def __init__(self, df, formula, covariates, needs_onset):
        self.formula = formula
        frame = df.dropna(subset=["onset_month"]) if needs_onset else df
        cells = build_cells(frame, covariates)
        self.states = np.array(sorted(cells["site"].unique()))
        self.covariates = covariates

        pooled = cells.groupby(covariates, as_index=False).agg(
            deaths=("deaths", "sum"), n=("n", "sum"))
        pooled["cell"] = np.arange(len(pooled))
        self.design = patsy.dmatrix(formula, pooled, return_type="dataframe").to_numpy()

        indexed = cells.merge(pooled[covariates + ["cell"]], on=covariates, how="left")
        state_index = {s: i for i, s in enumerate(self.states)}
        rows = indexed["site"].map(state_index).to_numpy()
        cols = indexed["cell"].to_numpy()
        self.counts = np.zeros((len(self.states), len(pooled)), dtype=np.float32)
        self.deaths = np.zeros_like(self.counts)
        np.add.at(self.counts, (rows, cols), indexed["n"].to_numpy())
        np.add.at(self.deaths, (rows, cols), indexed["deaths"].to_numpy())

        self.state_n = self.counts.sum(axis=1)
        self.state_deaths = self.deaths.sum(axis=1)
        elderly = np.array([pooled["elderly"].to_numpy()])
        self.state_elderly = (self.counts * elderly).sum(axis=1) / self.state_n
        self.observed = self.state_deaths / self.state_n
        self.start_params = None

    def draw(self, weights):
        """Fit on states weighted by their bootstrap multiplicity.

        Returns the slope difference and `None`, or `nan` and the reason the draw
        carries no estimate. A draw never aborts the run.
        """
        deaths = (weights @ self.deaths).astype(np.float64)
        n = (weights @ self.counts).astype(np.float64)
        present = n > 0  # cells with no records in this draw carry no likelihood
        endog = np.column_stack([deaths[present], n[present] - deaths[present]])
        try:
            fit = sm.GLM(endog, self.design[present], family=sm.families.Binomial()).fit(
                start_params=self.start_params, maxiter=200)
        except FIT_FAILURES as error:
            return np.nan, f"fit: {type(error).__name__}"
        if self.start_params is None:
            self.start_params = fit.params
        if not converged(fit):
            return np.nan, "convergence"
        probability = fit.predict(self.design)
        implied = (self.counts @ probability) / self.state_n
        drawn = np.repeat(np.arange(len(self.states)), weights.astype(int))
        exposure = self.state_elderly[drawn]
        if len(exposure) < 3 or exposure.std(ddof=1) < MINIMUM_EXPOSURE_SD:
            return np.nan, "no_exposure_variation"
        observed_slope = stats.linregress(exposure, self.observed[drawn]).slope
        implied_slope = stats.linregress(exposure, implied[drawn]).slope
        return observed_slope - implied_slope, None


def bootstrap_ladder(df, formulas, covariate_sets, seed, draws):
    """State bootstrap for every model, refitting inside each draw.

    One model is built, bootstrapped and released before the next is built, so the
    largest cell matrices are never in memory together. Every model sees the same
    sequence of state multiplicities, because each starts from the same seed.
    """
    collected, failures = {}, {}
    for (name, formula, needs_onset), covariates in zip(formulas, covariate_sets):
        started = time.time()
        model = BootstrapModel(df, formula, covariates, needs_onset)
        print(f"    {name}: {model.design.shape[0]:,} cells x {model.design.shape[1]} terms, "
              f"built in {time.time() - started:.0f}s", flush=True)

        rng = np.random.default_rng(seed)
        states = len(model.states)
        values, reasons = [], {}
        for draw in range(draws):
            weights = np.bincount(rng.integers(0, states, size=states),
                                  minlength=states).astype(np.float32)
            value, reason = model.draw(weights)
            values.append(value)
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
            if draw and draw % 250 == 0:
                print(f"      {name} draw {draw}/{draws} [{datetime.now():%H:%M:%S}]", flush=True)
        collected[name] = np.array(values)
        failures[name] = reasons
        print(f"    {name}: {draws} draws in {time.time() - started:.0f}s, "
              f"CI {[round(x, 4) for x in interval(collected[name])]}, "
              f"failed {sum(reasons.values())} {reasons or ''}", flush=True)
        del model
        gc.collect()
    return collected, failures


def interval(values):
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def covariates_for(formula, needs_onset):
    covariates = ["elderly"]
    if "has_comorbidity" in formula:
        covariates.append("has_comorbidity")
    if "male" in formula:
        covariates.append("male")
    if "age_band" in formula:
        covariates.append("age_band")
    if "cr(age" in formula:
        covariates.append("age")
    covariates += [c for c in COMORBIDITIES if f"{c} " in formula or formula.endswith(c)]
    if needs_onset:
        covariates.append("onset_month")
    return sorted(set(covariates))


def run_ladder(df, formulas, label):
    results = {}
    previous = None
    for name, formula, needs_onset in formulas:
        started = time.time()
        covariates = covariates_for(formula, needs_onset)
        sub = df.dropna(subset=["onset_month"]) if needs_onset else df
        cells = build_cells(sub, covariates)
        cells, n_terms = fit_and_predict(cells, formula)
        entry = slopes(state_table(cells))
        entry["formula"] = formula
        entry["terms"] = int(n_terms)
        entry["cells"] = int(len(cells))
        entry["records"] = int(sub.shape[0])
        entry["seconds"] = round(time.time() - started, 1)
        if previous is not None:
            entry["change_in_difference_from_previous"] = (
                entry["slope_difference"] - previous["slope_difference"])
        results[name] = entry
        previous = entry
        print(f"    {label} {name}: implied {entry['implied_slope']:+.4f}, difference "
              f"{entry['slope_difference']:+.4f}, calibration {entry['calibration_error']:.5f} "
              f"({entry['seconds']}s, {entry['cells']:,} cells)", flush=True)
    return results


def spline_basis(ages, knots):
    """The centered model-5 spline, evaluated once per distinct age."""
    frame = pd.DataFrame({"age": np.asarray(sorted(set(ages)), dtype=float)})
    design = patsy.dmatrix(spline_term(knots), frame, return_type="dataframe")
    design = design.drop(columns=[c for c in design.columns if c.lower() == "intercept"])
    design.columns = [f"age_spline_{i + 1}" for i in range(design.shape[1])]
    return pd.concat([frame, design], axis=1)


def contextual_model(df, knots):
    """H6: the state's elderly share beside the record-level covariates.

    Age enters as the same centered spline as model 5, so the fitted covariates are
    the registered ones. The random intercept over states is fitted by adaptive
    Gauss-Hermite quadrature in `glmm.py`: `lme4::glmer` segfaults in `pwrssUpdate`
    on this machine at this cell count, and on the one S9 fit `lme4` did complete the
    two implementations agree to 3.1e-07 in the log odds ratio and to six decimals in
    the standard error.
    """
    state = df.groupby("site").agg(
        elderly_share=("elderly", "mean"), unknown_rate=("comorbidity_unknown", "mean"))
    for column in ("elderly_share", "unknown_rate"):
        state[f"{column}_z"] = ((state[column] - state[column].mean())
                                / state[column].std(ddof=1))

    cells = df.groupby(["site", "age", "male"] + COMORBIDITIES, as_index=False).agg(
        deaths=("died", "sum"), n=("died", "size"))
    registered_covariates = len(cells) <= MAX_CONTEXTUAL_CELLS
    if registered_covariates:
        basis = spline_basis(df["age"], knots)
        cells = cells.merge(basis, on="age", how="left").drop(columns=["age"])
        age_columns = [c for c in basis.columns if c != "age"]
        age_term = "the centered model-5 spline, evaluated per distinct age"
    else:
        cells = df.groupby(["site", "age_band", "male"] + COMORBIDITIES, as_index=False).agg(
            deaths=("died", "sum"), n=("died", "size"))
        bands = pd.get_dummies(cells.pop("age_band"), prefix="age_band", drop_first=True)
        cells = pd.concat([cells, bands.astype(float)], axis=1)
        age_columns = list(bands.columns)
        age_term = ("ten-year bands: the spline's cells crossed with state exceeded "
                     f"{MAX_CONTEXTUAL_CELLS:,} rows, so this is not the registered model")
    cells = cells.merge(state[["elderly_share_z"]], left_on="site", right_index=True, how="left")

    terms = age_columns + ["male"] + COMORBIDITIES + ["elderly_share_z"]
    design = np.column_stack([np.ones(len(cells))] +
                             [cells[c].to_numpy(dtype=float) for c in terms])
    sites = sorted(cells["site"].unique())
    groups = cells["site"].map({s: i for i, s in enumerate(sites)}).to_numpy()
    print(f"    {len(cells):,} cells, {len(terms) + 1} terms, age as "
          f"{age_term.split(':')[0]}; fitting by quadrature...", flush=True)

    started = time.time()
    try:
        fit = glmm.fit_random_intercept(design, cells["deaths"], cells["n"], groups)
        converged_flag, message = True, ""
    except glmm.ConvergenceError as error:
        fit, converged_flag, message = None, False, str(error)
    elapsed = round(time.time() - started, 1)

    if fit is None:
        return {"converged": False, "convergence_message": message, "seconds": elapsed,
                "n_cells": int(len(cells)), "age_term": age_term,
                "registered_covariates": bool(registered_covariates), "evaluable": False,
                "elderly_share_per_sd": None,
                "state_table": json.loads(state.to_json(orient="index"))}

    index = terms.index("elderly_share_z") + 1
    share = glmm.wald(fit["beta"][index], fit["se"][index])
    profile = None if fit["singular"] else glmm.profile_interval(fit, index)
    if profile is not None:
        share["profile_ci"] = [float(np.exp(profile[0])), float(np.exp(profile[1]))]
        share["profile_log_or_ci"] = [float(profile[0]), float(profile[1])]
    share["interval_used"] = "profile likelihood" if profile else "Wald"

    gradient = glmm.gradient_check(fit)  # None at the boundary, where there is no interior
    relative_gradient = (None if gradient is None
                         else abs(gradient / fit["log_likelihood"]))
    diagnostics = fit["hessian"] or {}
    gates = {
        "registered_covariates": bool(registered_covariates),
        "converged": bool(converged_flag),
        "variance_component_off_the_boundary": not fit["singular"],
        "finite_estimate": bool(np.isfinite(fit["beta"]).all() and np.isfinite(fit["se"]).all()),
        "hessian_positive_definite": bool(diagnostics.get("positive_definite", False)),
        "hessian_agrees_across_steps": bool(
            diagnostics.get("max_relative_difference_between_steps", np.inf)
            < H6_GATES["hessian_step_agreement"]),
        "hessian_well_conditioned": bool(
            diagnostics.get("condition_number", np.inf) < H6_GATES["condition_number"]),
        "gradient_at_the_optimum": bool(relative_gradient is not None
                                        and relative_gradient < H6_GATES["relative_gradient"]),
        "profile_interval_available": profile is not None,
    }
    return {
        "estimator": ("binomial random intercept by adaptive Gauss-Hermite quadrature "
                      f"({fit['quadrature_nodes']} nodes), glmm.py"),
        "fitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": elapsed,
        "n_cells": int(len(cells)),
        "n_records": int(cells["n"].sum()),
        "terms": ["intercept"] + terms,
        "elderly_share_per_sd": share,
        "standard_error_source": fit["se_source"],
        "gates": gates,
        "gate_thresholds": H6_GATES,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "variance_starts": fit["variance_starts"],
        "conditional_mode_step_at_the_solution": fit["conditional_mode_step_at_the_solution"],
        "relative_gradient": relative_gradient,
        "se_conditional_on_sigma": (None if fit["se_conditional_on_sigma"] is None
                                    else float(fit["se_conditional_on_sigma"][index])),
        "hessian_diagnostics": fit["hessian"],
        "max_absolute_gradient": gradient,
        "random_intercept_sd": fit["sigma"],
        "boundary": {
            "deviance_difference_against_zero_variance": fit["boundary_lrt"],
            "p": fit["boundary_p"],
            "reference": fit["boundary_reference"],
            "at_boundary": fit["singular"],
        },
        "log_likelihood": fit["log_likelihood"],
        "iterations": fit["iterations"],
        "singular": fit["singular"],
        "converged": converged_flag,
        "n_states": len(sites),
        "age_term": age_term,
        "age_representation": ("model-5 centered spline basis" if registered_covariates
                               else "ten-year age bands"),
        "registered_covariates": bool(registered_covariates),
        "evaluable": all(gates.values()),
        "state_table": json.loads(state.to_json(orient="index")),
    }


def unknown_comorbidity_sensitivity(df, formulas):
    """The registered sensitivity: the state's rate of all-unknown comorbidity flags."""
    rates = df.groupby("site")["comorbidity_unknown"].mean()
    shares = df.groupby("site")["elderly"].mean()
    name, formula, needs_onset = formulas[4]  # model 5
    covariates = covariates_for(formula, needs_onset) + ["unknown_rate"]
    frame = df.dropna(subset=["onset_month"]) if needs_onset else df
    frame = frame.assign(unknown_rate=frame["site"].map(rates))
    cells = build_cells(frame, covariates)
    cells, _ = fit_and_predict(cells, f"{formula} + unknown_rate")
    entry = slopes(state_table(cells))

    clustered = patsy.dmatrix(f"{formula} + unknown_rate", cells, return_type="dataframe")
    endog = np.column_stack([cells["deaths"], cells["n"] - cells["deaths"]]).astype(float)
    groups = cells["site"].to_numpy()
    robust = sm.GLM(endog, clustered.to_numpy(), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True,
                                      "df_correction": True}, maxiter=200)
    index = list(clustered.columns).index("unknown_rate")
    coefficient, se = float(robust.params[index]), float(robust.bse[index])
    entry["unknown_rate_coefficient"] = {
        "log_or": coefficient, "se": se, "or": float(np.exp(coefficient)),
        "ci": [float(np.exp(coefficient - 1.96 * se)),
               float(np.exp(coefficient + 1.96 * se))],
        "converged": bool(converged(robust)),
        "clusters": int(len(np.unique(groups))),
        "uncertainty": ("state-clustered sandwich with a small-sample correction; the "
                         "covariate takes one value per state, so a model-based interval "
                         "would treat millions of records as independent information "
                         "about it, and 32 clusters is few enough that even the clustered "
                         "interval is read as a nuisance-parameter diagnostic"),
    }
    entry["correlation_with_elderly_share"] = float(np.corrcoef(rates, shares)[0, 1])
    entry["per_state_unknown_rate"] = {str(k): float(v) for k, v in rates.items()}
    entry["records_with_all_flags_unknown"] = int(df["comorbidity_unknown"].sum())
    entry["note"] = ("model 5 plus the state's rate of records whose nine comorbidity flags are "
                      "all unknown, which the primary coding places in the reference group")
    return entry


def main():
    require_clean_tree()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S12: composition ladder  [{ts}]")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    df = df.assign(age_band=pd.cut(df["age"], bins=AGE_BAND_EDGES, right=False).astype(str))
    knots = np.percentile(df["age"], SPLINE_PERCENTILES).round(1)
    formulas = model_specifications(knots)
    covariate_sets = [covariates_for(f, o) for _, f, o in formulas]

    onset_missing = int(df["onset_month"].isna().sum())
    print(f"\n  knots at ages {list(knots)}; {onset_missing:,} records have no valid onset date "
          f"({onset_missing / len(df):.2%})")

    print("\n  primary ladder:")
    primary = run_ladder(df, formulas, "primary")

    cutoff = pd.Timestamp(SNAPSHOT_DATE) - pd.Timedelta(days=RECENT_ONSET_DAYS)
    recent = df[df["onset"].notna() & (df["onset"] < cutoff)]
    print(f"\n  sensitivity, onset before {cutoff:%Y-%m-%d} ({len(recent):,} records):")
    recent_onset = run_ladder(recent, formulas, "recent-onset")

    plausible = df[df["age"] <= IMPLAUSIBLE_AGE]
    print(f"\n  sensitivity, ages <= {IMPLAUSIBLE_AGE} ({len(plausible):,} records):")
    age_sensitivity = run_ladder(plausible, formulas, "age")

    print("\n  H6, the contextual model:")
    contextual = contextual_model(df, knots)
    share = contextual["elderly_share_per_sd"]
    if share is None:
        print(f"    did not converge: {contextual['convergence_message']}")
    else:
        interval_shown = share.get("profile_ci", share["ci"])
        print(f"    elderly share per SD: OR {share['or']:.3f} "
              f"[{interval_shown[0]:.3f}, {interval_shown[1]:.3f}] "
              f"({share['interval_used']}), p {share['p']:.2e}, "
              f"sigma {contextual['random_intercept_sd']:.4f}, boundary deviance "
              f"{contextual['boundary']['deviance_difference_against_zero_variance']:.1f}, "
              f"{contextual['seconds']}s")

    print("\n  unknown-comorbidity sensitivity:")
    unknown = unknown_comorbidity_sensitivity(df, formulas)
    print(f"    model 5 plus the state unknown rate: difference {unknown['slope_difference']:+.4f} "
          f"against {primary['model_5_age_spline']['slope_difference']:+.4f}")

    print(f"\n  state bootstrap, {BOOTSTRAP_DRAWS} draws, refitting every model in each draw:")
    draws, draw_failures = bootstrap_ladder(df, formulas, covariate_sets, BOOTSTRAP_SEED,
                                            BOOTSTRAP_DRAWS)
    write_table(pd.DataFrame(draws), OUTPUT_DIR / "s12_bootstrap_draws.csv")
    for name, values in draws.items():
        failed = int(np.isnan(values).sum())
        primary[name]["bootstrap"] = {
            "draws": int(len(values)),
            "valid_draws": int(len(values) - failed),
            "failed_draws": failed,
            "failure_reasons": draw_failures[name],
            "unstable": bool(failed > INSTABILITY_RATE * len(values)),
            "interval_is_conditional_on_valid_draws": bool(failed > 0),
            "slope_difference_ci": interval(values),
            "mc_se": float(np.nanstd(values, ddof=1) / np.sqrt(np.sum(~np.isnan(values)))),
        }
        print(f"    {name}: CI {[round(x, 4) for x in interval(values)]}")

    composition = primary["model_5_age_spline"]["slope_difference"]
    benchmark = primary["model_2_elderly_sex_any_comorbidity"]["slope_difference"]
    temporal = primary["model_6_age_spline_and_month"]["slope_difference"]
    output = {
        "timestamp": ts,
        "run": run_metadata(f"s12_{ts.replace(' ', 'T')}"),
        "registration": {"file": "ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md",
                          "commit": "a38196b", "tag": "registration-s11-s12"},
        "spline_knots": [float(k) for k in knots],
        "records_without_valid_onset": onset_missing,
        "primary_ladder": primary,
        "h6_contextual_model": contextual,
        "sensitivity_unknown_comorbidity_rate": unknown,
        "bootstrap_draws_file": "results/s12_bootstrap_draws.csv",
        "sensitivity_recent_onset_excluded": recent_onset,
        "sensitivity_implausible_ages_excluded": age_sensitivity,
        "h4_composition_contrast": {
            "criterion": ("the model-5 difference is smaller than the model-2 difference and its "
                           "bootstrap CI excludes 0"),
            "model_2_difference": benchmark,
            "model_5_difference": composition,
            "smaller": bool(composition < benchmark),
            "ci": primary["model_5_age_spline"]["bootstrap"]["slope_difference_ci"],
            "holds": bool(composition < benchmark
                           and primary["model_5_age_spline"]["bootstrap"]["slope_difference_ci"][0] > 0),
            "share_of_model_2_difference_explained": float(1 - composition / benchmark),
        },
        "h6_contextual_association": {
            "criterion": ("the per-standard-deviation state-level odds ratio under the "
                           "record-level covariates has a CI excluding 1"),
            "per_sd_or": share["or"] if share else None,
            "ci": (share.get("profile_ci", share["ci"]) if share else None),
            "interval_used": share["interval_used"] if share else None,
            "wald_ci": share["ci"] if share else None,
            "evaluable": contextual["evaluable"],
            "holds": bool(share and contextual["evaluable"]
                          and not (share.get("profile_ci", share["ci"])[0] <= 1
                                   <= share.get("profile_ci", share["ci"])[1])),
            "age_term": contextual["age_term"],
            "note": ("a model that does not converge, or whose random intercept sits at its "
                      "boundary, leaves H6 not evaluable: the estimate is reported "
                      "descriptively and the registered criterion is not applied. The "
                      "registration voids a hypothesis on nonconvergence; extending that to a "
                      "boundary solution is a pre-run implementation decision recorded in "
                      "PROTOCOL_CORRECTION_ADDENDUM.md section 11, because at sigma = 0 the "
                      "interval on a state-level covariate treats records as independent"),
        },
        "h5_temporal_contrast": {
            "criterion": ("the model-6 difference is smaller than the model-5 difference and its "
                           "bootstrap CI excludes 0"),
            "model_5_difference": composition,
            "model_6_difference": temporal,
            "smaller": bool(temporal < composition),
            "ci": primary["model_6_age_spline_and_month"]["bootstrap"]["slope_difference_ci"],
            "holds": bool(temporal < composition
                           and primary["model_6_age_spline_and_month"]["bootstrap"]["slope_difference_ci"][0] > 0),
        },
    }
    print(f"\n  H4 holds: {output['h4_composition_contrast']['holds']}; "
          f"H5 holds: {output['h5_temporal_contrast']['holds']}; "
          f"H6 holds: {output['h6_contextual_association']['holds']}")

    outpath = write_result(output, OUTPUT_DIR / "s12_composition_ladder.json")
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
