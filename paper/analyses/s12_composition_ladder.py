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
from scipy import stats

import mexico_confirmed_cases
from paths import RESULTS

OUTPUT_DIR = RESULTS
COMORBIDITIES = [c.lower() for c in mexico_confirmed_cases.COMORBIDITY_COLS]
SPLINE_PERCENTILES = [5, 27.5, 50, 72.5, 95]
AGE_BAND_EDGES = list(range(0, 130, 10))
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_DRAWS = 2000
RECENT_ONSET_DAYS = 28
SNAPSHOT_DATE = "2022-01-03"
IMPLAUSIBLE_AGE = 110


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


def fit_and_predict(cells, formula):
    """Binomial GLM on the cells, returning predicted probability per cell row."""
    pooled = cells.groupby([c for c in cells.columns if c not in ("site", "deaths", "n")],
                           as_index=False).agg(deaths=("deaths", "sum"), n=("n", "sum"))
    design = patsy.dmatrix(formula, pooled, return_type="dataframe")
    endog = np.column_stack([pooled["deaths"], pooled["n"] - pooled["deaths"]]).astype(float)
    model = sm.GLM(endog, design.to_numpy(), family=sm.families.Binomial())
    fit = model.fit(maxiter=200)
    if not converged(fit):
        fit = model.fit(method="lbfgs", maxiter=1000)
        if not converged(fit):
            raise RuntimeError(f"{formula} did not converge under IRLS or lbfgs")
    predicted = pd.Series(fit.predict(design.to_numpy()), index=pooled.index)
    pooled = pooled.assign(predicted=predicted)
    key = [c for c in pooled.columns if c not in ("deaths", "n", "predicted")]
    return cells.merge(pooled[key + ["predicted"]], on=key, how="left"), len(design.columns)


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
        """Fit on states weighted by their bootstrap multiplicity; return the slope difference."""
        deaths = (weights @ self.deaths).astype(np.float64)
        n = (weights @ self.counts).astype(np.float64)
        present = n > 0  # cells with no records in this draw carry no likelihood
        endog = np.column_stack([deaths[present], n[present] - deaths[present]])
        fit = sm.GLM(endog, self.design[present], family=sm.families.Binomial()).fit(
            start_params=self.start_params, maxiter=200)
        if self.start_params is None:
            self.start_params = fit.params
        if not converged(fit):
            return np.nan
        probability = fit.predict(self.design)
        implied = (self.counts @ probability) / self.state_n
        drawn = np.repeat(np.arange(len(self.states)), weights.astype(int))
        observed_slope = stats.linregress(self.state_elderly[drawn], self.observed[drawn]).slope
        implied_slope = stats.linregress(self.state_elderly[drawn], implied[drawn]).slope
        return observed_slope - implied_slope


def bootstrap_ladder(df, formulas, covariate_sets, seed, draws):
    """State bootstrap for every model, refitting inside each draw.

    One model is built, bootstrapped and released before the next is built, so the
    largest cell matrices are never in memory together. Every model sees the same
    sequence of state multiplicities, because each starts from the same seed.
    """
    collected = {}
    for (name, formula, needs_onset), covariates in zip(formulas, covariate_sets):
        started = time.time()
        model = BootstrapModel(df, formula, covariates, needs_onset)
        print(f"    {name}: {model.design.shape[0]:,} cells x {model.design.shape[1]} terms, "
              f"built in {time.time() - started:.0f}s", flush=True)

        rng = np.random.default_rng(seed)
        states = len(model.states)
        values = []
        for draw in range(draws):
            weights = np.bincount(rng.integers(0, states, size=states),
                                  minlength=states).astype(np.float32)
            values.append(model.draw(weights))
            if draw and draw % 250 == 0:
                print(f"      {name} draw {draw}/{draws} [{datetime.now():%H:%M:%S}]", flush=True)
        collected[name] = np.array(values)
        print(f"    {name}: {draws} draws in {time.time() - started:.0f}s, "
              f"CI {[round(x, 4) for x in interval(collected[name])]}", flush=True)
        del model
        gc.collect()
    return collected


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


def main():
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

    print(f"\n  state bootstrap, {BOOTSTRAP_DRAWS} draws, refitting every model in each draw:")
    draws = bootstrap_ladder(df, formulas, covariate_sets, BOOTSTRAP_SEED, BOOTSTRAP_DRAWS)
    for name, values in draws.items():
        primary[name]["bootstrap"] = {
            "draws": int(len(values)),
            "draws_failed_to_converge": int(np.isnan(values).sum()),
            "slope_difference_ci": interval(values),
            "mc_se": float(np.nanstd(values, ddof=1) / np.sqrt(np.sum(~np.isnan(values)))),
        }
        print(f"    {name}: CI {[round(x, 4) for x in interval(values)]}")

    composition = primary["model_5_age_spline"]["slope_difference"]
    benchmark = primary["model_2_elderly_sex_any_comorbidity"]["slope_difference"]
    temporal = primary["model_6_age_spline_and_month"]["slope_difference"]
    output = {
        "timestamp": ts,
        "registration": {"file": "ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md",
                          "commit": "a38196b", "tag": "registration-s11-s12"},
        "spline_knots": [float(k) for k in knots],
        "records_without_valid_onset": onset_missing,
        "primary_ladder": primary,
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
          f"H5 holds: {output['h5_temporal_contrast']['holds']}")

    outpath = OUTPUT_DIR / "s12_composition_ladder.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
