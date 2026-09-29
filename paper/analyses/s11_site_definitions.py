"""
Extension S11: every defensible site definition, scored under one rule

Registered in ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md, frozen at commit a38196b
(tag registration-s11-s12), before any grouping other than the treating unit's
state had been run.

For each definition the script computes what S3 computes: the observed ecological
slope of site mortality on site elderly share, the slope implied by the
record-level model averaged over each site's own records, their difference and
their ratio. The slope difference is primary; the ratio is reported only where the
resampling distribution of the implied slope excludes zero.
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tools.sm_exceptions import PerfectSeparationError

import mexico_confirmed_cases
from paths import RESULTS, require_clean_tree, run_metadata, write_result, write_table

OUTPUT_DIR = RESULTS
COVARIATES = ["elderly", "male", "has_comorbidity"]
MIN_SITE_RECORDS = 1000
MUNICIPALITY_THRESHOLDS = [1000, 500, 250]
PARTITION_SEED = 20260924
BOOTSTRAP_SEED = 20260925
PARTITION_REPLICATIONS = 500
BOOTSTRAP_DRAWS = 2000
STATE_DIFFERENCE = 0.8568009064494755  # S3, treating-unit states; the H1 margin is 20% of this
H1_MARGIN = 0.2 * STATE_DIFFERENCE
RATIO_REPORTABLE = "the resampling distribution of the implied slope excludes zero"
MINIMUM_VALID_DRAWS = 100  # below this an interval is not reported at all
INSTABILITY_RATE = 0.01  # a bootstrap with more than this share of failed draws is unstable
MINIMUM_EXPOSURE_SD = 1e-6  # a draw with no elderly-share variation carries no slope
FIT_FAILURES = (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError)
ANALYSES = ["treating_state", "residence_state", "sector", "municipality_1000",
            "municipality_500", "municipality_250", "municipality_all_valid", "municipality_paired"]


def cells_by(df, group_column):
    """Deaths and records per group x covariate cell."""
    return df.groupby([group_column] + COVARIATES, as_index=False).agg(
        deaths=("died", "sum"), n=("died", "size")).rename(columns={group_column: "group"})


def fit_record_model(cells):
    """Binomial GLM on covariate cells; the covariates are binary, so this is the record-level fit."""
    pooled = cells.groupby(COVARIATES, as_index=False).agg(deaths=("deaths", "sum"), n=("n", "sum"))
    pooled["alive"] = pooled["n"] - pooled["deaths"]
    X = np.column_stack([np.ones(len(pooled))] +
                        [pooled[c].to_numpy(dtype=float) for c in COVARIATES])
    fit = sm.GLM(pooled[["deaths", "alive"]].to_numpy(dtype=float), X,
                 family=sm.families.Binomial()).fit()
    if not fit.converged:
        raise RuntimeError("record-level model did not converge")
    keys = pooled[COVARIATES].apply(tuple, axis=1)
    return dict(zip(keys, fit.predict(X))), {
        "intercept": float(fit.params[0]),
        **{f"beta_{c}": float(fit.params[i + 1]) for i, c in enumerate(COVARIATES)},
    }


def site_table(cells, predicted):
    """Per site: records, deaths, observed and implied mortality, elderly share."""
    keys = cells[COVARIATES].apply(tuple, axis=1)
    cells = cells.assign(predicted_deaths=keys.map(predicted) * cells["n"])
    table = cells.groupby("group").apply(
        lambda g: pd.Series({
            "n": g["n"].sum(),
            "deaths": g["deaths"].sum(),
            "observed_mortality": g["deaths"].sum() / g["n"].sum(),
            "implied_mortality": g["predicted_deaths"].sum() / g["n"].sum(),
            "prop_elderly": (g["n"] * g["elderly"]).sum() / g["n"].sum(),
        }), include_groups=False)
    return table


def discrepancy(table):
    """Observed, implied and difference under unweighted OLS, binomial and weighted fits."""
    observed = stats.linregress(table["prop_elderly"], table["observed_mortality"])
    implied = stats.linregress(table["prop_elderly"], table["implied_mortality"])
    weighted = sm.WLS(table["observed_mortality"], sm.add_constant(table["prop_elderly"]),
                      weights=table["n"]).fit()
    weighted_implied = sm.WLS(table["implied_mortality"], sm.add_constant(table["prop_elderly"]),
                              weights=table["n"]).fit()
    binomial = sm.GLM(np.column_stack([table["deaths"], table["n"] - table["deaths"]]),
                      sm.add_constant(table["prop_elderly"]),
                      family=sm.families.Binomial()).fit()
    binomial_implied = sm.GLM(table["implied_mortality"], sm.add_constant(table["prop_elderly"]),
                              family=sm.families.Binomial(), var_weights=table["n"]).fit()
    return {
        "observed_slope": float(observed.slope),
        "implied_slope": float(implied.slope),
        "slope_difference": float(observed.slope - implied.slope),
        "observed_p": float(observed.pvalue),
        "observed_r_squared": float(observed.rvalue ** 2),
        "weighted_observed_slope": float(weighted.params.iloc[1]),
        "weighted_implied_slope": float(weighted_implied.params.iloc[1]),
        "weighted_slope_difference": float(weighted.params.iloc[1] - weighted_implied.params.iloc[1]),
        "binomial_log_odds_slope": float(binomial.params.iloc[1]),
        "binomial_se": float(binomial.bse.iloc[1]),
        "binomial_implied_log_odds_slope": float(binomial_implied.params.iloc[1]),
        "binomial_implied_note": ("a binomial-deviance projection of model-implied mortality onto "
                                   "elderly share, weighted by site records; its standard error is "
                                   "not sampling uncertainty, because the response comes from a "
                                   "fitted model, so uncertainty comes from the bootstrap"),
        "binomial_log_odds_difference": float(binomial.params.iloc[1]
                                               - binomial_implied.params.iloc[1]),
        "n_sites": int(len(table)),
        "n_records": int(table["n"].sum()),
        "n_deaths": int(table["deaths"].sum()),
        "median_site_size": float(table["n"].median()),
        "elderly_share_min": float(table["prop_elderly"].min()),
        "elderly_share_max": float(table["prop_elderly"].max()),
        "elderly_share_sd": float(table["prop_elderly"].std(ddof=1)),
    }


def leave_one_out(cells):
    """Slope difference with each site dropped in turn, and each site's leverage."""
    groups = sorted(cells["group"].unique())
    differences = {}
    for dropped in groups:
        kept = cells[cells["group"] != dropped]
        predicted, _ = fit_record_model(kept)
        table = site_table(kept, predicted)
        observed = stats.linregress(table["prop_elderly"], table["observed_mortality"]).slope
        implied = stats.linregress(table["prop_elderly"], table["implied_mortality"]).slope
        differences[str(dropped)] = float(observed - implied)

    predicted, _ = fit_record_model(cells)
    table = site_table(cells, predicted)
    share = table["prop_elderly"].to_numpy()  # leverage from the observed ecological regression
    design = np.column_stack([np.ones(len(share)), share])
    hat = design @ np.linalg.solve(design.T @ design, design.T)
    full = float(stats.linregress(table["prop_elderly"], table["observed_mortality"]).slope
                 - stats.linregress(table["prop_elderly"], table["implied_mortality"]).slope)
    changes = {site: value - full for site, value in differences.items()}
    worst = max(changes, key=lambda k: abs(changes[k]))
    return {
        "full_slope_difference": full,
        "slope_difference_without_each_site": differences,
        "deletion_change": changes,
        "largest_deletion_change": {"site": worst, "change": changes[worst]},
        "range": [min(differences.values()), max(differences.values())],
        "leverage": {str(g): float(h) for g, h in zip(table.index, np.diag(hat))},
        "site_records": {str(g): int(n) for g, n in zip(table.index, table["n"])},
    }


def bootstrap_sites(cells, groups, rng, draws):
    """Resample sites with replacement, refitting the record-level model each draw."""
    differences, ratios, implied_slopes = [], [], []
    failures = {"fit": 0, "no_exposure_variation": 0}
    indexed = {g: c for g, c in cells.groupby("group")}
    for _ in range(draws):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        resampled = pd.concat([indexed[g].assign(group=f"{g}_{i}") for i, g in enumerate(drawn)],
                              ignore_index=True)
        try:
            predicted, _ = fit_record_model(resampled)
        except FIT_FAILURES:
            failures["fit"] += 1
            differences.append(np.nan); ratios.append(np.nan); implied_slopes.append(np.nan)
            continue
        table = site_table(resampled, predicted)
        if table["prop_elderly"].std(ddof=1) < MINIMUM_EXPOSURE_SD:
            failures["no_exposure_variation"] += 1
            differences.append(np.nan); ratios.append(np.nan); implied_slopes.append(np.nan)
            continue
        observed = stats.linregress(table["prop_elderly"], table["observed_mortality"]).slope
        implied = stats.linregress(table["prop_elderly"], table["implied_mortality"]).slope
        differences.append(observed - implied)
        implied_slopes.append(implied)
        ratios.append(observed / implied if implied != 0 else np.nan)
    return np.array(differences), np.array(ratios), np.array(implied_slopes), failures


def failure_summary(values, failures, draws):
    """Registered instability rule: more than 1% of draws failing makes a bootstrap unstable.

    The denominator is every draw requested; no draw is regenerated to replace a failure,
    so an interval computed with failures present is conditional on the valid draws.
    """
    failed = int(np.isnan(np.asarray(values, dtype=float)).sum())
    valid = draws - failed
    return {
        "draws": draws,
        "valid_draws": valid,
        "failed_draws": failed,
        "failure_reasons": failures,
        "unstable": bool(failed > INSTABILITY_RATE * draws or valid < MINIMUM_VALID_DRAWS),
        "minimum_valid_draws": MINIMUM_VALID_DRAWS,
        "interval_is_conditional_on_valid_draws": bool(failed > 0),
    }


def interval_or_none(values):
    """A percentile interval, or nothing when too few draws survived to support one."""
    values = np.asarray(values, dtype=float)
    if int(np.sum(~np.isnan(values))) < MINIMUM_VALID_DRAWS:
        return None
    return interval(values)


def interval(values):
    values = np.asarray(values, dtype=float)
    if not np.any(~np.isnan(values)):
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def monte_carlo_se(values):
    values = np.asarray(values, dtype=float)
    valid = int(np.sum(~np.isnan(values)))
    if valid < 2:
        return float("nan")
    return float(np.nanstd(values, ddof=1) / np.sqrt(valid))


def report_ratio(ratios, implied_draws):
    """The registered rule: a ratio only where the implied slope's distribution excludes zero."""
    bounds = interval_or_none(implied_draws)
    if bounds is None or not np.all(np.isfinite(bounds)):
        return {"reportable": False, "reason": "too few valid draws to bound the implied slope"}
    if bounds[0] <= 0 <= bounds[1]:
        return {"reportable": False, "reason": "implied-slope distribution includes zero"}
    return {"reportable": True, "rule": RATIO_REPORTABLE, "ci": interval_or_none(ratios)}


def analyse_definition(df, group_column, rng, minimum=MIN_SITE_RECORDS, draws=BOOTSTRAP_DRAWS):
    sizes = df.groupby(group_column).size()
    keep = sizes[sizes >= minimum].index
    retained = df[df[group_column].isin(keep)]
    cells = cells_by(retained, group_column)
    predicted, coefficients = fit_record_model(cells)
    table = site_table(cells, predicted)

    result = discrepancy(table)
    result["coefficients"] = coefficients
    result["minimum_site_records"] = minimum
    result["sites_before_size_rule"] = int(len(sizes))
    result["records_excluded_by_size_rule"] = int(len(df) - len(retained))
    result["treating_states_represented"] = int(retained["site"].nunique())
    result["residence_states_represented"] = int(retained["site_residence"].nunique())

    groups = np.array(sorted(table.index))
    differences, ratios, implied_slopes, failures = bootstrap_sites(cells, groups, rng, draws)
    result["bootstrap"] = {
        **failure_summary(differences, failures, draws),
        "slope_difference_ci": interval_or_none(differences),
        "slope_difference_mc_se": monte_carlo_se(differences),
        "implied_slope_ci": interval_or_none(implied_slopes),
    }
    result["ratio"] = report_ratio(ratios, implied_slopes)
    result["ratio"]["point"] = float(result["observed_slope"] / result["implied_slope"])
    return result, cells, table


def analyse_municipalities(df, streams, draws=BOOTSTRAP_DRAWS):
    """Municipality discrepancy at each registered threshold, with clustered intervals.

    Records whose municipality code is a sentinel carry no municipality and are
    excluded here; the loader audit counts them.
    """
    known = df[df["municipality"].notna()].copy()
    known["municipality"] = known["municipality"].astype(int)
    out = {"records_without_a_municipality_code": int(len(df) - len(known))}

    for threshold in MUNICIPALITY_THRESHOLDS:
        sizes = known.groupby("municipality").size()
        keep = sizes[sizes >= threshold].index
        retained = known[known["municipality"].isin(keep)]
        if len(keep) < 3:
            out[f"threshold_{threshold}"] = {
                "threshold": threshold, "n_sites": int(len(keep)),
                "municipalities_before_rule": int(len(sizes)),
                "note": "fewer than three municipalities meet this rule; no slope is defined",
            }
            continue
        cells = cells_by(retained, "municipality")
        predicted, _ = fit_record_model(cells)
        table = site_table(cells, predicted)
        entry = discrepancy(table)
        entry["threshold"] = threshold
        entry["municipalities_before_rule"] = int(len(sizes))
        entry["states_represented"] = int(retained["site_residence"].nunique())
        entry["states_absent"] = sorted(set(range(1, 33)) - set(retained["site_residence"].unique()))
        entry["records_excluded_by_size_rule"] = int(len(known) - len(retained))
        entry["clustered_bootstrap"] = clustered_municipality_interval(
            retained, streams[f"municipality_{threshold}"], draws)
        out[f"threshold_{threshold}"] = entry

    cells = cells_by(known, "municipality")
    predicted, _ = fit_record_model(cells)
    table = site_table(cells, predicted)
    grouped = discrepancy(table)
    grouped["note"] = ("every valid municipality, no size rule; read the grouped-binomial slopes, "
                        "because equal-site rates are unstable in the smallest municipalities")
    grouped["clustered_bootstrap"] = clustered_binomial_interval(
        known, streams["municipality_all_valid"], draws)
    out["all_valid_municipalities"] = grouped
    return out


def clustered_binomial_interval(retained, rng, draws):
    """State-clustered bootstrap of the grouped-binomial discrepancy, on its own scale.

    The unthresholded municipality analysis is registered as grouped-binomial, so its
    uncertainty is resampled on the log-odds scale the point estimate is reported on,
    with the record-level model refitted inside every draw.
    """
    cells = retained.groupby(["site_residence", "municipality"] + COVARIATES,
                             as_index=False).agg(deaths=("died", "sum"), n=("died", "size"))
    by_state = {s: g for s, g in cells.groupby("site_residence")}
    states = np.array(sorted(by_state))
    observed_slopes, implied_slopes, differences = [], [], []
    failures: dict[str, int] = {}

    def count(reason):
        failures[reason] = failures.get(reason, 0) + 1
        observed_slopes.append(np.nan); implied_slopes.append(np.nan)
        differences.append(np.nan)

    for _ in range(draws):
        drawn = rng.choice(states, size=len(states), replace=True)
        frame = pd.concat([by_state[s].assign(
            municipality=by_state[s]["municipality"].astype(str) + f"_{i}")
            for i, s in enumerate(drawn)], ignore_index=True)
        frame = frame.rename(columns={"municipality": "group"})
        try:
            predicted, _ = fit_record_model(frame)
        except FIT_FAILURES as error:
            count(f"fit: {type(error).__name__}")
            continue
        table = site_table(frame, predicted)
        if table["prop_elderly"].std(ddof=1) < MINIMUM_EXPOSURE_SD:
            count("no_exposure_variation")
            continue
        exog = sm.add_constant(table["prop_elderly"])
        try:
            observed = sm.GLM(np.column_stack([table["deaths"], table["n"] - table["deaths"]]),
                              exog, family=sm.families.Binomial()).fit()
            implied = sm.GLM(table["implied_mortality"], exog, family=sm.families.Binomial(),
                             var_weights=table["n"]).fit()
        except FIT_FAILURES as error:
            count(f"binomial projection: {type(error).__name__}")
            continue
        observed_slopes.append(float(observed.params.iloc[1]))
        implied_slopes.append(float(implied.params.iloc[1]))
        differences.append(float(observed.params.iloc[1] - implied.params.iloc[1]))

    return {
        **failure_summary(differences, failures, draws),
        "scale": "binomial log odds, the scale the unthresholded analysis is registered on",
        "observed_slope_ci": interval_or_none(observed_slopes),
        "implied_slope_ci": interval_or_none(implied_slopes),
        "slope_difference_ci": interval_or_none(differences),
        "slope_difference_mc_se": monte_carlo_se(differences),
        "resampling": "states of residence drawn with replacement, municipalities kept within them",
    }


def clustered_municipality_interval(retained, rng, draws):
    """State-clustered bootstrap of the municipality discrepancy itself."""
    cells = retained.groupby(["site_residence", "municipality"] + COVARIATES,
                             as_index=False).agg(deaths=("died", "sum"), n=("died", "size"))
    by_state = {s: g for s, g in cells.groupby("site_residence")}
    states = np.array(sorted(by_state))
    differences, implied_slopes = [], []
    failures = {"fit": 0, "no_exposure_variation": 0}
    for _ in range(draws):
        drawn = rng.choice(states, size=len(states), replace=True)
        frame = pd.concat([by_state[s].assign(
            municipality=by_state[s]["municipality"].astype(str) + f"_{i}")
            for i, s in enumerate(drawn)], ignore_index=True)
        frame = frame.rename(columns={"municipality": "group"})
        try:
            predicted, _ = fit_record_model(frame)
        except FIT_FAILURES:
            failures["fit"] += 1
            differences.append(np.nan); implied_slopes.append(np.nan)
            continue
        table = site_table(frame, predicted)
        if table["prop_elderly"].std(ddof=1) < MINIMUM_EXPOSURE_SD:
            failures["no_exposure_variation"] += 1
            differences.append(np.nan); implied_slopes.append(np.nan)
            continue
        observed = stats.linregress(table["prop_elderly"], table["observed_mortality"]).slope
        implied = stats.linregress(table["prop_elderly"], table["implied_mortality"]).slope
        differences.append(observed - implied)
        implied_slopes.append(implied)
    return {
        **failure_summary(differences, failures, draws),
        "slope_difference_ci": interval_or_none(differences),
        "slope_difference_mc_se": monte_carlo_se(differences),
        "implied_slope_ci": interval_or_none(implied_slopes),
        "resampling": "states of residence drawn with replacement, municipalities kept within them",
    }


def paired_municipality_state(df, rng, draws=BOOTSTRAP_DRAWS, threshold=MIN_SITE_RECORDS):
    """H3: municipality minus state discrepancy, both on the municipality-retained records.

    Resampling changes only the per-cell counts, so the cell matrices are built once
    and the record-level model is refitted on reweighted counts inside every draw.
    """
    known = df[df["municipality"].notna()].copy()
    known["municipality"] = known["municipality"].astype(int)
    sizes = known.groupby("municipality").size()
    keep = sizes[sizes >= threshold].index
    matched = known[known["municipality"].isin(keep)]

    cells = matched.groupby(["site_residence", "municipality"] + COVARIATES, as_index=False).agg(
        deaths=("died", "sum"), n=("died", "size"))
    cell_index = {combo: i for i, combo in enumerate(
        sorted(set(map(tuple, cells[COVARIATES].to_numpy()))))}
    municipalities = sorted(set(cells["municipality"]))
    municipality_index = {m: i for i, m in enumerate(municipalities)}
    municipality_home = dict(zip(cells["municipality"], cells["site_residence"]))
    states = sorted(set(cells["site_residence"]))
    state_index = {s: i for i, s in enumerate(states)}

    counts = np.zeros((len(municipalities), len(cell_index)))
    deaths = np.zeros_like(counts)
    rows = [municipality_index[m] for m in cells["municipality"].to_numpy()]
    cols = [cell_index[tuple(c)] for c in cells[COVARIATES].to_numpy()]
    np.add.at(counts, (rows, cols), cells["n"].to_numpy())
    np.add.at(deaths, (rows, cols), cells["deaths"].to_numpy())

    municipality_state = np.array([state_index[municipality_home[m]] for m in municipalities])
    design = np.column_stack([np.ones(len(cell_index))] + [
        np.array([combo[i] for combo in sorted(cell_index)], dtype=float)
        for i in range(len(COVARIATES))])

    reasons: dict[str, int] = {}

    def discrepancies(weights):
        """One draw: weights per state; returns municipality and state slope differences."""
        municipality_weight = weights[municipality_state]
        pooled_deaths = municipality_weight @ deaths
        pooled_n = municipality_weight @ counts
        present = pooled_n > 0
        try:
            fit = sm.GLM(
                np.column_stack([pooled_deaths[present], (pooled_n - pooled_deaths)[present]]),
                design[present], family=sm.families.Binomial()).fit()
        except FIT_FAILURES as error:
            reasons[f"fit: {type(error).__name__}"] = reasons.get(
                f"fit: {type(error).__name__}", 0) + 1
            return np.nan, np.nan
        if not fit.converged:
            reasons["convergence"] = reasons.get("convergence", 0) + 1
            return np.nan, np.nan
        probability = fit.predict(design)

        def slope_difference(count_matrix, death_matrix, repeats):
            n = count_matrix.sum(axis=1)
            share = (count_matrix @ design[:, 1]) / n
            observed = death_matrix.sum(axis=1) / n
            implied = (count_matrix @ probability) / n
            index = np.repeat(np.arange(len(n)), repeats)
            if len(index) < 3 or share[index].std(ddof=1) < MINIMUM_EXPOSURE_SD:
                reasons["no_exposure_variation"] = reasons.get("no_exposure_variation", 0) + 1
                return np.nan
            return (stats.linregress(share[index], observed[index]).slope
                    - stats.linregress(share[index], implied[index]).slope)

        state_counts = np.zeros((len(states), counts.shape[1]))
        state_deaths = np.zeros_like(state_counts)
        np.add.at(state_counts, municipality_state, counts)
        np.add.at(state_deaths, municipality_state, deaths)
        return (slope_difference(counts, deaths, weights[municipality_state].astype(int)),
                slope_difference(state_counts, state_deaths, weights.astype(int)))

    ones = np.ones(len(states))
    municipality_point, state_point = discrepancies(ones)
    point = {
        "grouping": ("municipality of residence against state of residence, both on the records "
                      "retained by the municipality size rule; municipalities nest in states of "
                      "residence, so the bootstrap clusters on state of residence"),
        "municipality": float(municipality_point),
        "state_of_residence": float(state_point),
        "paired_difference": float(municipality_point - state_point),
        "matched_records": int(len(matched)),
        "matched_municipalities": int(len(municipalities)),
        "matched_states": int(len(states)),
        "threshold": threshold,
    }

    paired = []
    for _ in range(draws):
        drawn = rng.integers(0, len(states), size=len(states))
        weights = np.bincount(drawn, minlength=len(states)).astype(float)
        municipality_difference, state_difference = discrepancies(weights)
        paired.append(municipality_difference - state_difference)
    paired = np.array(paired)
    point["bootstrap"] = {
        "draws": draws,
        **failure_summary(paired, reasons, draws),
        "paired_difference_ci": interval_or_none(paired),
        "paired_difference_mc_se": monte_carlo_se(paired),
        "resampling": "states drawn with replacement; municipalities kept within drawn states",
    }
    return point


def partition_scheme(df, rng, replications, preserve_composition):
    """Composition-preserving or size-matched unrestricted random partition."""
    quotas = df.groupby("site").agg(n=("died", "size"), elderly=("elderly", "sum"))
    quotas = quotas.sort_index()
    sizes = quotas["n"].to_numpy()
    elderly_quota = quotas["elderly"].to_numpy()
    n_groups = len(quotas)

    cell_code = (df["elderly"].to_numpy() * 8 + df["male"].to_numpy() * 4
                 + df["has_comorbidity"].to_numpy() * 2 + df["died"].to_numpy()).astype(np.int16)
    elderly_mask = df["elderly"].to_numpy().astype(bool)

    pooled_cells = cells_by(df, "site")
    predicted, _ = fit_record_model(pooled_cells)
    probability = np.array([predicted[(e, m, c)] for e in (0, 1) for m in (0, 1) for c in (0, 1)])

    def group_labels(counts):
        return np.repeat(np.arange(n_groups), counts)

    quota_table = [{"state": int(s), "n": int(n), "elderly": int(e)}
                   for s, n, e in zip(quotas.index, sizes, elderly_quota)]
    realized_checks = {"size_deviations": 0, "elderly_deviations": 0}
    per_replication = []
    differences, observed_slopes, implied_slopes, elderly_sds = [], [], [], []
    for _ in range(replications):
        if preserve_composition:
            codes = np.empty(len(cell_code), dtype=np.int16)
            labels = np.empty(len(cell_code), dtype=np.int32)
            for mask, counts in ((elderly_mask, elderly_quota), (~elderly_mask, sizes - elderly_quota)):
                stratum = cell_code[mask]
                order = rng.permutation(len(stratum))
                codes[mask] = stratum[order]
                labels[mask] = group_labels(counts)
        else:
            order = rng.permutation(len(cell_code))
            codes = cell_code[order]
            labels = group_labels(sizes)

        counts = np.bincount(labels * 16 + codes, minlength=n_groups * 16).reshape(n_groups, 16)
        group_n = counts.sum(axis=1)
        deaths = counts[:, 1::2].sum(axis=1)
        covariate_counts = counts[:, ::2] + counts[:, 1::2]
        observed = deaths / group_n
        implied = (covariate_counts * probability).sum(axis=1) / group_n
        share = covariate_counts[:, 4:].sum(axis=1) / group_n

        realized_checks["size_deviations"] += int(np.abs(group_n - sizes).sum())
        if preserve_composition:
            realized_elderly = covariate_counts[:, 4:].sum(axis=1)
            realized_checks["elderly_deviations"] += int(np.abs(realized_elderly - elderly_quota).sum())

        observed_slope = stats.linregress(share, observed).slope
        implied_slope = stats.linregress(share, implied).slope
        per_replication.append({"observed_slope": float(observed_slope),
                                 "implied_slope": float(implied_slope),
                                 "slope_difference": float(observed_slope - implied_slope),
                                 "elderly_share_sd": float(share.std(ddof=1))})
        observed_slopes.append(observed_slope)
        implied_slopes.append(implied_slope)
        differences.append(observed_slope - implied_slope)
        elderly_sds.append(share.std(ddof=1))

    differences = np.array(differences)
    scheme = "composition_preserving" if preserve_composition else "size_matched_unrestricted"
    write_table(pd.DataFrame(per_replication), OUTPUT_DIR / f"s11_partition_{scheme}_draws.csv")
    write_table(pd.DataFrame(quota_table), OUTPUT_DIR / "s11_partition_quotas.csv")
    return {
        "replications": replications,
        "quota_table": OUTPUT_DIR.name + "/s11_partition_quotas.csv",
        "per_replication_estimates": OUTPUT_DIR.name + f"/s11_partition_{scheme}_draws.csv",
        "realized_quota_deviations": realized_checks,
        "mean_slope_difference": float(differences.mean()),
        "median_slope_difference": float(np.median(differences)),
        "randomization_interval": interval(differences),
        "mc_se_of_mean": monte_carlo_se(differences),
        "mean_observed_slope": float(np.mean(observed_slopes)),
        "mean_implied_slope": float(np.mean(implied_slopes)),
        "mean_elderly_share_sd": float(np.mean(elderly_sds)),
        "attenuation_vs_treating_state": float(
            1 - abs(np.median(differences)) / abs(STATE_DIFFERENCE)),
    }


def disposition(criterion, bootstrap, estimate, holds_when):
    """A hypothesis outcome that separates "does not hold" from "cannot be judged"."""
    bounds = bootstrap.get("slope_difference_ci") or bootstrap.get("paired_difference_ci")
    evaluable = bool(bounds is not None and not bootstrap.get("unstable"))
    return {
        "criterion": criterion,
        "estimate": float(estimate),
        "ci": bounds,
        "valid_draws": bootstrap.get("valid_draws"),
        "unstable": bootstrap.get("unstable"),
        "evaluable": evaluable,
        "holds": bool(evaluable and holds_when(bounds)),
    }


def substreams(seed, names):
    """One generator per analysis, so adding or reordering analyses moves no other result."""
    return dict(zip(names, [np.random.default_rng(s)
                            for s in np.random.SeedSequence(seed).spawn(len(names))]))


def main():
    require_clean_tree()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S11: site definitions  [{ts}]")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    streams = substreams(BOOTSTRAP_SEED, ANALYSES)
    output = {
        "timestamp": ts,
        "run": run_metadata(f"s11_{ts.replace(' ', 'T')}"),
        "registration": {"file": "ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md",
                          "commit": "a38196b", "tag": "registration-s11-s12"},
        "seeds": {"partition": PARTITION_SEED, "bootstrap": BOOTSTRAP_SEED,
                   "substreams": ("numpy SeedSequence(20260925).spawn, one per analysis, in the "
                                   f"order {ANALYSES}")},
        "definitions": {},
    }

    for name, column, stream in [("treating_unit_state", "site", "treating_state"),
                                  ("residence_state", "site_residence", "residence_state"),
                                  ("health_care_sector", "sector", "sector")]:
        print(f"\n  {name}...")
        # residence state and sector are nullable now that invalid codes carry no value,
        # so the site key is made a plain integer before it becomes a group label.
        usable = df[df[column].notna()].copy()
        usable[column] = usable[column].astype(int)
        result, definition_cells, _ = analyse_definition(usable, column, streams[stream])
        result["records_without_this_code"] = int(len(df) - len(usable))
        if name == "health_care_sector":
            result["influence"] = leave_one_out(definition_cells)
        print(f"    observed {result['observed_slope']:+.4f}, implied {result['implied_slope']:+.4f}, "
              f"difference {result['slope_difference']:+.4f} "
              f"CI {result['bootstrap']['slope_difference_ci']}")
        output["definitions"][name] = result

    print("\n  municipality thresholds...")
    output["definitions"]["municipality"] = analyse_municipalities(df, streams)
    for key, entry in output["definitions"]["municipality"].items():
        if not isinstance(entry, dict) or "slope_difference" not in entry:
            print(f"    {key}: {entry}")
            continue
        print(f"    {key}: {entry['n_sites']} municipalities, difference "
              f"{entry['slope_difference']:+.4f}")

    print("\n  H3 paired contrast (municipality minus state, matched records)...")
    output["paired_municipality_state"] = paired_municipality_state(
        df, streams["municipality_paired"])
    print(f"    paired difference {output['paired_municipality_state']['paired_difference']:+.4f} "
          f"CI {output['paired_municipality_state']['bootstrap']['paired_difference_ci']}")

    partition_rng = np.random.default_rng(PARTITION_SEED)
    print("\n  composition-preserving random partition...")
    composition = partition_scheme(df, partition_rng, PARTITION_REPLICATIONS, True)
    print(f"    median difference {composition['median_slope_difference']:+.4f}, "
          f"interval {composition['randomization_interval']}, "
          f"attenuation {composition['attenuation_vs_treating_state']:.3f}")
    print("\n  size-matched unrestricted random partition...")
    unrestricted = partition_scheme(df, partition_rng, PARTITION_REPLICATIONS, False)
    print(f"    median difference {unrestricted['median_slope_difference']:+.4f}, "
          f"mean elderly-share SD {unrestricted['mean_elderly_share_sd']:.5f}")

    output["composition_preserving_partition"] = composition
    output["size_matched_unrestricted_partition"] = unrestricted
    output["h1"] = {
        "criterion": ("the 97.5th percentile of the composition-preserving randomization "
                       f"distribution falls below {H1_MARGIN:.4f}, 20% of the {STATE_DIFFERENCE:.4f} "
                       "measured across treating states"),
        "margin": H1_MARGIN,
        "upper_percentile": composition["randomization_interval"][1],
        "holds": bool(composition["randomization_interval"][1] < H1_MARGIN),
    }
    residence = output["definitions"]["residence_state"]
    output["h2_residence_state"] = disposition(
        criterion=("the discrepancy under residence states is of the same sign and order as "
                    "under treating states, with a bootstrap CI excluding 0"),
        bootstrap=residence["bootstrap"],
        estimate=residence["slope_difference"],
        holds_when=lambda bounds: bounds[0] > 0)
    paired = output["paired_municipality_state"]
    output["h3_paired_municipality_state"] = disposition(
        criterion=("the municipality discrepancy is smaller than the residence-state "
                    "discrepancy on the same records, with a paired CI excluding 0"),
        bootstrap=paired["bootstrap"],
        estimate=paired["paired_difference"],
        holds_when=lambda bounds: bounds[1] < 0)
    for name in ("h2_residence_state", "h3_paired_municipality_state"):
        entry = output[name]
        print(f"  {name}: evaluable {entry['evaluable']}, holds {entry['holds']}, "
              f"estimate {entry['estimate']:+.4f}")

    print(f"\n  H1 holds: {output['h1']['holds']} "
          f"(97.5th percentile {output['h1']['upper_percentile']:+.4f} against margin {H1_MARGIN:.4f})")

    outpath = write_result(output, OUTPUT_DIR / "s11_site_definitions.json")
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
