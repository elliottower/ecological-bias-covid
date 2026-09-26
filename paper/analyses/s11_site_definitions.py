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

import mexico_confirmed_cases
from paths import RESULTS

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
        "n_sites": int(len(table)),
        "n_records": int(table["n"].sum()),
        "n_deaths": int(table["deaths"].sum()),
        "median_site_size": float(table["n"].median()),
        "elderly_share_min": float(table["prop_elderly"].min()),
        "elderly_share_max": float(table["prop_elderly"].max()),
        "elderly_share_sd": float(table["prop_elderly"].std(ddof=1)),
    }


def bootstrap_sites(cells, groups, rng, draws):
    """Resample sites with replacement, refitting the record-level model each draw."""
    differences, ratios, implied_slopes = [], [], []
    indexed = {g: c for g, c in cells.groupby("group")}
    for _ in range(draws):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        resampled = pd.concat([indexed[g].assign(group=f"{g}_{i}") for i, g in enumerate(drawn)],
                              ignore_index=True)
        predicted, _ = fit_record_model(resampled)
        table = site_table(resampled, predicted)
        observed = stats.linregress(table["prop_elderly"], table["observed_mortality"]).slope
        implied = stats.linregress(table["prop_elderly"], table["implied_mortality"]).slope
        differences.append(observed - implied)
        implied_slopes.append(implied)
        ratios.append(observed / implied if implied != 0 else np.nan)
    return np.array(differences), np.array(ratios), np.array(implied_slopes)


def interval(values):
    values = np.asarray(values, dtype=float)
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def monte_carlo_se(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanstd(values, ddof=1) / np.sqrt(np.sum(~np.isnan(values))))


def report_ratio(ratios, implied_draws):
    """The registered rule: a ratio only where the implied slope's distribution excludes zero."""
    lower, upper = interval(implied_draws)
    if lower <= 0 <= upper:
        return {"reportable": False, "reason": "implied-slope distribution includes zero"}
    return {"reportable": True, "rule": RATIO_REPORTABLE, "ci": interval(ratios)}


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
    differences, ratios, implied_slopes = bootstrap_sites(cells, groups, rng, draws)
    result["bootstrap"] = {
        "draws": draws,
        "slope_difference_ci": interval(differences),
        "slope_difference_mc_se": monte_carlo_se(differences),
        "implied_slope_ci": interval(implied_slopes),
    }
    result["ratio"] = report_ratio(ratios, implied_slopes)
    result["ratio"]["point"] = float(result["observed_slope"] / result["implied_slope"])
    return result, cells, table


def analyse_municipalities(df, rng):
    """Municipality discrepancy, with the paired contrast against the state discrepancy."""
    out = {}
    for threshold in MUNICIPALITY_THRESHOLDS:
        sizes = df.groupby("municipality").size()
        keep = sizes[sizes >= threshold].index
        retained = df[df["municipality"].isin(keep)]
        cells = cells_by(retained, "municipality")
        predicted, _ = fit_record_model(cells)
        table = site_table(cells, predicted)
        entry = discrepancy(table)
        entry["threshold"] = threshold
        entry["municipalities_before_rule"] = int(len(sizes))
        entry["states_represented"] = int(retained["site_residence"].nunique())
        entry["states_absent"] = sorted(set(range(1, 33)) - set(retained["site_residence"].unique()))
        entry["records_excluded_by_size_rule"] = int(len(df) - len(retained))
        out[f"threshold_{threshold}"] = entry
    return out


def paired_municipality_state(df, rng, draws=BOOTSTRAP_DRAWS, threshold=MIN_SITE_RECORDS):
    """H3: municipality minus state discrepancy, both on the municipality-retained records.

    Resampling changes only the per-cell counts, so the cell matrices are built once
    and the record-level model is refitted on reweighted counts inside every draw.
    """
    sizes = df.groupby("municipality").size()
    keep = sizes[sizes >= threshold].index
    matched = df[df["municipality"].isin(keep)]

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

    def discrepancies(weights):
        """One draw: weights per state; returns municipality and state slope differences."""
        municipality_weight = weights[municipality_state]
        pooled_deaths = municipality_weight @ deaths
        pooled_n = municipality_weight @ counts
        present = pooled_n > 0
        fit = sm.GLM(np.column_stack([pooled_deaths[present], (pooled_n - pooled_deaths)[present]]),
                     design[present], family=sm.families.Binomial()).fit()
        if not fit.converged:
            return np.nan, np.nan
        probability = fit.predict(design)

        def slope_difference(count_matrix, death_matrix, repeats):
            n = count_matrix.sum(axis=1)
            share = (count_matrix @ design[:, 1]) / n
            observed = death_matrix.sum(axis=1) / n
            implied = (count_matrix @ probability) / n
            index = np.repeat(np.arange(len(n)), repeats)
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
        "draws_failed_to_converge": int(np.isnan(paired).sum()),
        "paired_difference_ci": interval(paired),
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

        observed_slope = stats.linregress(share, observed).slope
        implied_slope = stats.linregress(share, implied).slope
        observed_slopes.append(observed_slope)
        implied_slopes.append(implied_slope)
        differences.append(observed_slope - implied_slope)
        elderly_sds.append(share.std(ddof=1))

    differences = np.array(differences)
    return {
        "replications": replications,
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


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S11: site definitions  [{ts}]")
    print("=" * 70)

    df = mexico_confirmed_cases.load()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        "timestamp": ts,
        "registration": {"file": "ANALYSIS_PROTOCOL_2_SITE_DEFINITIONS.md",
                          "commit": "a38196b", "tag": "registration-s11-s12"},
        "seeds": {"partition": PARTITION_SEED, "bootstrap": BOOTSTRAP_SEED},
        "definitions": {},
    }

    for name, column in [("treating_unit_state", "site"),
                          ("residence_state", "site_residence"),
                          ("health_care_sector", "sector")]:
        print(f"\n  {name}...")
        result, _, _ = analyse_definition(df, column, rng)
        print(f"    observed {result['observed_slope']:+.4f}, implied {result['implied_slope']:+.4f}, "
              f"difference {result['slope_difference']:+.4f} "
              f"CI {result['bootstrap']['slope_difference_ci']}")
        output["definitions"][name] = result

    print("\n  municipality thresholds...")
    output["definitions"]["municipality"] = analyse_municipalities(df, rng)
    for key, entry in output["definitions"]["municipality"].items():
        print(f"    {key}: {entry['n_sites']} municipalities, difference "
              f"{entry['slope_difference']:+.4f}")

    print("\n  H3 paired contrast (municipality minus state, matched records)...")
    output["paired_municipality_state"] = paired_municipality_state(df, rng)
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
    print(f"\n  H1 holds: {output['h1']['holds']} "
          f"(97.5th percentile {output['h1']['upper_percentile']:+.4f} against margin {H1_MARGIN:.4f})")

    outpath = OUTPUT_DIR / "s11_site_definitions.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
