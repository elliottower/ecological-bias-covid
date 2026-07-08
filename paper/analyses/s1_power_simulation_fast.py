"""
S1 (vectorized): Power vs. Bias Decomposition

Same analysis as s1_power_simulation.py but vectorized — samples site-level
mortality directly via the 8 covariate cells (2^3 for elderly/comorbidity/male)
instead of generating millions of individual records.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

np.random.seed(20260707)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "cdc_case_surveillance/results/cdc_ecological_fallacy.json") as f:
    cdc = json.load(f)

INDIVIDUAL_OR_AGE = cdc["individual_or_80plus"]
INDIVIDUAL_OR_COMORBIDITY = 1.5
INDIVIDUAL_OR_MALE = 1.7
BASELINE_INTERCEPT = -4.0

N_SIMS = 2000
N_SITES_RANGE = [9, 20, 50, 100, 500]

CDC_SITES = []
for s in cdc["site_data"]:
    CDC_SITES.append({
        "name": s["site"],
        "n": s["n"],
        "elderly": s["prop_80plus"],
        "comorbidity": s["prop_comorbidity"],
        "male": s["prop_male"],
    })

POPULATION_MEAN_ELDERLY = np.mean([s["elderly"] for s in CDC_SITES])
POPULATION_MEAN_COMORBIDITY = np.mean([s["comorbidity"] for s in CDC_SITES])
POPULATION_MEAN_MALE = np.mean([s["male"] for s in CDC_SITES])

CELLS = np.array([[e, c, m] for e in [0, 1] for c in [0, 1] for m in [0, 1]],
                  dtype=np.float64)
CELL_LOGITS = (BASELINE_INTERCEPT
               + np.log(INDIVIDUAL_OR_AGE) * CELLS[:, 0]
               + np.log(INDIVIDUAL_OR_COMORBIDITY) * CELLS[:, 1]
               + np.log(INDIVIDUAL_OR_MALE) * CELLS[:, 2])
CELL_MORTALITIES = 1.0 / (1.0 + np.exp(-CELL_LOGITS))


def cell_proportions(elderly_prop, comorbidity_prop, male_prop):
    props = np.empty(8)
    for i, (e, c, m) in enumerate(CELLS):
        props[i] = ((elderly_prop if e else 1 - elderly_prop) *
                    (comorbidity_prop if c else 1 - comorbidity_prop) *
                    (male_prop if m else 1 - male_prop))
    return props


def simulate_sites_vectorized(site_specs, n_sims):
    """Simulate n_sims replications of ecological regression over sites.

    Returns array of (slope, p_value) for each sim.
    """
    n_sites = len(site_specs)
    elderly_props = np.array([s["elderly"] for s in site_specs])
    mortalities = np.zeros((n_sims, n_sites))

    for j, s in enumerate(site_specs):
        cp = cell_proportions(s["elderly"], s["comorbidity"], s["male"])
        n = s["n"]
        for sim in range(n_sims):
            n_per_cell = np.random.multinomial(n, cp)
            deaths = sum(np.random.binomial(nc, cm)
                         for nc, cm in zip(n_per_cell, CELL_MORTALITIES))
            mortalities[sim, j] = deaths / n

    results = []
    for sim in range(n_sims):
        slope, _, _, p_value, _ = stats.linregress(
            elderly_props, mortalities[sim])
        results.append({"slope": float(slope), "p_value": float(p_value),
                        "significant": bool(p_value < 0.05)})
    return results


def summarize(results):
    n = len(results)
    sig = sum(1 for r in results if r["significant"])
    slopes = [r["slope"] for r in results]
    pvals = [r["p_value"] for r in results]
    return {
        "power": sig / n,
        "mean_slope": float(np.mean(slopes)),
        "median_slope": float(np.median(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_p": float(np.mean(pvals)),
        "median_p": float(np.median(pvals)),
        "n_sims": n,
    }


def condition_a():
    return simulate_sites_vectorized(CDC_SITES, N_SIMS)


def condition_b(n_sites):
    idx = np.random.choice(len(CDC_SITES), size=(N_SIMS, n_sites), replace=True)
    all_results = []
    for sim in range(N_SIMS):
        sites = [CDC_SITES[i] for i in idx[sim]]
        elderly_props = np.array([s["elderly"] for s in sites])
        cp_list = [cell_proportions(s["elderly"], s["comorbidity"], s["male"])
                   for s in sites]
        morts = []
        for j, s in enumerate(sites):
            n_per_cell = np.random.multinomial(s["n"], cp_list[j])
            deaths = sum(np.random.binomial(nc, cm)
                         for nc, cm in zip(n_per_cell, CELL_MORTALITIES))
            morts.append(deaths / s["n"])
        slope, _, _, p_value, _ = stats.linregress(elderly_props, morts)
        all_results.append({"slope": float(slope), "p_value": float(p_value),
                            "significant": bool(p_value < 0.05)})
    return all_results


def condition_c(n_sites):
    elderly_vals = [s["elderly"] for s in CDC_SITES]
    comorbidity_vals = [s["comorbidity"] for s in CDC_SITES]
    male_vals = [s["male"] for s in CDC_SITES]
    size_vals = [s["n"] for s in CDC_SITES]

    all_results = []
    for sim in range(N_SIMS):
        e_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
        c_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
        m_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
        s_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)

        elderly_props = np.array([elderly_vals[i] for i in e_idx])
        morts = []
        for j in range(n_sites):
            ep = elderly_vals[e_idx[j]]
            cp_val = comorbidity_vals[c_idx[j]]
            mp = male_vals[m_idx[j]]
            n = size_vals[s_idx[j]]
            cp = cell_proportions(ep, cp_val, mp)
            n_per_cell = np.random.multinomial(n, cp)
            deaths = sum(np.random.binomial(nc, cm)
                         for nc, cm in zip(n_per_cell, CELL_MORTALITIES))
            morts.append(deaths / n)
        slope, _, _, p_value, _ = stats.linregress(elderly_props, morts)
        all_results.append({"slope": float(slope), "p_value": float(p_value),
                            "significant": bool(p_value < 0.05)})
    return all_results


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S1: Power vs. Bias Decomposition (vectorized)  [{ts}]")
    print(f"  Individual-level ORs: age={INDIVIDUAL_OR_AGE:.2f}, "
          f"comorbidity={INDIVIDUAL_OR_COMORBIDITY}, male={INDIVIDUAL_OR_MALE}")
    print(f"  N_SIMS = {N_SIMS} per condition")
    print(f"  CDC sites: {len(CDC_SITES)}")
    print(f"  Population means: elderly={POPULATION_MEAN_ELDERLY:.4f}, "
          f"comorbidity={POPULATION_MEAN_COMORBIDITY:.4f}, "
          f"male={POPULATION_MEAN_MALE:.4f}")
    print("=" * 70)

    results = {}

    print("\n[a] CDC-matched (n=9, exact compositions and sizes)")
    results["cdc_matched_9"] = summarize(condition_a())
    print(f"    power={results['cdc_matched_9']['power']:.3f}  "
          f"mean_slope={results['cdc_matched_9']['mean_slope']:+.4f}")

    for n_sites in N_SITES_RANGE:
        if n_sites == 9:
            continue
        label = f"cdc_resampled_{n_sites}"
        print(f"\n[b] CDC-resampled (n={n_sites})")
        results[label] = summarize(condition_b(n_sites))
        print(f"    power={results[label]['power']:.3f}  "
              f"mean_slope={results[label]['mean_slope']:+.4f}")

    for n_sites in N_SITES_RANGE:
        label = f"independent_{n_sites}"
        print(f"\n[c] Independent confounders (n={n_sites})")
        results[label] = summarize(condition_c(n_sites))
        print(f"    power={results[label]['power']:.3f}  "
              f"mean_slope={results[label]['mean_slope']:+.4f}")

    print("\n" + "=" * 70)
    print("DECISION RULE CHECK")
    print("=" * 70)

    power_b_50 = results.get("cdc_resampled_50", {}).get("power", None)
    power_b_500 = results.get("cdc_resampled_500", {}).get("power", None)
    power_c_500 = results.get("independent_500", {}).get("power")
    power_a = results["cdc_matched_9"]["power"]

    print(f"\n  CDC-matched n=9:          power = {power_a:.3f}")
    if power_b_50 is not None:
        print(f"  CDC-resampled n=50:       power = {power_b_50:.3f}")
    if power_b_500 is not None:
        print(f"  CDC-resampled n=500:      power = {power_b_500:.3f}")
    if power_c_500 is not None:
        print(f"  Independent conf n=500:   power = {power_c_500:.3f}")

    if power_b_50 is not None and power_b_50 > 0.80:
        conclusion = ("FALSIFICATION TRIGGERED: Power > 80% at n<=50 under "
                       "CDC-matched confounding. The CDC null is primarily "
                       "a power problem, not bias.")
    elif (power_b_500 is not None and power_c_500 is not None
          and power_b_500 < 0.50 and power_c_500 > 0.80):
        conclusion = ("BIAS CONFIRMED: Power stays below 50% at n=500 with "
                       "CDC confounding but exceeds 80% with independent "
                       "confounders. The null is driven by correlated "
                       "confounding, not low power.")
    else:
        conclusion = ("MIXED: Neither clear falsification nor clear "
                       "confirmation. Both power and bias contribute.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "seed": 20260707,
        "note": "vectorized version — cell-level multinomial instead of per-individual",
        "individual_ors": {
            "age_80plus": INDIVIDUAL_OR_AGE,
            "comorbidity": INDIVIDUAL_OR_COMORBIDITY,
            "male": INDIVIDUAL_OR_MALE,
        },
        "n_sims": N_SIMS,
        "cdc_sites": CDC_SITES,
        "population_means": {
            "elderly": float(POPULATION_MEAN_ELDERLY),
            "comorbidity": float(POPULATION_MEAN_COMORBIDITY),
            "male": float(POPULATION_MEAN_MALE),
        },
        "results": results,
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s1_power_simulation.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
