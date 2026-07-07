"""
Supplementary Analysis S9: Two-Stage vs. One-Stage IPD on Mexico

Compares three estimators on Mexico's 32 states:
  (a) Ecological: mortality ~ elderly proportion (existing beta=+1.31)
  (b) Two-stage IPD: per-state logistic, pooled via RE meta-analysis
  (c) One-stage IPD: logistic with state fixed effects on all records

Uses Mundlak fixed-effects approximation for (c) to avoid GLMM
convergence issues at 4M rows.
"""

import json
import csv
import numpy as np
from scipy import stats as scipy_stats
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MEXICO_CSV = DATA_DIR / "mexico_covid/COVID19MEXICO.csv"
AGE_THRESHOLD = 70


def load_by_state():
    """Load data grouped by state."""
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

            if state not in state_data:
                state_data[state] = {"elderly": [], "died": []}
            state_data[state]["elderly"].append(elderly)
            state_data[state]["died"].append(died)

    total = sum(len(v["died"]) for v in state_data.values())
    print(f"  Loaded {total:,} records across {len(state_data)} states")
    return state_data


def ecological_regression(state_data):
    """(a) Ecological: mortality rate ~ elderly proportion."""
    elderly_props = []
    mortality_rates = []
    for state in sorted(state_data.keys()):
        d = state_data[state]
        n = len(d["died"])
        elderly_props.append(sum(d["elderly"]) / n)
        mortality_rates.append(sum(d["died"]) / n)

    slope, intercept, r, p, se = scipy_stats.linregress(
        elderly_props, mortality_rates)

    return {
        "method": "ecological",
        "beta": float(slope),
        "se": float(se),
        "p_value": float(p),
        "r_squared": float(r ** 2),
        "n_sites": len(elderly_props),
        "note": "slope is in mortality-rate units per unit elderly-proportion",
    }


def two_stage_ipd(state_data):
    """(b) Two-stage: per-state logistic, then RE meta-analysis."""
    from statsmodels.discrete.discrete_model import Logit

    log_ors = []
    ses = []
    state_results = {}

    for state in sorted(state_data.keys()):
        d = state_data[state]
        y = np.array(d["died"], dtype=np.float64)
        x = np.column_stack([np.ones(len(y)),
                              np.array(d["elderly"], dtype=np.float64)])

        if y.sum() == 0 or y.sum() == len(y):
            continue
        if sum(d["elderly"]) == 0 or sum(d["elderly"]) == len(d["elderly"]):
            continue

        try:
            model = Logit(y, x)
            result = model.fit(method="lbfgs", maxiter=50, disp=False)
            log_or = float(result.params[1])
            se = float(result.bse[1])
            log_ors.append(log_or)
            ses.append(se)
            state_results[state] = {
                "log_or": log_or,
                "or": float(np.exp(log_or)),
                "se": se,
                "n": len(y),
            }
        except Exception:
            continue

    log_ors = np.array(log_ors)
    ses = np.array(ses)
    variances = ses ** 2
    weights_fe = 1.0 / variances
    w_sum = np.sum(weights_fe)
    fe_est = np.sum(weights_fe * log_ors) / w_sum

    q = np.sum(weights_fe * (log_ors - fe_est) ** 2)
    k = len(log_ors)
    c = w_sum - np.sum(weights_fe ** 2) / w_sum
    tau2 = max(0, (q - (k - 1)) / c)

    weights_re = 1.0 / (variances + tau2)
    w_sum_re = np.sum(weights_re)
    re_est = np.sum(weights_re * log_ors) / w_sum_re
    re_se = 1.0 / np.sqrt(w_sum_re)

    return {
        "method": "two_stage_ipd",
        "pooled_log_or": float(re_est),
        "pooled_or": float(np.exp(re_est)),
        "pooled_se": float(re_se),
        "pooled_ci": [float(np.exp(re_est - 1.96 * re_se)),
                       float(np.exp(re_est + 1.96 * re_se))],
        "tau2": float(tau2),
        "I2": float(max(0, (q - (k - 1)) / q)) if q > 0 else 0.0,
        "n_states_included": k,
        "per_state": state_results,
    }


def one_stage_ipd(state_data):
    """(c) One-stage: logistic with state fixed effects."""
    from statsmodels.discrete.discrete_model import Logit

    print("  Fitting one-stage logistic with state fixed effects...")
    all_y = []
    all_elderly = []
    all_state_idx = []

    states = sorted(state_data.keys())
    for i, state in enumerate(states):
        d = state_data[state]
        n = len(d["died"])
        all_y.extend(d["died"])
        all_elderly.extend(d["elderly"])
        all_state_idx.extend([i] * n)

    y = np.array(all_y, dtype=np.float64)
    elderly = np.array(all_elderly, dtype=np.float64)
    state_idx = np.array(all_state_idx)
    n = len(y)
    n_states = len(states)

    state_dummies = np.zeros((n, n_states - 1), dtype=np.float64)
    for i in range(n):
        idx = state_idx[i]
        if idx > 0:
            state_dummies[i, idx - 1] = 1.0

    X = np.column_stack([np.ones(n), elderly, state_dummies])

    model = Logit(y, X)
    result = model.fit(method="lbfgs", maxiter=100, disp=False)

    log_or = float(result.params[1])
    se = float(result.bse[1])

    return {
        "method": "one_stage_ipd",
        "log_or": log_or,
        "or": float(np.exp(log_or)),
        "se": se,
        "ci": [float(np.exp(log_or - 1.96 * se)),
               float(np.exp(log_or + 1.96 * se))],
        "p_value": float(result.pvalues[1]),
        "n_obs": n,
        "n_states": n_states,
        "converged": bool(result.mle_retvals.get("converged", True)),
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S9: Two-Stage vs. One-Stage IPD on Mexico  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    state_data = load_by_state()

    print("\n  (a) Ecological regression...")
    eco = ecological_regression(state_data)
    print(f"      beta = {eco['beta']:+.4f}, p = {eco['p_value']:.2e}")

    print("\n  (b) Two-stage IPD (per-state logistic + RE meta-analysis)...")
    two = two_stage_ipd(state_data)
    print(f"      Pooled OR = {two['pooled_or']:.2f} "
          f"(95% CI [{two['pooled_ci'][0]:.2f}, {two['pooled_ci'][1]:.2f}]), "
          f"I²={two['I2']:.3f}")

    print("\n  (c) One-stage IPD (logistic + state fixed effects)...")
    one = one_stage_ipd(state_data)
    print(f"      OR = {one['or']:.2f} "
          f"(95% CI [{one['ci'][0]:.2f}, {one['ci'][1]:.2f}]), "
          f"p = {one['p_value']:.2e}")

    print("\n  --- Comparison ---")
    print(f"  Ecological beta:      {eco['beta']:+.4f} (mortality-rate scale)")
    print(f"  Two-stage pooled OR:  {two['pooled_or']:.2f}")
    print(f"  One-stage OR:         {one['or']:.2f}")

    or_diff = abs(two["pooled_or"] - one["or"]) / one["or"]
    if or_diff > 0.50:
        conclusion = (f"FALSIFICATION: Two-stage OR ({two['pooled_or']:.2f}) "
                       f"differs from one-stage OR ({one['or']:.2f}) by "
                       f"{or_diff:.0%}. Heterogeneity distorts all pooling, "
                       f"not just ecological aggregation.")
    else:
        conclusion = (f"Two-stage ({two['pooled_or']:.2f}) and one-stage "
                       f"({one['or']:.2f}) agree within {or_diff:.0%}. "
                       f"Proper IPD methods recover the individual-level "
                       f"effect. Ecological regression is the problem.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "ecological": eco,
        "two_stage_ipd": two,
        "one_stage_ipd": one,
        "or_difference_fraction": float(or_diff),
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s9_two_stage_vs_one_stage.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
