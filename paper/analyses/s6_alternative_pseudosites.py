"""
Supplementary Analysis S6: Alternative Pseudo-Site Definitions

Re-partitions CDC data into random pseudo-sites and runs ecological
regression on each partition. Tests whether failure mode 1 is an
artifact of race/ethnicity grouping or a general property of n=9.

500 replications per condition with Wilson score CI on power.
"""

import json
import csv
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

np.random.seed(20260708)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CDC_CSV = DATA_DIR / "cdc_case_surveillance/cdc_full.csv"
N_REPS = 500
AGE_THRESHOLD = 80


def load_cdc_individual():
    """Stream CDC CSV and extract (elderly, died) per record."""
    print("  Loading CDC individual-level data (streaming)...")
    records = []
    n_processed = 0

    with open(CDC_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ag = row.get("age_group", "").strip()
            if not ag or ag in ("Missing", "Unknown"):
                continue

            elderly = 1 if "80" in ag else 0
            died = 1 if row.get("death_yn", "").strip() == "Yes" else 0
            records.append((elderly, died))

            n_processed += 1
            if n_processed % 5_000_000 == 0:
                print(f"    {n_processed:,} records processed...")

    print(f"  Loaded {len(records):,} records")
    return np.array(records, dtype=np.int8)


def run_partition(data, n_groups, rng):
    """Randomly partition records into n_groups and run ecological regression."""
    n = len(data)
    assignments = rng.integers(0, n_groups, size=n)

    elderly_props = np.zeros(n_groups)
    mortality_rates = np.zeros(n_groups)
    group_sizes = np.zeros(n_groups)

    for g in range(n_groups):
        mask = assignments == g
        group_data = data[mask]
        if len(group_data) < 100:
            return None
        group_sizes[g] = len(group_data)
        elderly_props[g] = group_data[:, 0].mean()
        mortality_rates[g] = group_data[:, 1].mean()

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        elderly_props, mortality_rates)

    return {
        "slope": float(slope),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }


def wilson_ci(n_success, n_total, z=1.96):
    """Wilson score confidence interval for a proportion."""
    p = n_success / n_total
    denom = 1 + z ** 2 / n_total
    center = (p + z ** 2 / (2 * n_total)) / denom
    spread = z * np.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return max(0, center - spread), min(1, center + spread)


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S6: Alternative Pseudo-Site Definitions  [{ts}]")
    print(f"  N_REPS = {N_REPS}")
    print("=" * 70)

    data = load_cdc_individual()
    rng = np.random.default_rng(20260708)

    results = {}

    for n_groups in [9, 50]:
        print(f"\n  Random partition into {n_groups} groups "
              f"({N_REPS} replications)...")
        sig_count = 0
        slopes = []
        valid = 0

        for i in range(N_REPS):
            res = run_partition(data, n_groups, rng)
            if res is None:
                continue
            valid += 1
            if res["significant"]:
                sig_count += 1
            slopes.append(res["slope"])

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{N_REPS} done, "
                      f"power so far = {sig_count/valid:.3f}")

        power = sig_count / valid if valid > 0 else 0
        ci_lower, ci_upper = wilson_ci(sig_count, valid)

        results[f"random_{n_groups}"] = {
            "n_groups": n_groups,
            "n_reps": N_REPS,
            "n_valid": valid,
            "n_significant": sig_count,
            "power": power,
            "wilson_ci_lower": float(ci_lower),
            "wilson_ci_upper": float(ci_upper),
            "mean_slope": float(np.mean(slopes)) if slopes else None,
            "std_slope": float(np.std(slopes)) if slopes else None,
        }

        print(f"    Power = {power:.3f} "
              f"(95% Wilson CI [{ci_lower:.3f}, {ci_upper:.3f}])")

    power_9 = results["random_9"]["power"]
    ci_lower_9 = results["random_9"]["wilson_ci_lower"]

    if power_9 > 0.80 and ci_lower_9 > 0.75:
        conclusion = ("FALSIFICATION: Random partitions into 9 groups "
                       f"achieve power={power_9:.2f} (CI lower={ci_lower_9:.2f}). "
                       "The failure is specific to race/ethnicity grouping, "
                       "not ecological regression at n=9.")
    else:
        conclusion = (f"Random partitions into 9 groups achieve "
                       f"power={power_9:.2f} (CI [{ci_lower_9:.2f}, "
                       f"{results['random_9']['wilson_ci_upper']:.2f}]). "
                       f"Ecological regression at n=9 struggles regardless "
                       f"of grouping strategy.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "seed": 20260708,
        "n_reps": N_REPS,
        "n_records": len(data),
        "results": results,
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s6_alternative_pseudosites.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
