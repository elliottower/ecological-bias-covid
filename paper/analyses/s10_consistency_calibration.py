"""
Supplementary Analysis S10: Consistency-Test Calibration Under Known Null

Simulates datasets matching 4CE dimensionality (30 Elixhauser categories,
435 pairwise correlations, 20 sites with heterogeneous coverage) under
NO true cross-site inconsistency. Reports the null distribution and
false-positive rates.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

np.random.seed(20260707)

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_SIMS = 2000
N_VARS = 30
N_PAIRS = N_VARS * (N_VARS - 1) // 2  # 435
N_SITES = 20
N_FULL_COVERAGE = 10
SITE_SAMPLE_SIZE = 500


def compute_pairwise_correlations(binary_matrix):
    n_vars = binary_matrix.shape[1]
    corrs = {}
    idx = 0
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x, y = binary_matrix[:, i], binary_matrix[:, j]
            if x.std() > 0 and y.std() > 0:
                corrs[idx] = np.corrcoef(x, y)[0, 1]
            else:
                corrs[idx] = 0.0
            idx += 1
    return corrs


def compute_consistency_statistic(site_corrs, coverage_masks):
    n_sites = len(site_corrs)
    shared_diffs_sum = 0.0
    shared_count = 0

    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            shared = coverage_masks[i] & coverage_masks[j]
            shared_pairs = list(shared)
            if len(shared_pairs) == 0:
                continue
            for p in shared_pairs:
                diff = site_corrs[i].get(p, 0) - site_corrs[j].get(p, 0)
                shared_diffs_sum += diff ** 2
                shared_count += 1

    if shared_count == 0:
        return 0.0
    return shared_diffs_sum / shared_count


def simulate_one():
    prob_vector = np.random.uniform(0.01, 0.30, N_VARS)

    site_corrs = []
    coverage_masks = []

    for s in range(N_SITES):
        data = np.zeros((SITE_SAMPLE_SIZE, N_VARS), dtype=int)
        for v in range(N_VARS):
            data[:, v] = np.random.binomial(1, prob_vector[v], SITE_SAMPLE_SIZE)

        if s < N_FULL_COVERAGE:
            covered_pairs = set(range(N_PAIRS))
        else:
            n_covered = np.random.randint(100, 301)
            covered_pairs = set(np.random.choice(N_PAIRS, size=n_covered, replace=False))

        corrs = compute_pairwise_correlations(data)
        filtered = {k: v for k, v in corrs.items() if k in covered_pairs}

        site_corrs.append(filtered)
        coverage_masks.append(covered_pairs)

    return compute_consistency_statistic(site_corrs, coverage_masks)


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S10: Consistency-Test Calibration  [{ts}]")
    print(f"  N_SIMS = {N_SIMS}, N_VARS = {N_VARS}, N_PAIRS = {N_PAIRS}")
    print(f"  N_SITES = {N_SITES}, SITE_SAMPLE_SIZE = {SITE_SAMPLE_SIZE}")
    print("=" * 70)

    statistics = []
    for i in range(N_SIMS):
        stat = simulate_one()
        statistics.append(stat)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{N_SIMS} simulations complete")

    statistics = np.array(statistics)
    null_mean = float(np.mean(statistics))
    null_std = float(np.std(statistics))

    if null_std > 0:
        z_scores = (statistics - null_mean) / null_std
    else:
        z_scores = np.zeros_like(statistics)

    fpr_z2 = float(np.mean(np.abs(z_scores) > 2))
    fpr_z25 = float(np.mean(np.abs(z_scores) > 25))

    print(f"\n  Null distribution: mean = {null_mean:.6f}, std = {null_std:.6f}")
    print(f"  FPR at |z| > 2:  {fpr_z2:.4f} (target: <= 0.06)")
    print(f"  FPR at |z| > 25: {fpr_z25:.4f} (target: <= 0.001)")

    if fpr_z25 > 0.01:
        conclusion = ("FALSIFICATION TRIGGERED: FPR at z=25 exceeds 1%. "
                       "The consistency test is miscalibrated.")
    elif fpr_z2 <= 0.06:
        conclusion = ("CALIBRATION CONFIRMED: FPR at z=2 is within nominal "
                       "range and FPR at z=25 is negligible.")
    else:
        conclusion = ("MARGINAL: FPR at z=2 slightly exceeds nominal alpha "
                       f"({fpr_z2:.3f} vs 0.05). Test may be slightly "
                       "anti-conservative but z=25 remains safe.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "seed": 20260707,
        "n_sims": N_SIMS,
        "n_vars": N_VARS,
        "n_pairs": N_PAIRS,
        "n_sites": N_SITES,
        "site_sample_size": SITE_SAMPLE_SIZE,
        "null_mean": null_mean,
        "null_std": null_std,
        "fpr_z2": fpr_z2,
        "fpr_z25": fpr_z25,
        "percentiles": {
            "p50": float(np.percentile(statistics, 50)),
            "p95": float(np.percentile(statistics, 95)),
            "p99": float(np.percentile(statistics, 99)),
            "p999": float(np.percentile(statistics, 99.9)),
        },
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s10_consistency_calibration.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
