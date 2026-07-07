"""
Orchestrate all three Visweswaran experiments end-to-end.

Usage:
    uv run python -m src.run_all [--skip-download] [--n-permutations 1000]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.extract_dag import build_assumed_dag, get_testable_implications, save_dag, print_dag_summary
from src.generate_synthetic import generate_known_truth, generate_bcd_ntz_synthetic, SyntheticConfig
from src.test_ci import test_all_implications, apply_fdr_correction
from src.discover_dag import discover_dag_pc, compare_dags, dag_comparison_summary
from src.sheaf_4ce import (
    build_site_data_synthetic,
    compute_sheaf_h1,
    sheaf_vs_cochran_comparison,
)
from src.figures import (
    plot_annotated_dag,
    plot_dag_comparison,
    plot_calibration,
    plot_sheaf_vs_cochran,
)


RESULTS_DIR = Path(__file__).parent.parent / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


def run_experiment_1_calibration(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Experiment 0: calibration on known-truth synthetic data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 0: RCIT CALIBRATION ON KNOWN-TRUTH DATA")
    print("=" * 60)
    t0 = time.time()

    df, ground_truth = generate_known_truth(n_samples=2000, seed=seed)
    G = build_assumed_dag()
    implications = get_testable_implications(G)

    gt_implications = []
    for (x, y, cond), is_ci in ground_truth.items():
        gt_implications.append({
            "X": x, "Y": y,
            "conditioning_set": sorted(cond),
            "type": "known_truth",
            "expected_independent": is_ci,
        })

    results = test_all_implications(df, gt_implications, use_rcit=True)

    correct_fisher = 0
    correct_kernel = 0
    total = 0
    for idx, ((x, y, cond), expected) in enumerate(ground_truth.items()):
        if idx >= len(results):
            break
        row = results.iloc[idx]
        fisher_says_ci = row["fisher_verdict"] == "independent"
        kernel_says_ci = row["kernel_verdict"] == "independent"
        if fisher_says_ci == expected:
            correct_fisher += 1
        if kernel_says_ci == expected:
            correct_kernel += 1
        total += 1

    print(f"\n  Calibration results ({total} tests):")
    print(f"    Fisher's z accuracy: {correct_fisher}/{total} = {correct_fisher/max(total,1):.1%}")
    print(f"    Kernel CIT accuracy: {correct_kernel}/{total} = {correct_kernel/max(total,1):.1%}")

    results.to_csv(TABLES_DIR / "T5_calibration.csv", index=False)
    print(f"  Saved T5_calibration.csv")
    print(f"  Time: {time.time()-t0:.1f}s")

    return results, ground_truth


def run_experiment_1_dag_testing(seed: int = 42) -> pd.DataFrame:
    """Experiment 3.1: RCIT CI testing on BCD/NTZ DAG."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.1: DAG ASSUMPTION TESTING WITH RCIT")
    print("=" * 60)
    t0 = time.time()

    G = build_assumed_dag()
    print_dag_summary(G)
    save_dag(G, RESULTS_DIR / "assumed_dag.json")

    implications = get_testable_implications(G)
    print(f"\n  {len(implications)} testable CI implications from d-separation")

    df = generate_bcd_ntz_synthetic(SyntheticConfig(seed=seed))
    print(f"  Synthetic data: {df.shape[0]} samples, {df.shape[1]} variables")

    results = test_all_implications(df, implications, use_rcit=True)
    results = apply_fdr_correction(results)

    n_consistent = (results["fisher_verdict"] == "independent").sum()
    n_violated = (results["fisher_verdict"] == "dependent").sum()
    n_agree = results["methods_agree"].sum()
    print(f"\n  Results:")
    print(f"    Consistent (Fisher): {n_consistent}/{len(results)}")
    print(f"    Violated (Fisher):   {n_violated}/{len(results)}")
    print(f"    Methods agree:       {n_agree}/{len(results)}")

    results.to_csv(TABLES_DIR / "T1_dag_edges.csv", index=False)
    print(f"  Saved T1_dag_edges.csv")

    plot_annotated_dag(G, results)

    print(f"  Time: {time.time()-t0:.1f}s")
    return results


def run_experiment_2_dag_discovery(seed: int = 42) -> pd.DataFrame:
    """Experiment 3.2: data-driven DAG discovery with PC algorithm."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.2: DATA-DRIVEN DAG DISCOVERY (PC ALGORITHM)")
    print("=" * 60)
    t0 = time.time()

    assumed = build_assumed_dag()
    df = generate_bcd_ntz_synthetic(SyntheticConfig(seed=seed))

    print("  Running PC algorithm with Fisher's z...")
    discovered = discover_dag_pc(df, alpha=0.05, ci_test="fisherz")
    comparison = compare_dags(assumed, discovered)
    summary = dag_comparison_summary(comparison)

    print(f"\n  Comparison:")
    print(f"    Assumed edges:  {summary['n_assumed_edges']}")
    print(f"    Discovered edges: {summary['n_discovered_edges']}")
    print(f"    Confirmed:      {summary['confirmed']}")
    print(f"    Missing:        {summary['missing_in_discovered']}")
    print(f"    Extra:          {summary['extra_in_discovered']}")
    print(f"    Precision: {summary['precision']:.3f}")
    print(f"    Recall:    {summary['recall']:.3f}")
    print(f"    F1:        {summary['f1']:.3f}")

    comparison.to_csv(TABLES_DIR / "T2_dag_comparison.csv", index=False)
    print(f"  Saved T2_dag_comparison.csv")

    with open(TABLES_DIR / "T2_dag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_dag_comparison(assumed, discovered, comparison)

    print(f"  Time: {time.time()-t0:.1f}s")
    return comparison


def run_experiment_3_sheaf(
    n_permutations: int = 1000,
    use_real_data: bool = False,
    seed: int = 42,
) -> dict:
    """Experiment 3.3: sheaf consistency on 4CE multi-site data."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3.3: SHEAF CONSISTENCY ON 4CE DATA")
    print("=" * 60)
    t0 = time.time()

    csv_path = Path(__file__).parent.parent / "data" / "4ce_neuro" / "site_neuro_counts.csv"
    if use_real_data and csv_path.exists():
        from src.sheaf_4ce import load_real_4ce_data
        site_data = load_real_4ce_data(csv_path)
        print(f"  Loaded {len(site_data)} sites from REAL 4CE data")
        for name, data in sorted(site_data.items()):
            print(f"    {name}: n={data['n_patients']}, {len(data['measured'])} outcomes")
    else:
        print("  Using synthetic 4CE-like data with heterogeneous stalks")
        site_data = build_site_data_synthetic(n_sites=15, seed=seed)

    sheaf = compute_sheaf_h1(site_data)
    print(f"\n  Sheaf H^1:")
    print(f"    Inconsistency: {sheaf['inconsistency_score']:.6f}")
    print(f"    Sites: {sheaf['n_sites']}, Edges: {sheaf['n_edges']}")
    for site, n_measured in sheaf["site_coverage"].items():
        print(f"    {site}: {n_measured} outcomes")

    print(f"\n  Running sheaf vs Cochran comparison ({n_permutations} permutations)...")
    comparison = sheaf_vs_cochran_comparison(site_data, n_permutations=n_permutations, seed=seed)

    print(f"\n  Results:")
    print(f"    Sheaf inconsistency: {comparison['sheaf']['inconsistency']:.6f}")
    print(f"    Sheaf z-score:       {comparison['sheaf']['z_score']:.2f}")
    print(f"    Sheaf p-value:       {comparison['sheaf']['permutation_p']:.4f}")
    print(f"    Sheaf sites used:    {comparison['advantage']['sheaf_n_sites']}")
    print(f"    Cochran max k:       {comparison['advantage']['cochran_max_sites']}")
    print(f"    Sheaf uses more:     {comparison['advantage']['sheaf_uses_more_sites']}")

    cochran_rows = []
    for outcome, data in comparison["cochran"].items():
        cochran_rows.append({"outcome": outcome, **data})
    cochran_df = pd.DataFrame(cochran_rows)

    sheaf_row = pd.DataFrame([{
        "method": "sheaf_h1",
        "score": comparison["sheaf"]["inconsistency"],
        "z_score": comparison["sheaf"]["z_score"],
        "permutation_p": comparison["sheaf"]["permutation_p"],
        "n_sites": comparison["advantage"]["sheaf_n_sites"],
    }])

    cochran_df.to_csv(TABLES_DIR / "T4_sheaf_vs_cochran.csv", index=False)
    sheaf_row.to_csv(TABLES_DIR / "T4_sheaf_summary.csv", index=False)
    print(f"  Saved T4_sheaf_vs_cochran.csv and T4_sheaf_summary.csv")

    plot_sheaf_vs_cochran(comparison)

    print(f"  Time: {time.time()-t0:.1f}s")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run all Visweswaran experiments")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading 4CE data")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutations for null tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-real-4ce", action="store_true",
                        help="Attempt to use downloaded 4CE .rda files")
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc)
    print(f"Starting experiments at {start.isoformat()}")
    print(f"Seed: {args.seed}, Permutations: {args.n_permutations}")

    if not args.skip_download and args.use_real_4ce:
        print("\n--- Downloading 4CE data ---")
        from src.download_4ce import download_4ce_rda_files
        data_dir = Path(__file__).parent.parent / "data" / "4ce_neuro"
        download_4ce_rda_files(data_dir)

    calib_results, ground_truth = run_experiment_1_calibration(seed=args.seed)
    plot_calibration(ground_truth, calib_results)

    dag_results = run_experiment_1_dag_testing(seed=args.seed)

    comparison = run_experiment_2_dag_discovery(seed=args.seed)

    sheaf = run_experiment_3_sheaf(
        n_permutations=args.n_permutations,
        use_real_data=args.use_real_4ce,
        seed=args.seed,
    )

    end = datetime.now(timezone.utc)
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Duration: {(end - start).total_seconds():.0f}s")
    print(f"Results in: {RESULTS_DIR}")
    print(f"\nTables:")
    for f in sorted(TABLES_DIR.glob("*.csv")):
        print(f"  {f.name}")
    print(f"\nFigures:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
