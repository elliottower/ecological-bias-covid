"""
Calibration tests for CI testing tools on known-truth synthetic data.

Verifies that RCIT / Fisher's z achieve nominal type I error and adequate
power before we trust them on real DAG testing.
"""

import numpy as np
import pandas as pd
import pytest

from src.extract_dag import build_assumed_dag
from src.generate_synthetic import generate_known_truth
from src.test_ci import fishers_z_test
from src.sheaf_4ce import build_site_data_synthetic, compute_sheaf_h1, cochrans_q, NEURO_CATEGORIES


def test_fishers_z_detects_planted_dependence():
    df, ground_truth = generate_known_truth(n_samples=5000, seed=None)
    result = fishers_z_test(df, "age_at_dmt_initiation", "disease_duration", [])
    assert result["p_value"] < 0.01, (
        f"Fisher's z failed to detect planted age-disease_duration dependence "
        f"(p={result['p_value']:.4f})"
    )


def test_fishers_z_respects_true_independence():
    df, ground_truth = generate_known_truth(n_samples=5000, seed=None)
    result = fishers_z_test(df, "sex", "followup_duration", [])
    assert result["p_value"] > 0.01, (
        f"Fisher's z falsely detected dependence between independent variables "
        f"(p={result['p_value']:.4f})"
    )


def test_fishers_z_type_i_error_calibrated():
    n_tests = 200
    false_positives = 0
    for _ in range(n_tests):
        df, _ = generate_known_truth(n_samples=1000, seed=None)
        result = fishers_z_test(df, "sex", "followup_duration", [])
        if result["p_value"] < 0.05:
            false_positives += 1

    type_i_rate = false_positives / n_tests
    assert type_i_rate < 0.12, (
        f"Type I error rate {type_i_rate:.3f} exceeds 0.12 (expected ~0.05)"
    )


def test_fishers_z_power_adequate():
    n_tests = 100
    true_positives = 0
    for _ in range(n_tests):
        df, _ = generate_known_truth(n_samples=1000, seed=None)
        result = fishers_z_test(df, "age_at_dmt_initiation", "disease_duration", [])
        if result["p_value"] < 0.05:
            true_positives += 1

    power = true_positives / n_tests
    assert power > 0.70, (
        f"Power {power:.3f} is too low (expected >0.70 for planted effect)"
    )


def test_dag_has_correct_structure():
    G = build_assumed_dag()
    assert G.number_of_nodes() == 11
    assert G.number_of_edges() == 19
    assert G.has_edge("DMT", "PDDS_change")
    assert G.has_edge("age_at_dmt_initiation", "DMT")
    assert G.has_edge("age_at_dmt_initiation", "PDDS_change")
    assert not G.has_edge("PDDS_change", "DMT")


def test_synthetic_data_matches_published_stats():
    from src.generate_synthetic import generate_bcd_ntz_synthetic, SyntheticConfig
    dfs = [generate_bcd_ntz_synthetic(SyntheticConfig(n_samples=5000, seed=None)) for _ in range(20)]
    ages_bcd = [df.loc[df["DMT"] == 1, "age_at_dmt_initiation"].mean() for df in dfs]
    ages_ntz = [df.loc[df["DMT"] == 0, "age_at_dmt_initiation"].mean() for df in dfs]
    assert pytest.approx(np.mean(ages_bcd), abs=2.0) == 45.5
    assert pytest.approx(np.mean(ages_ntz), abs=2.0) == 40.2


def test_sheaf_laplacian_psd():
    site_data = build_site_data_synthetic(n_sites=10, seed=None)
    sheaf = compute_sheaf_h1(site_data)
    assert sheaf["inconsistency_score"] >= -1e-10, "Sheaf inconsistency should be non-negative"


def test_cochrans_q_on_homogeneous_data():
    site_data = {}
    for i in range(10):
        site_data[f"s{i}"] = {
            "n_patients": 500,
            "measured": ["headache"],
            "outcomes": {
                "headache": {"count": 50, "n_patients": 500, "rate": 0.10},
            },
        }
    result = cochrans_q(site_data, "headache")
    assert result["Q"] == pytest.approx(0.0, abs=1e-10), "Identical rates should give Q=0"
    assert result["k"] == 10


def test_sheaf_detects_inconsistency():
    site_data = {}
    shared_outcomes = ["headache", "seizure", "stroke"]
    for i in range(5):
        rates = {o: {"count": 50, "n_patients": 1000, "rate": 0.05} for o in shared_outcomes}
        site_data[f"consistent_{i}"] = {
            "n_patients": 1000, "measured": shared_outcomes, "outcomes": rates,
        }

    sheaf_consistent = compute_sheaf_h1(site_data)

    rates_divergent = {
        "headache": {"count": 200, "n_patients": 1000, "rate": 0.20},
        "seizure": {"count": 10, "n_patients": 1000, "rate": 0.01},
        "stroke": {"count": 5, "n_patients": 1000, "rate": 0.005},
    }
    site_data["divergent"] = {
        "n_patients": 1000, "measured": shared_outcomes, "outcomes": rates_divergent,
    }

    sheaf_with_outlier = compute_sheaf_h1(site_data)

    assert sheaf_with_outlier["inconsistency_score"] > sheaf_consistent["inconsistency_score"], (
        "Adding a divergent site should increase inconsistency"
    )
