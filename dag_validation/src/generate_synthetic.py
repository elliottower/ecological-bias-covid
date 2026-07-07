"""
Generate synthetic data calibrated to BCD/NTZ published summary statistics.

Two modes:
1. known_truth: Generate from a KNOWN DAG with ground-truth CI structure for
   calibration testing. We know exactly which CIs hold and which don't.
2. bcd_ntz: Generate data matching the published summary statistics (Table 1
   of medRxiv 2025) with the assumed DAG structure.

Summary statistics from the preprint (n=1,175; BCD=614, NTZ=561):
  - Age at DMT: BCD 45.5±10.8, NTZ 40.2±10.7
  - Disease duration: BCD 11.7±8.7, NTZ 8.5±7.4
  - Female: BCD 74.3%, NTZ 72.2%
  - White/Non-Hispanic: BCD 82.6%, NTZ 78.4%
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.extract_dag import CONFOUNDERS_APRIORI, TREATMENT, OUTCOME


@dataclass
class SyntheticConfig:
    n_samples: int = 1175
    treatment_prevalence: float = 0.523  # 614/1175
    seed: int | None = None


def generate_known_truth(
    n_samples: int = 2000, seed: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Generate data from a KNOWN DAG with planted CI violations.

    Ground truth DAG adds edges between some confounders that the assumed
    BCD/NTZ DAG does NOT include (to test whether RCIT can detect them).

    Returns (data, ground_truth) where ground_truth maps each CI to True/False.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(43, 11, n_samples)
    disease_dur = 0.3 * age + rng.normal(0, 5, n_samples)
    sex = rng.binomial(1, 0.73, n_samples).astype(float)
    race = rng.binomial(1, 0.80, n_samples).astype(float)
    prior_dmt = 0.2 * disease_dur + rng.normal(0, 3, n_samples)
    followup = rng.normal(730, 200, n_samples)
    n_phecodes = 0.1 * disease_dur + 0.05 * age + rng.poisson(5, n_samples).astype(float)
    n_cuis = 0.8 * n_phecodes + rng.normal(0, 2, n_samples)
    utilization = 0.3 * n_phecodes + 0.1 * disease_dur + rng.normal(10, 3, n_samples)

    confounders = np.column_stack([
        race, sex, age, disease_dur, prior_dmt,
        followup, n_phecodes, n_cuis, utilization,
    ])

    logit_treatment = (
        -0.5
        + 0.3 * (age - 43) / 11
        + 0.2 * (disease_dur - 10) / 8
        + 0.1 * sex
        + 0.05 * race
        - 0.1 * (prior_dmt - 2) / 3
    )
    treatment = rng.binomial(1, _sigmoid(logit_treatment)).astype(float)

    outcome = (
        -0.5 * treatment
        + 0.2 * (age - 43) / 11
        + 0.15 * (disease_dur - 10) / 8
        + 0.1 * (n_phecodes - 5) / 3
        + rng.normal(0, 1, n_samples)
    )

    df = pd.DataFrame(confounders, columns=CONFOUNDERS_APRIORI)
    df[TREATMENT] = treatment
    df[OUTCOME] = outcome

    ground_truth = {
        ("age_at_dmt_initiation", "disease_duration", frozenset()): False,
        ("n_ms_phecodes", "n_ms_cuis", frozenset()): False,
        ("n_ms_phecodes", "utilization_total", frozenset()): False,
        ("age_at_dmt_initiation", "disease_duration",
         frozenset(CONFOUNDERS_APRIORI) - {"age_at_dmt_initiation", "disease_duration"}): False,
        ("sex", "followup_duration", frozenset()): True,
        ("white_nonhispanic", "followup_duration", frozenset()): True,
    }

    return df, ground_truth


def generate_bcd_ntz_synthetic(
    config: SyntheticConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic data matching BCD/NTZ published summary statistics.

    Calibrated to Table 1 of the medRxiv preprint. The correlation structure
    between confounders is plausible but not exactly the real data.
    """
    if config is None:
        config = SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    treatment = rng.binomial(1, config.treatment_prevalence, n).astype(float)
    is_bcd = treatment == 1

    age = np.where(
        is_bcd,
        rng.normal(45.5, 10.8, n),
        rng.normal(40.2, 10.7, n),
    )

    disease_dur = np.where(
        is_bcd,
        rng.normal(11.7, 8.7, n),
        rng.normal(8.5, 7.4, n),
    )
    disease_dur = np.maximum(disease_dur, 0.5)

    sex = np.where(
        is_bcd,
        rng.binomial(1, 0.743, n),
        rng.binomial(1, 0.722, n),
    ).astype(float)

    race = np.where(
        is_bcd,
        rng.binomial(1, 0.826, n),
        rng.binomial(1, 0.784, n),
    ).astype(float)

    prior_dmt = 0.2 * disease_dur + rng.normal(0, 3, n)
    followup = rng.normal(730, 200, n)
    n_phecodes = np.maximum(0, 0.1 * disease_dur + rng.poisson(5, n).astype(float))
    n_cuis = np.maximum(0, 0.6 * n_phecodes + rng.normal(0, 2, n))
    utilization = np.maximum(0, 0.2 * n_phecodes + 0.1 * disease_dur + rng.normal(10, 3, n))

    outcome = (
        -0.4 * treatment
        + 0.15 * (age - 43) / 11
        + 0.1 * (disease_dur - 10) / 8
        + rng.normal(0, 1, n)
    )

    df = pd.DataFrame({
        "white_nonhispanic": race,
        "sex": sex,
        "age_at_dmt_initiation": age,
        "disease_duration": disease_dur,
        "days_earliest_to_study_dmt": prior_dmt,
        "followup_duration": followup,
        "n_ms_phecodes": n_phecodes,
        "n_ms_cuis": n_cuis,
        "utilization_total": utilization,
        TREATMENT: treatment,
        OUTCOME: outcome,
    })

    return df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


if __name__ == "__main__":
    print("=== Known-truth synthetic data ===")
    df_known, gt = generate_known_truth(n_samples=2000, seed=42)
    print(f"  Shape: {df_known.shape}")
    print(f"  Treatment prevalence: {df_known[TREATMENT].mean():.3f}")
    print(f"  Ground truth CI tests: {len(gt)}")
    for (x, y, cond), is_ci in gt.items():
        cond_str = ", ".join(sorted(cond)) if cond else "{}"
        print(f"    {x} _||_ {y} | {{{cond_str}}}: {'INDEPENDENT' if is_ci else 'DEPENDENT'}")

    print("\n=== BCD/NTZ-calibrated synthetic data ===")
    df_bcd = generate_bcd_ntz_synthetic(SyntheticConfig(seed=42))
    print(f"  Shape: {df_bcd.shape}")
    print(f"  Treatment prevalence: {df_bcd[TREATMENT].mean():.3f}")
    print(f"  Age (BCD): {df_bcd.loc[df_bcd[TREATMENT]==1, 'age_at_dmt_initiation'].mean():.1f} "
          f"± {df_bcd.loc[df_bcd[TREATMENT]==1, 'age_at_dmt_initiation'].std():.1f}")
    print(f"  Age (NTZ): {df_bcd.loc[df_bcd[TREATMENT]==0, 'age_at_dmt_initiation'].mean():.1f} "
          f"± {df_bcd.loc[df_bcd[TREATMENT]==0, 'age_at_dmt_initiation'].std():.1f}")
