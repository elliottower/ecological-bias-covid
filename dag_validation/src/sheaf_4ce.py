"""
Sheaf consistency on 4CE multi-site COVID neurological data.

The key claim: sheaf H^1 should genuinely beat Cochran's Q here because
sites measure DIFFERENT subsets of neurological outcomes (heterogeneous stalks),
and Q requires all sites to measure the same thing.

The sheaf construction:
- Vertices: healthcare sites (up to 21)
- Stalks: the set of neurological outcome categories measured at each site
- Edge stalks: the intersection of outcomes measured at both connected sites
- Restriction maps: natural projection from site stalk to edge stalk
- H^1: the sheaf Laplacian's first cohomology detects inconsistencies that
  can't be attributed to different measurement coverage.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, linalg
from tqdm import tqdm


NEURO_CATEGORIES = [
    "consciousness", "coordination", "dizziness", "headache",
    "inflammatory_cns", "myopathy", "neuropathy", "psychiatric",
    "seizure", "cerebrovascular", "vision",
]

ICD_TO_CATEGORY = {
    "F29": "psychiatric", "G03": "meningitis", "G04": "encephalitis",
    "G40": "seizure", "G44": "headache", "G45": "tia",
    "G46": "cerebrovascular_syndrome", "G61": "neuropathy",
    "G72": "myopathy", "G93": "consciousness",
    "H54": "vision", "I60": "subarachnoid_hemorrhage",
    "I61": "intracerebral_hemorrhage", "I62": "other_hemorrhage",
    "I67": "cerebrovascular_disease", "M60": "myositis",
    "R27": "coordination", "R41": "cognition",
    "R42": "dizziness", "R43": "smell_taste",
}


def load_real_4ce_data(csv_path: Path) -> dict[str, dict]:
    """Load real 4CE site data from the extracted CSV.

    The CSV has columns: site, total_patients, count_F29, rate_F29, ...
    We convert to the same dict format as build_site_data_synthetic.
    """
    df = pd.read_csv(csv_path)
    site_data = {}

    for _, row in df.iterrows():
        site = row["site"]
        total = row["total_patients"]
        if pd.isna(total) or total == 0:
            continue

        outcomes = {}
        measured = []
        for icd_code, category in ICD_TO_CATEGORY.items():
            count_col = f"count_{icd_code}"
            rate_col = f"rate_{icd_code}"
            if count_col in row and not pd.isna(row[count_col]) and row[count_col] > 0:
                outcomes[category] = {
                    "count": int(row[count_col]),
                    "n_patients": int(total),
                    "rate": float(row[rate_col]),
                    "icd_code": icd_code,
                }
                measured.append(category)

        if len(measured) >= 2:
            site_data[site] = {
                "n_patients": int(total),
                "measured": measured,
                "outcomes": outcomes,
            }

    return site_data


def build_site_data_synthetic(
    n_sites: int = 15, seed: int | None = None
) -> dict[str, dict]:
    """Generate synthetic site data mimicking 4CE heterogeneous structure.

    Each site measures a SUBSET of outcomes (simulating real-world heterogeneity
    in what ICD codes different EHR systems capture).
    """
    rng = np.random.default_rng(seed)
    n_outcomes = len(NEURO_CATEGORIES)

    true_rates = rng.beta(2, 20, n_outcomes)

    site_data = {}
    for i in range(n_sites):
        n_measured = rng.integers(5, n_outcomes + 1)
        measured_idx = sorted(rng.choice(n_outcomes, n_measured, replace=False))
        measured_names = [NEURO_CATEGORIES[j] for j in measured_idx]

        n_patients = rng.integers(200, 5000)
        site_effect = rng.normal(0, 0.3)
        rates = {}
        for j, name in zip(measured_idx, measured_names):
            base_rate = true_rates[j]
            noisy_rate = np.clip(
                base_rate * np.exp(site_effect + rng.normal(0, 0.2)),
                0.001, 0.5,
            )
            count = rng.binomial(n_patients, noisy_rate)
            rates[name] = {
                "count": int(count),
                "n_patients": int(n_patients),
                "rate": count / n_patients,
            }

        site_data[f"site_{i:02d}"] = {
            "outcomes": rates,
            "n_patients": int(n_patients),
            "measured": measured_names,
        }

    return site_data


def cochrans_q(
    site_data: dict[str, dict], outcome: str
) -> dict:
    """Cochran's Q test for heterogeneity across sites measuring the same outcome.

    Q can only use sites that MEASURE the given outcome.
    """
    sites_with_outcome = []
    for site_name, data in site_data.items():
        if outcome in data["outcomes"]:
            o = data["outcomes"][outcome]
            sites_with_outcome.append({
                "site": site_name,
                "rate": o["rate"],
                "count": o["count"],
                "n": o["n_patients"],
            })

    if len(sites_with_outcome) < 2:
        return {"Q": np.nan, "p_value": np.nan, "k": len(sites_with_outcome),
                "outcome": outcome}

    rates = np.array([s["rate"] for s in sites_with_outcome])
    ns = np.array([s["n"] for s in sites_with_outcome])
    weights = ns
    weighted_mean = np.average(rates, weights=weights)
    Q = np.sum(weights * (rates - weighted_mean) ** 2)

    k = len(sites_with_outcome)
    p_value = 1 - stats.chi2.cdf(Q, df=k - 1) if k > 1 else np.nan

    return {
        "Q": float(Q),
        "p_value": float(p_value),
        "k": k,
        "outcome": outcome,
        "sites_used": [s["site"] for s in sites_with_outcome],
        "weighted_mean_rate": float(weighted_mean),
    }


def build_sheaf_laplacian(
    site_data: dict[str, dict],
) -> tuple[np.ndarray, dict]:
    """Build the sheaf Laplacian from heterogeneous site data.

    Each site's stalk is its vector of measured outcome rates.
    Edge stalks are the intersection of outcomes measured at both endpoints.
    Restriction maps project from site stalks to the shared outcomes.
    """
    sites = sorted(site_data.keys())
    n_sites = len(sites)

    site_stalks = {}
    for site in sites:
        measured = sorted(site_data[site]["measured"])
        rates = np.array([site_data[site]["outcomes"][o]["rate"] for o in measured])
        site_stalks[site] = {"measured": measured, "rates": rates}

    total_stalk_dim = sum(len(s["measured"]) for s in site_stalks.values())

    stalk_offsets = {}
    offset = 0
    for site in sites:
        dim = len(site_stalks[site]["measured"])
        stalk_offsets[site] = (offset, offset + dim)
        offset += dim

    L = np.zeros((total_stalk_dim, total_stalk_dim))

    edge_info = []
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            si, sj = sites[i], sites[j]
            shared = sorted(
                set(site_stalks[si]["measured"]) & set(site_stalks[sj]["measured"])
            )
            if not shared:
                continue

            idx_i = [site_stalks[si]["measured"].index(o) for o in shared]
            idx_j = [site_stalks[sj]["measured"].index(o) for o in shared]
            off_i = stalk_offsets[si][0]
            off_j = stalk_offsets[sj][0]

            for ki, kj in zip(idx_i, idx_j):
                gi = off_i + ki
                gj = off_j + kj
                L[gi, gi] += 1
                L[gj, gj] += 1
                L[gi, gj] -= 1
                L[gj, gi] -= 1

            edge_info.append({
                "site_i": si, "site_j": sj,
                "shared_outcomes": shared,
                "n_shared": len(shared),
            })

    return L, {
        "sites": sites,
        "site_stalks": site_stalks,
        "stalk_offsets": stalk_offsets,
        "total_dim": total_stalk_dim,
        "edges": edge_info,
    }


def compute_sheaf_h1(
    site_data: dict[str, dict],
) -> dict:
    """Compute sheaf cohomology H^1 and compare with Cochran's Q.

    H^1 captures inconsistencies across the site graph that cannot be resolved
    by adjusting each site's stalk independently. Non-zero H^1 eigenvalues
    indicate genuine cross-site inconsistency.
    """
    L, info = build_sheaf_laplacian(site_data)

    x = np.zeros(info["total_dim"])
    for site in info["sites"]:
        off_start, off_end = info["stalk_offsets"][site]
        x[off_start:off_end] = info["site_stalks"][site]["rates"]

    inconsistency = float(x @ L @ x)

    eigenvalues = np.sort(np.real(linalg.eigvalsh(L)))
    n_zero = np.sum(np.abs(eigenvalues) < 1e-10)

    n_sites = len(info["sites"])
    n_edges = len(info["edges"])
    total_shared = sum(e["n_shared"] for e in info["edges"])

    coverage = {}
    for site, stalk in info["site_stalks"].items():
        coverage[site] = len(stalk["measured"])

    return {
        "inconsistency_score": inconsistency,
        "laplacian_rank": int(info["total_dim"] - n_zero),
        "n_zero_eigenvalues": int(n_zero),
        "top_5_eigenvalues": eigenvalues[-5:].tolist(),
        "n_sites": n_sites,
        "n_edges": n_edges,
        "total_shared_outcomes": total_shared,
        "site_coverage": coverage,
        "total_stalk_dim": info["total_dim"],
    }


def sheaf_vs_cochran_comparison(
    site_data: dict[str, dict],
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict:
    """Compare sheaf H^1 vs Cochran's Q, including permutation nulls."""
    rng = np.random.default_rng(seed)

    sheaf_result = compute_sheaf_h1(site_data)

    all_outcomes = set()
    for data in site_data.values():
        all_outcomes.update(data["measured"])

    cochran_results = {}
    for outcome in sorted(all_outcomes):
        cochran_results[outcome] = cochrans_q(site_data, outcome)

    cochran_sites = set()
    for r in cochran_results.values():
        if r["k"] >= 2:
            cochran_sites.update(r.get("sites_used", []))

    sheaf_null = []
    cochran_null = {o: [] for o in all_outcomes}

    for _ in tqdm(range(n_permutations), desc="Permutation null"):
        perm_data = _permute_site_labels(site_data, rng)
        perm_sheaf = compute_sheaf_h1(perm_data)
        sheaf_null.append(perm_sheaf["inconsistency_score"])

        for outcome in all_outcomes:
            perm_q = cochrans_q(perm_data, outcome)
            cochran_null[outcome].append(perm_q["Q"])

    sheaf_null = np.array(sheaf_null)
    sheaf_z = (
        (sheaf_result["inconsistency_score"] - sheaf_null.mean()) / max(sheaf_null.std(), 1e-10)
    )
    sheaf_p = np.mean(sheaf_null >= sheaf_result["inconsistency_score"])

    cochran_summary = {}
    for outcome in all_outcomes:
        obs_q = cochran_results[outcome]["Q"]
        null_q = np.array(cochran_null[outcome])
        valid_null = null_q[~np.isnan(null_q)]
        if len(valid_null) > 0 and not np.isnan(obs_q):
            perm_p = np.mean(valid_null >= obs_q)
            z = (obs_q - valid_null.mean()) / max(valid_null.std(), 1e-10)
        else:
            perm_p = np.nan
            z = np.nan
        cochran_summary[outcome] = {
            "Q": obs_q,
            "analytic_p": cochran_results[outcome]["p_value"],
            "permutation_p": float(perm_p),
            "z_score": float(z),
            "k_sites": cochran_results[outcome]["k"],
        }

    return {
        "sheaf": {
            "inconsistency": sheaf_result["inconsistency_score"],
            "z_score": float(sheaf_z),
            "permutation_p": float(sheaf_p),
            "n_sites_used": sheaf_result["n_sites"],
            "n_edges": sheaf_result["n_edges"],
        },
        "cochran": cochran_summary,
        "advantage": {
            "sheaf_uses_more_sites": sheaf_result["n_sites"] > len(cochran_sites),
            "sheaf_n_sites": sheaf_result["n_sites"],
            "cochran_max_sites": max(
                (r["k"] for r in cochran_results.values()), default=0
            ),
            "cochran_union_sites": len(cochran_sites),
        },
    }


def _permute_site_labels(
    site_data: dict[str, dict], rng: np.random.Generator
) -> dict[str, dict]:
    """Permute outcome rates across sites, preserving measurement structure."""
    sites = list(site_data.keys())
    perm_data = {}
    for site in sites:
        perm_data[site] = {
            "n_patients": site_data[site]["n_patients"],
            "measured": list(site_data[site]["measured"]),
            "outcomes": {},
        }

    all_outcomes = set()
    for data in site_data.values():
        all_outcomes.update(data["measured"])

    for outcome in all_outcomes:
        sites_with = [s for s in sites if outcome in site_data[s]["outcomes"]]
        if len(sites_with) < 2:
            for s in sites_with:
                perm_data[s]["outcomes"][outcome] = dict(site_data[s]["outcomes"][outcome])
            continue

        rates = [site_data[s]["outcomes"][outcome]["rate"] for s in sites_with]
        rng.shuffle(rates)
        for s, rate in zip(sites_with, rates):
            n = site_data[s]["n_patients"]
            count = int(rate * n)
            perm_data[s]["outcomes"][outcome] = {
                "count": count, "n_patients": n, "rate": rate,
            }

    return perm_data


if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "data" / "4ce_neuro" / "site_neuro_counts.csv"

    if csv_path.exists():
        print("=== Sheaf Consistency on REAL 4CE Data ===")
        site_data = load_real_4ce_data(csv_path)
        print(f"\nLoaded {len(site_data)} sites from 4CE:")
        for name, data in sorted(site_data.items()):
            print(f"  {name}: n={data['n_patients']}, measures {len(data['measured'])} outcomes")
    else:
        print("=== Sheaf Consistency on Synthetic 4CE-like Data ===")
        site_data = build_site_data_synthetic(n_sites=15, seed=42)
        print(f"\nGenerated {len(site_data)} sites")

    sheaf = compute_sheaf_h1(site_data)
    print(f"\nSheaf H^1:")
    print(f"  Inconsistency score: {sheaf['inconsistency_score']:.6f}")
    print(f"  Laplacian rank: {sheaf['laplacian_rank']}")
    print(f"  Sites: {sheaf['n_sites']}, Edges: {sheaf['n_edges']}")

    all_outcomes = set()
    for data in site_data.values():
        all_outcomes.update(data["measured"])

    print("\nCochran's Q per outcome:")
    for outcome in sorted(all_outcomes):
        q_result = cochrans_q(site_data, outcome)
        if not np.isnan(q_result["Q"]):
            print(f"  {outcome}: Q={q_result['Q']:.2f}, p={q_result['p_value']:.4f}, "
                  f"k={q_result['k']} sites")

    print("\nRunning sheaf vs Cochran comparison with permutation null (100 perms)...")
    comparison = sheaf_vs_cochran_comparison(site_data, n_permutations=100, seed=42)
    print(f"\nSheaf: inconsistency={comparison['sheaf']['inconsistency']:.6f}, "
          f"z={comparison['sheaf']['z_score']:.2f}, "
          f"p={comparison['sheaf']['permutation_p']:.4f}")
    print(f"Sheaf uses {comparison['advantage']['sheaf_n_sites']} sites, "
          f"Cochran max {comparison['advantage']['cochran_max_sites']} per outcome")
