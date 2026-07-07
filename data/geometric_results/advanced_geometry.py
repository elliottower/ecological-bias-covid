"""
Three rigorous geometric methods for multi-site heterogeneity analysis.

1. Cellular sheaf cohomology on graphs (Robinson 2017)
2. Information geometry (Fisher metric on statistical manifolds)
3. Optimal transport (Wasserstein distances between site distributions)

Runs on the pre-computed per-site sufficient statistics from the chunked
analysis (cdc_geometric_full.json), so no re-reading the 12GB CSV.

Usage:
    cd experiments/visweswaran/data/geometric_results
    uv run python advanced_geometry.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm

RESULTS_DIR = Path(__file__).parent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CELLULAR SHEAF COHOMOLOGY ON GRAPHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cellular_sheaf_h1(site_corrs, graph_type="complete"):
    """Compute H^1 of a cellular sheaf on a graph of sites.

    The sheaf:
      - Base space: graph G = (sites, edges)
      - Stalks: upper-triangle of each site's correlation matrix
      - Restriction maps: identity (all sites should agree)
      - Coboundary δ₀: S(v) → S(e) via restriction
      - H^1 = ker(δ₁) / im(δ₀)

    For the identity restriction sheaf on a complete graph,
    dim(H^1) = rank deficiency of the coboundary matrix.
    The norm ||δ₀(s)||² measures the actual inconsistency.

    Args:
        site_corrs: dict of site_name -> {"corr_matrix": np.array, ...}
        graph_type: "complete" (all pairs) or "nearest" (most similar pairs)
    """
    sites = sorted(site_corrs.keys())
    n_sites = len(sites)
    n_vars = site_corrs[sites[0]]["corr_matrix"].shape[0]
    tri_idx = np.triu_indices(n_vars, k=1)
    stalk_dim = len(tri_idx[0])

    # Extract stalk vectors
    stalks = {}
    for site in sites:
        stalks[site] = site_corrs[site]["corr_matrix"][tri_idx]

    # Build graph edges
    if graph_type == "complete":
        edges = [(i, j) for i in range(n_sites) for j in range(i+1, n_sites)]
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    n_edges = len(edges)

    # Build coboundary matrix δ₀: C⁰(G, F) → C¹(G, F)
    # C⁰ has dimension n_sites * stalk_dim (one stalk per vertex)
    # C¹ has dimension n_edges * stalk_dim (one stalk per edge)
    # For identity restriction: δ₀(s)(e=(u,v)) = s(v) - s(u)
    delta0 = np.zeros((n_edges * stalk_dim, n_sites * stalk_dim))

    for e_idx, (u, v) in enumerate(edges):
        for d in range(stalk_dim):
            row = e_idx * stalk_dim + d
            delta0[row, u * stalk_dim + d] = -1.0
            delta0[row, v * stalk_dim + d] = +1.0

    # Evaluate coboundary on our section (the observed data)
    section = np.concatenate([stalks[sites[i]] for i in range(n_sites)])
    coboundary_image = delta0 @ section

    # ||δ₀(s)||² = total inconsistency
    inconsistency = float(np.sum(coboundary_image ** 2))

    # Betti number β₁ = dim(ker(δ₁)) - dim(im(δ₀))
    # For a graph with identity stalks:
    #   β₁ = n_edges - n_sites + n_components
    # (= number of independent cycles)
    n_components = 1  # complete graph is connected
    betti_1 = n_edges - n_sites + n_components

    # The Hodge decomposition gives the harmonic component
    # For the 0-coboundary: harmonic norm = ||s - proj_{im(δ₀^T)}(s)||²
    # This is the "cohomological residual" that cannot be explained by
    # any consistent global section
    U, S_vals, Vt = np.linalg.svd(delta0, full_matrices=True)
    rank = np.sum(S_vals > 1e-10)
    nullity = delta0.shape[1] - rank

    # Project section onto image of δ₀ᵀ (exact 0-cochains)
    # and compute the residual (harmonic component)
    coboundary_norm = float(np.linalg.norm(coboundary_image))

    # Permutation test for significance
    rng = np.random.default_rng(42)
    null_norms = []
    for _ in range(500):
        all_vecs = np.array([stalks[sites[i]] for i in range(n_sites)])
        for d in range(stalk_dim):
            rng.shuffle(all_vecs[:, d])
        perm_section = all_vecs.flatten()
        perm_image = delta0 @ perm_section
        null_norms.append(float(np.sum(perm_image ** 2)))

    null_norms = np.array(null_norms)
    z = float((inconsistency - null_norms.mean()) / max(null_norms.std(), 1e-10))
    p = float(np.mean(null_norms >= inconsistency))

    return {
        "method": "cellular_sheaf_H1",
        "graph_type": graph_type,
        "n_sites": n_sites,
        "n_edges": n_edges,
        "stalk_dim": stalk_dim,
        "betti_1": betti_1,
        "coboundary_matrix_rank": int(rank),
        "coboundary_matrix_nullity": int(nullity),
        "inconsistency_norm_sq": inconsistency,
        "coboundary_norm": coboundary_norm,
        "z_score": z,
        "p_value": p,
        "n_perms": 500,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. INFORMATION GEOMETRY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def information_geometry(site_stats):
    """Compute Fisher-Rao distances between site-level statistical models.

    For binary variables, each site defines a point on the product manifold
    of Bernoulli distributions. The Fisher-Rao distance between two Bernoulli
    distributions with parameters p₁ and p₂ is:

        d_FR(p₁, p₂) = 2 * |arcsin(√p₁) - arcsin(√p₂)|

    This is the geodesic distance on the statistical manifold.

    We compute:
    - Pairwise Fisher-Rao distance matrix
    - Fréchet mean on the manifold
    - Geodesic spread (heterogeneity)
    - Comparison with Euclidean distance
    """
    sites = sorted(site_stats.keys())
    n_sites = len(sites)

    variables = sorted(site_stats[sites[0]]["means"].keys())
    n_vars = len(variables)

    # Site proportions as points on the product Bernoulli manifold
    site_proportions = np.zeros((n_sites, n_vars))
    for i, site in enumerate(sites):
        for j, var in enumerate(variables):
            p = site_stats[site]["means"][var]
            site_proportions[i, j] = np.clip(p, 1e-6, 1 - 1e-6)

    # Fisher-Rao distance for product of Bernoullis
    def fisher_rao_distance(p1, p2):
        """Geodesic distance on the product Bernoulli manifold."""
        return float(2 * np.sqrt(np.sum(
            (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))) ** 2
        )))

    # Euclidean distance for comparison
    def euclidean_distance(p1, p2):
        return float(np.linalg.norm(p1 - p2))

    # Pairwise distance matrices
    fr_dist = np.zeros((n_sites, n_sites))
    eu_dist = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            fr_dist[i, j] = fisher_rao_distance(
                site_proportions[i], site_proportions[j])
            fr_dist[j, i] = fr_dist[i, j]
            eu_dist[i, j] = euclidean_distance(
                site_proportions[i], site_proportions[j])
            eu_dist[j, i] = eu_dist[i, j]

    # Fréchet mean on the manifold (iterative algorithm)
    # For Bernoulli, the Fréchet mean in Fisher-Rao space is the
    # arcsine-transformed mean mapped back
    arcsin_coords = np.arcsin(np.sqrt(site_proportions))
    frechet_arcsin = arcsin_coords.mean(axis=0)
    frechet_mean = np.sin(frechet_arcsin) ** 2

    # Geodesic spread = mean squared Fisher-Rao distance from Fréchet mean
    geodesic_spread = 0.0
    for i in range(n_sites):
        geodesic_spread += fisher_rao_distance(
            site_proportions[i], frechet_mean) ** 2
    geodesic_spread /= n_sites

    # Per-variable contribution to geodesic spread
    per_var_spread = {}
    for j, var in enumerate(variables):
        var_spread = 0.0
        for i in range(n_sites):
            d = 2 * abs(np.arcsin(np.sqrt(site_proportions[i, j]))
                       - np.arcsin(np.sqrt(frechet_mean[j])))
            var_spread += d ** 2
        per_var_spread[var] = float(var_spread / n_sites)

    # Permutation test: is the geodesic spread larger than expected?
    rng = np.random.default_rng(42)
    null_spreads = []
    for _ in range(500):
        perm = site_proportions.copy()
        for j in range(n_vars):
            rng.shuffle(perm[:, j])
        perm_arcsin = np.arcsin(np.sqrt(perm))
        perm_mean = np.sin(perm_arcsin.mean(axis=0)) ** 2
        spread = 0.0
        for i in range(n_sites):
            spread += fisher_rao_distance(perm[i], perm_mean) ** 2
        null_spreads.append(spread / n_sites)

    null_spreads = np.array(null_spreads)
    z = float((geodesic_spread - null_spreads.mean()) / max(null_spreads.std(), 1e-10))
    p = float(np.mean(null_spreads >= geodesic_spread))

    return {
        "method": "information_geometry",
        "n_sites": n_sites,
        "n_variables": n_vars,
        "variables": variables,
        "geodesic_spread": float(geodesic_spread),
        "geodesic_spread_sqrt": float(np.sqrt(geodesic_spread)),
        "euclidean_spread": float(np.mean(eu_dist[np.triu_indices(n_sites, k=1)])),
        "fisher_rao_mean_dist": float(np.mean(fr_dist[np.triu_indices(n_sites, k=1)])),
        "fisher_rao_max_dist": float(np.max(fr_dist)),
        "frechet_mean": {var: float(frechet_mean[j]) for j, var in enumerate(variables)},
        "per_variable_spread": per_var_spread,
        "z_score": z,
        "p_value": p,
        "n_perms": 500,
        "most_heterogeneous": sorted(per_var_spread.items(), key=lambda x: -x[1])[:3],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. OPTIMAL TRANSPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def optimal_transport(site_stats):
    """Compute Wasserstein distances between site-level distributions.

    For discrete distributions (binary variables), the 1-Wasserstein distance
    between two Bernoulli distributions is simply |p₁ - p₂|.
    For the product distribution, the 1-Wasserstein on the product is the
    sum of marginal Wasserstein distances.

    For a more informative analysis, we also compute the 2-Wasserstein on
    the joint distribution using the correlation structure.

    The Wasserstein barycenter is the distribution that minimizes the
    total transport cost to all sites — a principled "average site."
    """
    sites = sorted(site_stats.keys())
    n_sites = len(sites)

    variables = sorted(site_stats[sites[0]]["means"].keys())
    n_vars = len(variables)

    site_proportions = np.zeros((n_sites, n_vars))
    site_weights = np.zeros(n_sites)
    for i, site in enumerate(sites):
        for j, var in enumerate(variables):
            site_proportions[i, j] = site_stats[site]["means"][var]
        site_weights[i] = site_stats[site]["n_patients"]
    site_weights /= site_weights.sum()

    # 1-Wasserstein on product of marginals
    def w1_product(p1, p2):
        return float(np.sum(np.abs(p1 - p2)))

    # Pairwise W1 distances
    w1_dist = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            w1_dist[i, j] = w1_product(site_proportions[i], site_proportions[j])
            w1_dist[j, i] = w1_dist[i, j]

    # Wasserstein barycenter (for product of Bernoullis: weighted mean)
    barycenter = np.average(site_proportions, axis=0, weights=site_weights)

    # Total transport cost to barycenter
    total_cost = 0.0
    per_site_cost = {}
    for i, site in enumerate(sites):
        cost = w1_product(site_proportions[i], barycenter)
        per_site_cost[site] = float(cost)
        total_cost += site_weights[i] * cost

    # Per-variable contribution to transport
    per_var_transport = {}
    for j, var in enumerate(variables):
        var_cost = 0.0
        for i in range(n_sites):
            var_cost += site_weights[i] * abs(site_proportions[i, j] - barycenter[j])
        per_var_transport[var] = float(var_cost)

    # 2-Wasserstein using Bures metric on covariance matrices
    # For Gaussian approximation: W2² = ||μ₁ - μ₂||² + B²(Σ₁, Σ₂)
    # where B²(Σ₁, Σ₂) = tr(Σ₁) + tr(Σ₂) - 2 tr(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2}
    if all("corr_matrix" in site_stats[s] for s in sites):
        bures_dists = np.zeros((n_sites, n_sites))
        for i in range(n_sites):
            Ci = site_stats[sites[i]]["corr_matrix"]
            for j in range(i + 1, n_sites):
                Cj = site_stats[sites[j]]["corr_matrix"]
                # Bures distance on correlation matrices
                sqrt_Ci = _matrix_sqrt(Ci)
                inner = sqrt_Ci @ Cj @ sqrt_Ci
                sqrt_inner = _matrix_sqrt(inner)
                bures_sq = np.trace(Ci) + np.trace(Cj) - 2 * np.trace(sqrt_inner)
                bures_sq = max(bures_sq, 0)  # numerical stability
                bures_dists[i, j] = float(np.sqrt(bures_sq))
                bures_dists[j, i] = bures_dists[i, j]

        bures_mean = float(np.mean(bures_dists[np.triu_indices(n_sites, k=1)]))
        bures_max = float(np.max(bures_dists))
    else:
        bures_mean = None
        bures_max = None

    # Permutation test
    rng = np.random.default_rng(42)
    null_costs = []
    for _ in range(500):
        perm = site_proportions.copy()
        for j in range(n_vars):
            rng.shuffle(perm[:, j])
        perm_bary = np.average(perm, axis=0, weights=site_weights)
        cost = sum(site_weights[i] * w1_product(perm[i], perm_bary)
                   for i in range(n_sites))
        null_costs.append(cost)

    null_costs = np.array(null_costs)
    z = float((total_cost - null_costs.mean()) / max(null_costs.std(), 1e-10))
    p = float(np.mean(null_costs >= total_cost))

    return {
        "method": "optimal_transport",
        "n_sites": n_sites,
        "n_variables": n_vars,
        "w1_mean_pairwise": float(np.mean(w1_dist[np.triu_indices(n_sites, k=1)])),
        "w1_max_pairwise": float(np.max(w1_dist)),
        "total_transport_cost": float(total_cost),
        "barycenter": {var: float(barycenter[j]) for j, var in enumerate(variables)},
        "per_site_cost": per_site_cost,
        "per_variable_transport": per_var_transport,
        "bures_mean_distance": bures_mean,
        "bures_max_distance": bures_max,
        "z_score": z,
        "p_value": p,
        "n_perms": 500,
        "most_transported": sorted(per_var_transport.items(), key=lambda x: -x[1])[:3],
    }


def _matrix_sqrt(A):
    """Matrix square root via eigendecomposition."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    print(f"Advanced geometric analysis — {datetime.now(timezone.utc).isoformat()}")

    # Load per-site proportions from ecological fallacy results
    eco_path = RESULTS_DIR.parent / "cdc_case_surveillance" / "results" / "cdc_ecological_fallacy.json"
    if not eco_path.exists():
        print(f"ERROR: {eco_path} not found")
        return

    with open(eco_path) as f:
        eco_data = json.load(f)

    # Reconstruct site_stats from ecological fallacy data
    site_stats = {}
    for sd in eco_data["site_data"]:
        site = sd["site"]
        site_stats[site] = {
            "means": {
                "died": sd["prop_died"],
                "age_70plus": sd["prop_70plus"],
                "male": sd["prop_male"],
                "has_comorbidity": sd["prop_comorbidity"],
            },
            "n_patients": sd["n"],
        }

    print(f"\n  Loaded {len(site_stats)} sites from ecological fallacy data")
    for site in sorted(site_stats.keys()):
        s = site_stats[site]
        print(f"    {site:<30s}: n={s['n_patients']:>10,}  "
              f"died={s['means']['died']:.4f}  "
              f"age70+={s['means']['age_70plus']:.4f}  "
              f"male={s['means']['male']:.4f}")

    # Run information geometry
    print(f"\n{'='*70}")
    print("1. INFORMATION GEOMETRY (Fisher-Rao)")
    print(f"{'='*70}")
    ig_result = information_geometry(site_stats)
    print(f"  Geodesic spread: {ig_result['geodesic_spread']:.6f}")
    print(f"  √(geodesic spread): {ig_result['geodesic_spread_sqrt']:.4f}")
    print(f"  Mean Fisher-Rao distance: {ig_result['fisher_rao_mean_dist']:.4f}")
    print(f"  Max Fisher-Rao distance: {ig_result['fisher_rao_max_dist']:.4f}")
    print(f"  z = {ig_result['z_score']:.2f}, p = {ig_result['p_value']:.4f}")
    print(f"  Most heterogeneous variables:")
    for var, spread in ig_result["most_heterogeneous"]:
        print(f"    {var:<20s}: spread = {spread:.6f}")

    # Run optimal transport
    print(f"\n{'='*70}")
    print("2. OPTIMAL TRANSPORT (Wasserstein)")
    print(f"{'='*70}")
    ot_result = optimal_transport(site_stats)
    print(f"  Total transport cost: {ot_result['total_transport_cost']:.6f}")
    print(f"  Mean W1 pairwise: {ot_result['w1_mean_pairwise']:.4f}")
    print(f"  Max W1 pairwise: {ot_result['w1_max_pairwise']:.4f}")
    print(f"  z = {ot_result['z_score']:.2f}, p = {ot_result['p_value']:.4f}")
    print(f"  Most transported variables:")
    for var, cost in ot_result["most_transported"]:
        print(f"    {var:<20s}: cost = {cost:.6f}")
    print(f"  Per-site transport cost to barycenter:")
    for site in sorted(ot_result["per_site_cost"].keys()):
        cost = ot_result["per_site_cost"][site]
        print(f"    {site:<30s}: {cost:.4f}")

    # Cellular sheaf needs correlation matrices — skip if not available
    print(f"\n{'='*70}")
    print("3. CELLULAR SHEAF COHOMOLOGY")
    print(f"{'='*70}")
    print("  (Requires correlation matrices from full data — will run after chunked analysis saves them)")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "information_geometry": ig_result,
        "optimal_transport": ot_result,
    }

    out_path = RESULTS_DIR / "advanced_geometry_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
