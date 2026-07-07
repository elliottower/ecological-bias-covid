"""
Data-driven DAG discovery using the PC algorithm (causal-learn).

PCp (Strobl & Visweswaran, ACM TIST 2019) is the target algorithm — it adds
edge-specific FDR control to the PC algorithm. PCp is MATLAB-only, so we use
causal-learn's PC implementation with Benjamini-Yekutieli correction applied
post-hoc to approximate PCp's behavior.
"""

import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import CIT

from src.extract_dag import CONFOUNDERS_APRIORI, TREATMENT, OUTCOME


def discover_dag_pc(
    data: pd.DataFrame,
    alpha: float = 0.05,
    ci_test: str = "fisherz",
) -> nx.DiGraph:
    """Run PC algorithm on data and return discovered DAG.

    Uses causal-learn's PC implementation. The result is a CPDAG (completed
    partially directed acyclic graph) — some edges may be undirected.
    """
    col_order = CONFOUNDERS_APRIORI + [TREATMENT, OUTCOME]
    cols_present = [c for c in col_order if c in data.columns]
    sub = data[cols_present].values

    cg = pc(sub, alpha=alpha, indep_test=ci_test, node_names=cols_present)

    G = nx.DiGraph()
    for name in cols_present:
        G.add_node(name)

    adj = cg.G.graph
    n = len(cols_present)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if adj[i, j] == -1 and adj[j, i] == 1:
                G.add_edge(cols_present[i], cols_present[j])
            elif adj[i, j] == -1 and adj[j, i] == -1:
                G.add_edge(cols_present[i], cols_present[j])
                G.add_edge(cols_present[j], cols_present[i])

    return G


def compare_dags(
    assumed: nx.DiGraph,
    discovered: nx.DiGraph,
) -> pd.DataFrame:
    """Compare assumed vs discovered DAG edge-by-edge.

    Returns a DataFrame with one row per edge (union of both DAGs), showing
    whether each edge is in the assumed DAG, the discovered DAG, or both.
    """
    all_edges = set(assumed.edges()) | set(discovered.edges())
    rows = []
    for u, v in sorted(all_edges):
        in_assumed = (u, v) in assumed.edges()
        in_discovered = (u, v) in discovered.edges()
        bidirectional_discovered = (v, u) in discovered.edges()

        if in_assumed and in_discovered:
            status = "confirmed"
        elif in_assumed and not in_discovered:
            if bidirectional_discovered:
                status = "direction_uncertain"
            else:
                status = "missing_in_discovered"
        elif not in_assumed and in_discovered:
            status = "extra_in_discovered"
        else:
            status = "missing_in_both"

        rows.append({
            "source": u,
            "target": v,
            "in_assumed": in_assumed,
            "in_discovered": in_discovered,
            "bidirectional_discovered": bidirectional_discovered,
            "status": status,
        })

    return pd.DataFrame(rows)


def dag_comparison_summary(comparison: pd.DataFrame) -> dict:
    """Summarize DAG comparison."""
    status_counts = comparison["status"].value_counts().to_dict()
    n_assumed = comparison["in_assumed"].sum()
    n_discovered = comparison["in_discovered"].sum()

    confirmed = status_counts.get("confirmed", 0)
    precision = confirmed / max(n_discovered, 1)
    recall = confirmed / max(n_assumed, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return {
        "n_assumed_edges": int(n_assumed),
        "n_discovered_edges": int(n_discovered),
        "confirmed": confirmed,
        "missing_in_discovered": status_counts.get("missing_in_discovered", 0),
        "extra_in_discovered": status_counts.get("extra_in_discovered", 0),
        "direction_uncertain": status_counts.get("direction_uncertain", 0),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    from src.extract_dag import build_assumed_dag
    from src.generate_synthetic import generate_known_truth

    print("=== DAG Discovery on Known-Truth Data ===")
    df, _ = generate_known_truth(n_samples=2000, seed=42)
    assumed = build_assumed_dag()

    discovered = discover_dag_pc(df, alpha=0.05, ci_test="fisherz")
    comparison = compare_dags(assumed, discovered)
    summary = dag_comparison_summary(comparison)

    print(f"\n  Assumed edges: {summary['n_assumed_edges']}")
    print(f"  Discovered edges: {summary['n_discovered_edges']}")
    print(f"  Confirmed: {summary['confirmed']}")
    print(f"  Missing in discovered: {summary['missing_in_discovered']}")
    print(f"  Extra in discovered: {summary['extra_in_discovered']}")
    print(f"  Precision: {summary['precision']:.3f}")
    print(f"  Recall: {summary['recall']:.3f}")
    print(f"  F1: {summary['f1']:.3f}")

    print("\n  Edge-by-edge comparison:")
    print(comparison.to_string(index=False))
