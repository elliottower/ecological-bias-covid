"""
Encode the BCD/NTZ causal DAG from Xia et al. (medRxiv 2025, DOI 10.1101/2025.01.24.25321100).

The DAG structure is extracted from the doubly-robust semi-supervised AIPW code at
github.com/xialab2016/BCD_NTZ_SemiSupervisedCausal. The propensity model conditions
on all confounders W for treatment A, and the outcome model conditions on (A, W) for Y.
No mediators are modeled.
"""

import json
from pathlib import Path
from typing import Any

import networkx as nx


TREATMENT = "DMT"
OUTCOME = "PDDS_change"

CONFOUNDERS_APRIORI = [
    "white_nonhispanic",
    "sex",
    "age_at_dmt_initiation",
    "disease_duration",
    "days_earliest_to_study_dmt",
    "followup_duration",
    "n_ms_phecodes",
    "n_ms_cuis",
    "utilization_total",
]

CONFOUNDER_LABELS = {
    "white_nonhispanic": "Race/Ethnicity",
    "sex": "Sex",
    "age_at_dmt_initiation": "Age at DMT",
    "disease_duration": "Disease Duration",
    "days_earliest_to_study_dmt": "Prior DMT Exposure",
    "followup_duration": "Follow-up Duration",
    "n_ms_phecodes": "MS PheCode Count",
    "n_ms_cuis": "MS NLP CUI Count",
    "utilization_total": "Healthcare Utilization",
}


def build_assumed_dag() -> nx.DiGraph:
    """Build the assumed BCD/NTZ causal DAG.

    Structure: all confounders W cause both treatment A and outcome Y.
    A causes Y. No edges among confounders (they are treated as exogenous).
    """
    G = nx.DiGraph()
    G.add_node(TREATMENT, role="treatment")
    G.add_node(OUTCOME, role="outcome")
    for w in CONFOUNDERS_APRIORI:
        G.add_node(w, role="confounder", label=CONFOUNDER_LABELS.get(w, w))
        G.add_edge(w, TREATMENT)
        G.add_edge(w, OUTCOME)
    G.add_edge(TREATMENT, OUTCOME)
    return G


def get_testable_implications(G: nx.DiGraph) -> list[dict[str, Any]]:
    """Extract conditional independence implications from d-separation.

    For this DAG, the key testable implications are:
    1. Confounders are marginally independent of each other (no W_i -> W_j edges).
    2. Given all other confounders plus treatment, each confounder is conditionally
       independent of outcome ONLY through the adjustment set.

    We enumerate all pairs and test d-separation with various conditioning sets.
    """
    nodes = list(G.nodes())
    implications = []

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            for cond_set in _candidate_conditioning_sets(G, u, v):
                if nx.is_d_separator(G, {u}, {v}, set(cond_set)):
                    implications.append({
                        "X": u,
                        "Y": v,
                        "conditioning_set": sorted(cond_set),
                        "type": _classify_implication(G, u, v, cond_set),
                    })

    return implications


def _candidate_conditioning_sets(
    G: nx.DiGraph, u: str, v: str
) -> list[list[str]]:
    """Generate candidate conditioning sets for d-separation testing."""
    all_nodes = set(G.nodes()) - {u, v}
    sets = [
        [],
        list(all_nodes),
    ]
    confounders_only = [n for n in all_nodes if G.nodes[n].get("role") == "confounder"]
    if confounders_only != list(all_nodes):
        sets.append(confounders_only)
    for single in all_nodes:
        sets.append([single])
    return sets


def _classify_implication(
    G: nx.DiGraph, u: str, v: str, cond_set: list[str]
) -> str:
    roles = {G.nodes[n].get("role") for n in [u, v]}
    if roles == {"confounder"}:
        return "confounder_independence"
    if "treatment" in roles and "confounder" in roles:
        return "treatment_confounder"
    if "outcome" in roles and "confounder" in roles:
        return "outcome_confounder"
    return "other"


def dag_to_adjacency_matrix(G: nx.DiGraph) -> tuple[list[str], list[list[int]]]:
    """Convert DAG to adjacency matrix for comparison with discovered DAGs."""
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    adj = [[0] * n for _ in range(n)]
    for u, v in G.edges():
        adj[node_idx[u]][node_idx[v]] = 1
    return nodes, adj


def save_dag(G: nx.DiGraph, path: Path) -> None:
    """Save DAG as JSON for reproducibility."""
    data = {
        "nodes": [
            {"name": n, **G.nodes[n]}
            for n in G.nodes()
        ],
        "edges": [{"source": u, "target": v} for u, v in G.edges()],
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "confounders": CONFOUNDERS_APRIORI,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def print_dag_summary(G: nx.DiGraph) -> None:
    nodes = list(G.nodes())
    edges = list(G.edges())
    print(f"BCD/NTZ Assumed DAG: {len(nodes)} nodes, {len(edges)} edges")
    print(f"  Treatment: {TREATMENT}")
    print(f"  Outcome:   {OUTCOME}")
    print(f"  Confounders ({len(CONFOUNDERS_APRIORI)}):")
    for w in CONFOUNDERS_APRIORI:
        print(f"    - {w} ({CONFOUNDER_LABELS[w]})")
    print(f"\n  Edges:")
    for u, v in sorted(edges):
        print(f"    {u} -> {v}")


if __name__ == "__main__":
    G = build_assumed_dag()
    print_dag_summary(G)

    implications = get_testable_implications(G)
    print(f"\n  Testable CI implications: {len(implications)}")
    for imp in implications[:10]:
        cond_str = ", ".join(imp["conditioning_set"]) if imp["conditioning_set"] else "{}"
        print(f"    {imp['X']} _||_ {imp['Y']} | {{{cond_str}}}  [{imp['type']}]")
    if len(implications) > 10:
        print(f"    ... and {len(implications) - 10} more")

    results_dir = Path(__file__).parent.parent / "results"
    save_dag(G, results_dir / "assumed_dag.json")
    print(f"\n  DAG saved to {results_dir / 'assumed_dag.json'}")
