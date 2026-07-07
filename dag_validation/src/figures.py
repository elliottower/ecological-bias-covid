"""
Figure generation for the Visweswaran DAG validation experiment.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx


FIGURE_DIR = Path(__file__).parent.parent / "results" / "figures"


def setup_style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def plot_annotated_dag(
    G: nx.DiGraph,
    ci_results: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot the BCD/NTZ DAG with edges colored by CI test verdict."""
    setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    role_colors = {
        "confounder": "#4ECDC4",
        "treatment": "#FF6B6B",
        "outcome": "#45B7D1",
    }
    node_colors = [role_colors.get(G.nodes[n].get("role", ""), "#999") for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42, k=2.0)

    edge_colors = []
    if ci_results is not None:
        edge_verdicts = {}
        for _, row in ci_results.iterrows():
            key = (row["X"], row["Y"])
            if row.get("kernel_verdict") == "independent":
                edge_verdicts[key] = "#2ecc71"
            elif row.get("kernel_verdict") == "dependent":
                edge_verdicts[key] = "#e74c3c"
            else:
                edge_verdicts[key] = "#95a5a6"

        for u, v in G.edges():
            color = edge_verdicts.get((u, v), edge_verdicts.get((v, u), "#333"))
            edge_colors.append(color)
    else:
        edge_colors = ["#333"] * len(G.edges())

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1200, edgecolors="#333", linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=2, arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.1")

    labels = {}
    for n in G.nodes():
        label = G.nodes[n].get("label", n)
        if len(label) > 15:
            words = label.split()
            mid = len(words) // 2
            label = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        labels[n] = label
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    legend_elements = [
        mpatches.Patch(color="#4ECDC4", label="Confounder"),
        mpatches.Patch(color="#FF6B6B", label="Treatment (DMT)"),
        mpatches.Patch(color="#45B7D1", label="Outcome (PDDS)"),
    ]
    if ci_results is not None:
        legend_elements.extend([
            mpatches.Patch(color="#2ecc71", label="CI holds (consistent)"),
            mpatches.Patch(color="#e74c3c", label="CI violated"),
        ])
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_title("BCD/NTZ Assumed DAG — CI Test Results")
    ax.axis("off")

    if output_path is None:
        output_path = FIGURE_DIR / "F1_dag_annotated.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_dag_comparison(
    assumed: nx.DiGraph,
    discovered: nx.DiGraph,
    comparison: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plot assumed vs discovered DAG side-by-side."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    pos = nx.spring_layout(assumed, seed=42, k=2.0)

    role_colors = {"confounder": "#4ECDC4", "treatment": "#FF6B6B", "outcome": "#45B7D1"}
    node_colors = [role_colors.get(assumed.nodes[n].get("role", ""), "#999") for n in assumed.nodes()]

    nx.draw(assumed, pos, ax=ax1, node_color=node_colors, node_size=800,
            with_labels=True, font_size=6, edge_color="#333",
            arrows=True, arrowsize=12)
    ax1.set_title(f"Assumed DAG ({len(assumed.edges())} edges)")

    disc_colors = []
    for u, v in discovered.edges():
        mask = (comparison["source"] == u) & (comparison["target"] == v)
        if mask.any():
            status = comparison.loc[mask, "status"].values[0]
            if status == "confirmed":
                disc_colors.append("#2ecc71")
            elif status == "extra_in_discovered":
                disc_colors.append("#e74c3c")
            else:
                disc_colors.append("#f39c12")
        else:
            disc_colors.append("#e74c3c")

    disc_node_colors = [role_colors.get(assumed.nodes.get(n, {}).get("role", ""), "#999")
                        for n in discovered.nodes()]
    nx.draw(discovered, pos, ax=ax2, node_color=disc_node_colors, node_size=800,
            with_labels=True, font_size=6, edge_color=disc_colors,
            arrows=True, arrowsize=12)
    ax2.set_title(f"Discovered DAG ({len(discovered.edges())} edges)")

    legend = [
        mpatches.Patch(color="#2ecc71", label="Confirmed"),
        mpatches.Patch(color="#e74c3c", label="Extra/Missing"),
        mpatches.Patch(color="#f39c12", label="Direction uncertain"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Assumed vs PC-Discovered DAG", fontsize=14)

    if output_path is None:
        output_path = FIGURE_DIR / "F2_dag_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_calibration(
    known_truth: dict,
    ci_results: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """Plot RCIT calibration: observed vs expected type I/II error."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    true_ci = []
    true_dep = []
    for _, row in ci_results.iterrows():
        key_tuple = None
        for k in known_truth:
            if k[0] == row["X"] and k[1] == row["Y"]:
                key_tuple = k
                break
            if k[0] == row["Y"] and k[1] == row["X"]:
                key_tuple = k
                break
        if key_tuple is None:
            continue
        is_independent = known_truth[key_tuple]
        if is_independent:
            true_ci.append(row)
        else:
            true_dep.append(row)

    for method, label, color in [
        ("fisher_z_pval", "Fisher's z", "#3498db"),
        ("kernel_pval", "Kernel CIT", "#e74c3c"),
    ]:
        if true_ci:
            pvals = [r[method] for r in true_ci if not np.isnan(r[method])]
            if pvals:
                ax1.hist(pvals, bins=20, alpha=0.5, label=label, color=color, density=True)
        if true_dep:
            pvals = [r[method] for r in true_dep if not np.isnan(r[method])]
            if pvals:
                ax2.hist(pvals, bins=20, alpha=0.5, label=label, color=color, density=True)

    ax1.axhline(y=1, color="black", linestyle="--", alpha=0.5, label="Uniform (calibrated)")
    ax1.set_title("Truly Independent Pairs\n(should be uniform)")
    ax1.set_xlabel("p-value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    ax2.axvline(x=0.05, color="black", linestyle="--", alpha=0.5, label="alpha=0.05")
    ax2.set_title("Truly Dependent Pairs\n(should be near 0)")
    ax2.set_xlabel("p-value")
    ax2.legend(fontsize=8)

    fig.suptitle("CI Test Calibration on Known-Truth Data", fontsize=13)

    if output_path is None:
        output_path = FIGURE_DIR / "F5_calibration.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_sheaf_vs_cochran(
    comparison: dict,
    output_path: Path | None = None,
) -> None:
    """Plot sheaf inconsistency vs Cochran's Q across outcomes."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cochran = comparison["cochran"]
    outcomes = sorted(cochran.keys())
    q_values = [cochran[o]["Q"] for o in outcomes]
    q_pvals = [cochran[o]["analytic_p"] for o in outcomes]
    k_sites = [cochran[o]["k_sites"] for o in outcomes]

    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in q_pvals]
    bars = ax1.barh(outcomes, q_values, color=colors)
    ax1.set_xlabel("Cochran's Q")
    ax1.set_title("Cochran's Q per Outcome")
    for bar, k in zip(bars, k_sites):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"k={k}", va="center", fontsize=7)

    labels = ["Sheaf H^1\n(all sites)", "Cochran Q\n(max per outcome)"]
    sheaf_z = comparison["sheaf"]["z_score"]
    cochran_max_z = max(
        (cochran[o]["z_score"] for o in outcomes if not np.isnan(cochran[o]["z_score"])),
        default=0,
    )
    z_values = [sheaf_z, cochran_max_z]
    bar_colors = ["#3498db", "#e74c3c"]
    ax2.bar(labels, z_values, color=bar_colors)
    ax2.set_ylabel("z-score (permutation null)")
    ax2.set_title("Sheaf vs Cochran: Permutation z-scores")

    sheaf_n = comparison["advantage"]["sheaf_n_sites"]
    cochran_max = comparison["advantage"]["cochran_max_sites"]
    ax2.text(0, sheaf_z + 0.1, f"n={sheaf_n} sites", ha="center", fontsize=8)
    ax2.text(1, cochran_max_z + 0.1, f"max k={cochran_max}", ha="center", fontsize=8)

    fig.suptitle("Sheaf Consistency vs Cochran's Q on 4CE-like Data", fontsize=13)

    if output_path is None:
        output_path = FIGURE_DIR / "F4_sheaf_4ce.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved {output_path}")
