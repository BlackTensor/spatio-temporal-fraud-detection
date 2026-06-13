"""Phase 3.4: Graph visualization — GEXF export + matplotlib subgraph plot.

Samples a representative subgraph for visualization:
  - Take all nodes from time step 1 (smallest, most interpretable)
  - Colour by label: illicit=red, licit=green, unknown=lightgray

GEXF is opened in Gephi (free, open-source) for interactive exploration.
The matplotlib PNG is exported for the README / docs.

Outputs
-------
results/graph_visualization.gexf   (Gephi-ready; nodes coloured by label)
results/graph_visualization.png    (matplotlib, ≤500 nodes for readability)
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("graph_visualization")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"

LABEL_COLOR = {1: "#e74c3c", 0: "#2ecc71", -1: "#bdc3c7"}   # red / green / gray
LABEL_NAME = {1: "illicit", 0: "licit", -1: "unknown"}


def build_nx_subgraph(snapshots: list, ts_idx: int = 0) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a single temporal snapshot."""
    snap = snapshots[ts_idx]
    G = nx.DiGraph()

    global_ids = snap.global_node_ids.tolist()
    labels = snap.y.tolist()

    for local_i, (gid, lbl) in enumerate(zip(global_ids, labels)):
        G.add_node(
            local_i,
            global_id=int(gid),
            label=int(lbl),
            label_name=LABEL_NAME[int(lbl)],
            color=LABEL_COLOR[int(lbl)],
            time_step=int(snap.time_step),
        )

    src_arr, dst_arr = snap.edge_index.tolist()
    for s, d in zip(src_arr[0] if isinstance(src_arr, list) else src_arr,
                    dst_arr[0] if isinstance(dst_arr, list) else dst_arr):
        G.add_edge(int(s), int(d))

    return G


def build_nx_subgraph_v2(snap) -> nx.DiGraph:
    """Build NetworkX DiGraph from a snapshot (handles edge_index correctly)."""
    G = nx.DiGraph()
    global_ids = snap.global_node_ids.tolist()
    labels = snap.y.tolist()

    for local_i, (gid, lbl) in enumerate(zip(global_ids, labels)):
        G.add_node(
            local_i,
            global_id=int(gid),
            label=int(lbl),
            label_name=LABEL_NAME[int(lbl)],
            color=LABEL_COLOR[int(lbl)],
            time_step=int(snap.time_step),
        )

    edge_src = snap.edge_index[0].tolist()
    edge_dst = snap.edge_index[1].tolist()
    for s, d in zip(edge_src, edge_dst):
        G.add_edge(int(s), int(d))

    return G


def export_gexf(G: nx.DiGraph, out_path: Path) -> None:
    """Export with node attributes as GEXF 1.2."""
    gexf = ET.Element("gexf", xmlns="http://gexf.net/1.2", version="1.2")
    meta = ET.SubElement(gexf, "meta")
    ET.SubElement(meta, "creator").text = "spatio-temporal-fraud-detection Phase 3.4"
    ET.SubElement(meta, "description").text = "Elliptic Bitcoin transaction subgraph"

    graph_el = ET.SubElement(gexf, "graph", defaultedgetype="directed", mode="static")

    # Attribute definitions
    attrs = ET.SubElement(graph_el, "attributes", attclass="node")
    ET.SubElement(attrs, "attribute", id="0", title="global_id", type="integer")
    ET.SubElement(attrs, "attribute", id="1", title="label", type="integer")
    ET.SubElement(attrs, "attribute", id="2", title="label_name", type="string")
    ET.SubElement(attrs, "attribute", id="3", title="time_step", type="integer")

    nodes_el = ET.SubElement(graph_el, "nodes")
    for nid, attr in G.nodes(data=True):
        n_el = ET.SubElement(nodes_el, "node", id=str(nid),
                             label=attr.get("label_name", "unknown"))
        attvals = ET.SubElement(n_el, "attvalues")
        ET.SubElement(attvals, "attvalue", **{"for": "0", "value": str(attr.get("global_id", nid))})
        ET.SubElement(attvals, "attvalue", **{"for": "1", "value": str(attr.get("label", -1))})
        ET.SubElement(attvals, "attvalue", **{"for": "2", "value": attr.get("label_name", "unknown")})
        ET.SubElement(attvals, "attvalue", **{"for": "3", "value": str(attr.get("time_step", 0))})

    edges_el = ET.SubElement(graph_el, "edges")
    for i, (src, dst) in enumerate(G.edges()):
        ET.SubElement(edges_el, "edge", id=str(i), source=str(src), target=str(dst))

    tree = ET.ElementTree(gexf)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    logger.info("Saved GEXF: %s  (%d nodes, %d edges)", out_path, G.number_of_nodes(), G.number_of_edges())


def plot_subgraph(G: nx.DiGraph, max_nodes: int, out_path: Path, title: str) -> None:
    """Plot a subgraph with at most max_nodes nodes."""
    if G.number_of_nodes() > max_nodes:
        # Keep the max_nodes highest-degree nodes for visual clarity
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    node_colors = [G.nodes[n].get("color", "#bdc3c7") for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 10))
    try:
        pos = nx.spring_layout(G, seed=42, k=1.5 / max(1, np.sqrt(G.number_of_nodes())))
    except Exception:
        pos = nx.random_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True,
                           arrowsize=8, width=0.5, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    legend_patches = [
        mpatches.Patch(color=LABEL_COLOR[1], label="Illicit"),
        mpatches.Patch(color=LABEL_COLOR[0], label="Licit"),
        mpatches.Patch(color=LABEL_COLOR[-1], label="Unknown"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved PNG: %s", out_path)


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading temporal_snapshots.pt …")
    snapshots = torch.load(PROCESSED_DIR / "temporal_snapshots.pt", weights_only=False)
    logger.info("  %d snapshots loaded", len(snapshots))

    # Use time step 1 (first snapshot — smallest, most illustrative)
    snap = snapshots[0]
    logger.info("  Time step 1: %d nodes, %d edges",
                snap.num_nodes, snap.edge_index.shape[1])

    logger.info("Building NetworkX DiGraph from ts=1 snapshot …")
    G = build_nx_subgraph_v2(snap)
    logger.info("  NetworkX graph: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges())

    logger.info("Exporting full ts=1 subgraph as GEXF …")
    export_gexf(G, RESULTS_DIR / "graph_visualization.gexf")

    logger.info("Plotting top-500-degree subgraph …")
    ts_label = int(snap.time_step)
    n_illicit = int((snap.y == 1).sum())
    n_licit = int((snap.y == 0).sum())
    title = (
        f"Elliptic Bitcoin Graph — Time Step {ts_label}  "
        f"({G.number_of_nodes()} nodes: {n_illicit} illicit / {n_licit} licit / "
        f"{G.number_of_nodes()-n_illicit-n_licit} unknown)"
    )
    plot_subgraph(G, max_nodes=500, out_path=RESULTS_DIR / "graph_visualization.png", title=title)

    print("\n=== Phase 3.4 Visualization ===")
    print(f"  Snapshot used    : time step {ts_label}")
    print(f"  Nodes in snap    : {G.number_of_nodes()}")
    print(f"  Edges in snap    : {G.number_of_edges()}")
    print(f"  GEXF             : results/graph_visualization.gexf  (open in Gephi)")
    print(f"  PNG              : results/graph_visualization.png")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
