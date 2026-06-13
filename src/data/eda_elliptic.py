"""Phase 1.2: Exploratory Data Analysis for the Elliptic Bitcoin dataset.

Outputs:
    results/eda_report.html          self-contained HTML report (no server needed)
    results/eda_degree_histogram.png
    results/eda_edge_time_series.png
    results/eda_anomaly_over_time.png
    results/eda_class_balance.png

Usage:
    python -m src.data.eda_elliptic
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eda_elliptic")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "elliptic"
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading CSVs …")

    # Features: no header; col 0 = txId, col 1 = time_step, cols 2-166 = features
    feat_path = RAW_DIR / "elliptic_txs_features.csv"
    n_feat = 165
    feat_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, n_feat + 1)]
    features = pd.read_csv(feat_path, header=None, names=feat_cols)
    logger.info("  features: %s", features.shape)

    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv")
    logger.info("  classes : %s", classes.shape)

    edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    logger.info("  edges   : %s", edges.shape)

    return features, classes, edges


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def profile(
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    stats: dict = {}

    # --- node counts ---
    total_nodes = len(features)
    class_counts = classes["class"].value_counts()
    illicit = int(class_counts.get("1", 0))
    licit = int(class_counts.get("2", 0))
    unknown = int(class_counts.get("unknown", 0))
    labelled = illicit + licit

    stats["total_nodes"] = total_nodes
    stats["illicit_nodes"] = illicit
    stats["licit_nodes"] = licit
    stats["unknown_nodes"] = unknown
    stats["labelled_nodes"] = labelled
    stats["illicit_pct_of_labelled"] = round(100 * illicit / labelled, 2) if labelled else 0
    stats["licit_pct_of_labelled"] = round(100 * licit / labelled, 2) if labelled else 0

    # --- edge counts ---
    stats["total_edges"] = len(edges)
    stats["unique_src_nodes"] = int(edges["txId1"].nunique())
    stats["unique_dst_nodes"] = int(edges["txId2"].nunique())

    # --- time steps ---
    ts = features["time_step"]
    stats["time_steps"] = int(ts.nunique())
    stats["time_step_min"] = int(ts.min())
    stats["time_step_max"] = int(ts.max())
    stats["nodes_per_time_step_mean"] = round(float(features.groupby("time_step").size().mean()), 1)

    # --- missing values ---
    feat_missing = int(features.isnull().sum().sum())
    cls_missing = int(classes.isnull().sum().sum())
    edge_missing = int(edges.isnull().sum().sum())
    stats["missing_values_features"] = feat_missing
    stats["missing_values_classes"] = cls_missing
    stats["missing_values_edges"] = edge_missing

    # --- duplicate edges ---
    dup_edges = int(edges.duplicated().sum())
    stats["duplicate_edges"] = dup_edges

    # --- feature stats ---
    feat_only = features.drop(columns=["txId", "time_step"])
    stats["feature_count"] = len(feat_only.columns)
    stats["feature_mean_global"] = round(float(feat_only.mean().mean()), 4)
    stats["feature_std_global"] = round(float(feat_only.std().mean()), 4)

    return stats


# ---------------------------------------------------------------------------
# Degree computation
# ---------------------------------------------------------------------------

def compute_degrees(features: pd.DataFrame, edges: pd.DataFrame) -> pd.Series:
    all_ids = set(features["txId"])
    out_deg = edges["txId1"].value_counts()
    in_deg = edges["txId2"].value_counts()
    total_deg = out_deg.add(in_deg, fill_value=0).reindex(list(all_ids), fill_value=0)
    return total_deg


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_class_balance(classes: pd.DataFrame, save_path: Path) -> str:
    counts = {"Illicit": int((classes["class"] == "1").sum()),
              "Licit": int((classes["class"] == "2").sum()),
              "Unknown": int((classes["class"] == "unknown").sum())}
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c", "#2ecc71", "#95a5a6"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Node Class Distribution", fontsize=13, fontweight="bold")
    ax.set_ylabel("Node Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def plot_degree_histogram(degrees: pd.Series, save_path: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # linear scale
    axes[0].hist(degrees, bins=60, color="#3498db", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Degree Distribution (linear)", fontweight="bold")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Node Count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    # log-log scale
    deg_nonzero = degrees[degrees > 0]
    axes[1].hist(deg_nonzero, bins=np.logspace(0, np.log10(deg_nonzero.max() + 1), 50),
                 color="#9b59b6", edgecolor="white", linewidth=0.4)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Degree Distribution (log-log)", fontweight="bold")
    axes[1].set_xlabel("Degree (log)")
    axes[1].set_ylabel("Node Count (log)")
    axes[1].grid(axis="both", alpha=0.3, which="both")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Transaction Node Degree Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def plot_edge_time_series(features: pd.DataFrame, edges: pd.DataFrame, save_path: Path) -> str:
    # Attach time step to each edge via the source node
    ts_map = features.set_index("txId")["time_step"]
    src_ts = edges["txId1"].map(ts_map)
    edges_per_ts = src_ts.value_counts().sort_index()

    nodes_per_ts = features.groupby("time_step").size()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(nodes_per_ts.index, nodes_per_ts.values, color="#3498db", alpha=0.8)
    axes[0].set_title("Nodes per Time Step", fontweight="bold")
    axes[0].set_ylabel("Node Count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].bar(edges_per_ts.index, edges_per_ts.values, color="#e67e22", alpha=0.8)
    axes[1].set_title("Edges per Time Step (by source node's time step)", fontweight="bold")
    axes[1].set_xlabel("Time Step (1 = earliest, 49 = latest)")
    axes[1].set_ylabel("Edge Count")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Temporal Activity Over 49 Time Steps", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def plot_anomaly_over_time(features: pd.DataFrame, classes: pd.DataFrame, save_path: Path) -> str:
    merged = features[["txId", "time_step"]].merge(classes, on="txId")
    labelled = merged[merged["class"] != "unknown"].copy()
    labelled["is_illicit"] = (labelled["class"] == "1").astype(int)

    ts_illicit = labelled.groupby("time_step")["is_illicit"].sum()
    ts_licit = labelled.groupby("time_step").apply(lambda g: (g["class"] == "2").sum())
    ts_rate = labelled.groupby("time_step")["is_illicit"].mean() * 100

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ts_steps = ts_illicit.index
    axes[0].bar(ts_steps, ts_licit.values, label="Licit", color="#2ecc71", alpha=0.8)
    axes[0].bar(ts_steps, ts_illicit.values, bottom=ts_licit.values, label="Illicit", color="#e74c3c", alpha=0.8)
    axes[0].set_title("Labelled Nodes per Time Step (Stacked)", fontweight="bold")
    axes[0].set_ylabel("Node Count")
    axes[0].legend(loc="upper right")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(ts_rate.index, ts_rate.values, color="#e74c3c", marker="o", markersize=3, linewidth=1.5)
    axes[1].fill_between(ts_rate.index, ts_rate.values, alpha=0.15, color="#e74c3c")
    axes[1].set_title("Illicit Rate Over Time (% of labelled nodes)", fontweight="bold")
    axes[1].set_xlabel("Time Step (1 = earliest, 49 = latest)")
    axes[1].set_ylabel("Illicit %")
    axes[1].set_ylim(0, max(ts_rate.max() * 1.15, 5))
    axes[1].grid(axis="both", alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Anomaly Distribution Across Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Elliptic Bitcoin Dataset — EDA Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f6fa; color: #333; }}
  header {{ background: #1a1a2e; color: white; padding: 24px 40px; }}
  header h1 {{ margin: 0; font-size: 1.8rem; }}
  header p {{ margin: 4px 0 0; opacity: .7; font-size: .9rem; }}
  main {{ max-width: 1100px; margin: 32px auto; padding: 0 24px; }}
  section {{ background: white; border-radius: 10px; padding: 28px 32px; margin-bottom: 28px;
             box-shadow: 0 2px 8px rgba(0,0,0,.07); }}
  h2 {{ margin-top: 0; color: #1a1a2e; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  th {{ background: #1a1a2e; color: white; padding: 10px 14px; text-align: left; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f4ff; }}
  .metric {{ display: inline-block; background: #f0f4ff; border-radius: 8px;
             padding: 14px 22px; margin: 8px; text-align: center; min-width: 140px; }}
  .metric .val {{ font-size: 1.7rem; font-weight: 700; color: #1a1a2e; }}
  .metric .lbl {{ font-size: .75rem; color: #666; margin-top: 2px; }}
  img {{ max-width: 100%; border-radius: 8px; margin-top: 16px; }}
  .tag {{ display: inline-block; background: #e8f5e9; color: #2e7d32; border-radius: 4px;
          padding: 2px 8px; font-size: .8rem; font-weight: 600; }}
  .warn {{ background: #fff3e0; color: #e65100; }}
</style>
</head>
<body>
<header>
  <h1>Elliptic Bitcoin Dataset — EDA Report</h1>
  <p>Phase 1.2 · Spatio-Temporal Fraud Detection Project</p>
</header>
<main>

<section>
  <h2>Dataset Overview</h2>
  <div>
    <div class="metric"><div class="val">{total_nodes:,}</div><div class="lbl">Total Nodes</div></div>
    <div class="metric"><div class="val">{total_edges:,}</div><div class="lbl">Total Edges</div></div>
    <div class="metric"><div class="val">{illicit_nodes:,}</div><div class="lbl">Illicit Nodes</div></div>
    <div class="metric"><div class="val">{licit_nodes:,}</div><div class="lbl">Licit Nodes</div></div>
    <div class="metric"><div class="val">{unknown_nodes:,}</div><div class="lbl">Unknown Nodes</div></div>
    <div class="metric"><div class="val">{time_steps}</div><div class="lbl">Time Steps</div></div>
    <div class="metric"><div class="val">{feature_count}</div><div class="lbl">Node Features</div></div>
  </div>
</section>

<section>
  <h2>Class Balance</h2>
  <table>
    <tr><th>Class</th><th>Count</th><th>% of Labelled</th><th>% of All Nodes</th></tr>
    <tr><td><span class="tag warn">Illicit (1)</span></td>
        <td>{illicit_nodes:,}</td>
        <td>{illicit_pct_of_labelled}%</td>
        <td>{illicit_pct_all:.2f}%</td></tr>
    <tr><td><span class="tag">Licit (2)</span></td>
        <td>{licit_nodes:,}</td>
        <td>{licit_pct_of_labelled}%</td>
        <td>{licit_pct_all:.2f}%</td></tr>
    <tr><td>Unknown</td>
        <td>{unknown_nodes:,}</td>
        <td>—</td>
        <td>{unknown_pct_all:.2f}%</td></tr>
    <tr><td><strong>Total</strong></td>
        <td><strong>{total_nodes:,}</strong></td><td>—</td><td>100%</td></tr>
  </table>
  <p><strong>Imbalance ratio</strong> (licit : illicit among labelled): {imbalance_ratio:.1f} : 1</p>
  <img src="data:image/png;base64,{b64_class_balance}" alt="Class balance chart">
</section>

<section>
  <h2>Graph Structure</h2>
  <table>
    <tr><th>Statistic</th><th>Value</th></tr>
    <tr><td>Total directed edges</td><td>{total_edges:,}</td></tr>
    <tr><td>Unique source nodes (txId1)</td><td>{unique_src_nodes:,}</td></tr>
    <tr><td>Unique destination nodes (txId2)</td><td>{unique_dst_nodes:,}</td></tr>
    <tr><td>Duplicate edges</td><td>{duplicate_edges}</td></tr>
    <tr><td>Isolated nodes (degree 0)</td><td>{isolated_nodes:,}</td></tr>
    <tr><td>Median degree</td><td>{median_degree:.1f}</td></tr>
    <tr><td>Mean degree</td><td>{mean_degree:.2f}</td></tr>
    <tr><td>Max degree</td><td>{max_degree:,}</td></tr>
    <tr><td>95th percentile degree</td><td>{p95_degree:.0f}</td></tr>
  </table>
  <img src="data:image/png;base64,{b64_degree}" alt="Degree distribution">
</section>

<section>
  <h2>Temporal Structure</h2>
  <table>
    <tr><th>Statistic</th><th>Value</th></tr>
    <tr><td>Time steps</td><td>{time_steps} (step 1 = earliest, 49 = latest)</td></tr>
    <tr><td>Nodes per time step (mean)</td><td>{nodes_per_time_step_mean:,}</td></tr>
    <tr><td>Edges per time step (mean)</td><td>{edges_per_ts_mean:.0f}</td></tr>
  </table>
  <img src="data:image/png;base64,{b64_edge_ts}" alt="Edge time series">
</section>

<section>
  <h2>Anomaly Distribution Over Time</h2>
  <img src="data:image/png;base64,{b64_anomaly_ts}" alt="Anomaly over time">
</section>

<section>
  <h2>Feature Statistics</h2>
  <table>
    <tr><th>Statistic</th><th>Value</th></tr>
    <tr><td>Feature columns</td><td>{feature_count} (feat_1 … feat_165)</td></tr>
    <tr><td>Missing values in features</td><td>{missing_values_features}</td></tr>
    <tr><td>Missing values in classes</td><td>{missing_values_classes}</td></tr>
    <tr><td>Missing values in edges</td><td>{missing_values_edges}</td></tr>
    <tr><td>Global mean (across all features)</td><td>{feature_mean_global}</td></tr>
    <tr><td>Global std (across all features)</td><td>{feature_std_global}</td></tr>
  </table>
  <p>Features are already standardized (published dataset comes pre-normalized).</p>
</section>

<section>
  <h2>Key Findings</h2>
  <ul>
    <li><strong>Heavy class imbalance:</strong> only {illicit_pct_of_labelled}% of labelled nodes are illicit — requires weighted loss or oversampling.</li>
    <li><strong>Large unknown set:</strong> {unknown_nodes:,} nodes ({unknown_pct_all:.1f}%) have no label — semi-supervised or transductive strategies needed.</li>
    <li><strong>No missing values:</strong> dataset is clean and ready for Phase 1.3 processing.</li>
    <li><strong>No duplicate edges:</strong> edge list is already deduplicated.</li>
    <li><strong>Power-law degree distribution:</strong> most nodes have degree 1–2; a few hubs have very high degree — hub nodes may be informative features.</li>
    <li><strong>Temporal consistency:</strong> 49 time steps with roughly stable node counts per step — suitable for snapshot-based temporal GNN.</li>
  </ul>
</section>

</main>
</body>
</html>
"""


def build_report(
    stats: dict,
    degrees: pd.Series,
    features: pd.DataFrame,
    classes: pd.DataFrame,
    edges: pd.DataFrame,
) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    b64_class = plot_class_balance(classes, RESULTS_DIR / "eda_class_balance.png")
    b64_deg = plot_degree_histogram(degrees, RESULTS_DIR / "eda_degree_histogram.png")
    b64_ets = plot_edge_time_series(features, edges, RESULTS_DIR / "eda_edge_time_series.png")
    b64_ats = plot_anomaly_over_time(features, classes, RESULTS_DIR / "eda_anomaly_over_time.png")

    total = stats["total_nodes"]
    illicit = stats["illicit_nodes"]
    licit = stats["licit_nodes"]
    unknown = stats["unknown_nodes"]
    labelled = stats["labelled_nodes"]

    # edges per ts
    ts_map = features.set_index("txId")["time_step"]
    src_ts = edges["txId1"].map(ts_map)
    edges_per_ts_mean = float(src_ts.value_counts().mean())

    isolated = int((degrees == 0).sum())

    html = REPORT_TEMPLATE.format(
        **stats,
        illicit_pct_all=100 * illicit / total,
        licit_pct_all=100 * licit / total,
        unknown_pct_all=100 * unknown / total,
        imbalance_ratio=licit / illicit if illicit else float("inf"),
        isolated_nodes=isolated,
        median_degree=float(degrees.median()),
        mean_degree=float(degrees.mean()),
        max_degree=int(degrees.max()),
        p95_degree=float(np.percentile(degrees, 95)),
        edges_per_ts_mean=edges_per_ts_mean,
        b64_class_balance=b64_class,
        b64_degree=b64_deg,
        b64_edge_ts=b64_ets,
        b64_anomaly_ts=b64_ats,
    )
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    features, classes, edges = load_data()

    logger.info("Profiling …")
    stats = profile(features, classes, edges)

    logger.info("Computing degrees …")
    degrees = compute_degrees(features, edges)

    logger.info("Building report …")
    html = build_report(stats, degrees, features, classes, edges)

    report_path = RESULTS_DIR / "eda_report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Also dump stats as JSON for downstream use
    stats_path = RESULTS_DIR / "eda_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats JSON written to %s", stats_path)

    # Print summary to console
    print("\n=== EDA Summary ===")
    print(f"  Nodes          : {stats['total_nodes']:,}")
    print(f"  Edges          : {stats['total_edges']:,}")
    print(f"  Time steps     : {stats['time_steps']}")
    print(f"  Illicit        : {stats['illicit_nodes']:,}  ({stats['illicit_pct_of_labelled']}% of labelled)")
    print(f"  Licit          : {stats['licit_nodes']:,}  ({stats['licit_pct_of_labelled']}% of labelled)")
    print(f"  Unknown        : {stats['unknown_nodes']:,}")
    print(f"  Missing values : {stats['missing_values_features'] + stats['missing_values_classes'] + stats['missing_values_edges']}")
    print(f"  Duplicate edges: {stats['duplicate_edges']}")
    print(f"  Report         : {RESULTS_DIR / 'eda_report.html'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
