"""
Spatio-Temporal Fraud Detection — Interactive Streamlit Dashboard
Phase 12: Interactive Demo (free hosting on Streamlit Community Cloud)

Run locally:  streamlit run dashboard/app.py
Deploy free:  https://streamlit.io/cloud  (connect GitHub repo)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "processed"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection GNN",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ────────────────────────────────────────────────────────────────
# Single source of truth for colors, mirrored from .streamlit/config.toml so the
# Plotly figures match the app chrome exactly.
C = {
    "blue": "#60A5FA",
    "green": "#34D399",
    "violet": "#A78BFA",
    "red": "#F87171",
    "yellow": "#FBBF24",
    "sky": "#38BDF8",
    "orange": "#FB923C",
    "gray": "#94A3B8",
    "ink": "#F1F5F9",      # primary text
    "muted": "#94A3B8",    # secondary text
    "line": "#1E293B",     # gridlines
    "border": "#334155",   # card / axis borders
    "card": "#1E293B",     # card background
    "bg": "#0F172A",       # page background
}

PALETTE = [C["blue"], C["green"], C["violet"], C["red"],
           C["yellow"], C["sky"], C["orange"], C["gray"]]

PHASE_COLORS = {
    "Phase 2": C["sky"],
    "Phase 4": C["blue"],
    "Phase 5": C["violet"],
    "Phase 6": C["green"],
}

LABEL_COLORS = {"illicit": C["red"], "licit": C["green"], "unknown": C["gray"]}

# Sequential scale for heatmaps (slate → ice blue), matching config.toml
SEQ_BLUE = [
    [0.0, "#0C4A6E"], [0.5, "#0EA5E9"], [1.0, "#E0F2FE"],
]

# ── Custom Plotly template ───────────────────────────────────────────────────────
# Transparent backgrounds so figures sit flush on the slate page/cards, with the
# unified palette + Inter typography. Registered as default; charts are rendered
# with theme=None so this template is authoritative.
pio.templates["fraud"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif",
                  color="#CBD5E1", size=13),
        title=dict(font=dict(family="Inter", size=15, color=C["ink"]),
                   x=0.0, xanchor="left", y=0.97),
        xaxis=dict(gridcolor=C["line"], linecolor=C["border"], zerolinecolor=C["line"],
                   title_font=dict(size=12, color=C["muted"]),
                   tickfont=dict(color=C["muted"], size=11)),
        yaxis=dict(gridcolor=C["line"], linecolor=C["border"], zerolinecolor=C["line"],
                   title_font=dict(size=12, color=C["muted"]),
                   tickfont=dict(color=C["muted"], size=11)),
        legend=dict(font=dict(color="#CBD5E1", size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=54, b=46, l=56, r=24),
    )
)
pio.templates.default = "fraud"


def show(fig):
    """Render a figure with our template authoritative (no Streamlit re-theming)."""
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ── Global CSS ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* Tighter page rhythm */
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }

      /* Typography hierarchy */
      h1 { font-weight: 700; letter-spacing: -0.02em; font-size: 1.85rem !important; }
      h2 { font-weight: 600; letter-spacing: -0.01em; }
      h3 {
        font-weight: 600; font-size: 1.12rem !important;
        margin-top: 1.9rem !important; margin-bottom: 0.9rem !important;
        padding-bottom: 0.4rem; border-bottom: 1px solid #1E293B;
        color: #F1F5F9;
      }
      .block-container p, .block-container li { color: #CBD5E1; line-height: 1.6; }

      /* Metric cards */
      [data-testid="stMetric"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 14px 16px;
      }
      [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important; font-size: 0.74rem !important;
        font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;
      }
      [data-testid="stMetricValue"] { font-weight: 700; font-size: 1.55rem; }
      [data-testid="stMetricDelta"] { font-size: 0.82rem; }

      /* Reduce default horizontal-rule weight */
      hr { margin: 1.4rem 0; border-color: #1E293B; }

      /* Dataframes blend with the slate theme */
      [data-testid="stDataFrame"] { border-radius: 10px; }

      /* Sidebar brand + nav */
      .fd-brand {
        font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
        color: #F1F5F9; margin: 0.2rem 0 0.1rem 0;
      }
      .fd-brand-sub {
        color: #94A3B8; font-size: 0.78rem; line-height: 1.45; margin-bottom: 0.4rem;
      }
      [data-testid="stSidebar"] [role="radiogroup"] { gap: 0.15rem; }
      [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 0.28rem 0.2rem; border-radius: 6px;
      }
      .fd-foot { color: #64748B; font-size: 0.74rem; line-height: 1.5; }

      /* Custom content cards */
      .fd-card {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 15px 17px; height: 100%;
      }
      .fd-card .t { font-weight: 600; font-size: 0.92rem; color: #F1F5F9; margin-bottom: 0.45rem; }
      .fd-card .b { color: #CBD5E1; font-size: 0.83rem; line-height: 1.55; }

      /* Architecture pipeline steps */
      .fd-step {
        background: #1E293B; border: 1px solid #334155; border-radius: 10px;
        padding: 16px 12px; text-align: center; height: 100%;
      }
      .fd-step .n {
        color: #60A5FA; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.12em;
      }
      .fd-step .t { font-weight: 600; font-size: 0.95rem; color: #F1F5F9; margin: 0.35rem 0 0.4rem; }
      .fd-step .b { color: #94A3B8; font-size: 0.76rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def load_evaluation():
    with open(RESULTS / "all_models_evaluation.json") as f:
        return json.load(f)


@st.cache_data
def load_latency():
    return pd.read_csv(RESULTS / "latency_benchmark.csv")


@st.cache_data
def load_drift():
    with open(RESULTS / "concept_drift_analysis.json") as f:
        return json.load(f)


@st.cache_data
def load_top100():
    return pd.read_csv(RESULTS / "top_100_anomalies.csv")


@st.cache_data
def load_temporal_evolution():
    with open(RESULTS / "temporal_evolution.json") as f:
        return json.load(f)


@st.cache_data
def load_shap():
    with open(RESULTS / "shap_values.json") as f:
        return json.load(f)


@st.cache_data
def load_attention():
    with open(RESULTS / "attention_weights_summary.json") as f:
        return json.load(f)


@st.cache_data
def load_scalability():
    with open(RESULTS / "scalability_analysis.json") as f:
        return json.load(f)


@st.cache_data
def load_eda_stats():
    with open(RESULTS / "eda_stats.json") as f:
        return json.load(f)


def info_card(title, body, accent=C["blue"]):
    """Styled card replacing default st.info() boxes."""
    body = body.replace("\n", "<br>")
    st.markdown(
        f"<div class='fd-card' style='border-top:2px solid {accent}'>"
        f"<div class='t'>{title}</div><div class='b'>{body}</div></div>",
        unsafe_allow_html=True,
    )


MODEL_ORDER = [
    "xgboost", "gcn", "graphsage", "gat",
    "temporal_snapshot_gnn", "evolve_gcn",
    "hetero_sage", "hgat", "htgn",
]

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("<div class='fd-brand'>Fraud Detection GNN</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<div class='fd-brand-sub'>Elliptic Bitcoin Dataset<br/>Phases 2–11 complete</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Anomaly Explorer", "Model Comparison",
     "Concept Drift", "Interpretability", "Scalability"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='fd-foot'><b style='color:#94A3B8'>$0 budget</b> · Local CPU (ARM)<br/>"
    "Free deploy: <a href='https://streamlit.io/cloud'>Streamlit Cloud</a></div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Spatio-Temporal Fraud Detection with GNNs")
    st.markdown(
        "<div style='color:#94A3B8;font-size:0.85rem;margin:-0.6rem 0 0.7rem;"
        "letter-spacing:0.02em;'>Architected by "
        "<b style='color:#CBD5E1'>Shayan Ansari</b> · 2026</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "A research-grade Graph Neural Network system trained on the "
        "[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) "
        "to detect illicit transactions in dynamic, temporal graphs.  \n"
        "**Built entirely on free tooling — $0 budget enforced.**"
    )

    st.markdown("### Why This Project Exists")
    st.markdown(
        "Does graph, temporal, or heterogeneous structure actually help fraud detection "
        "**generalize as fraud evolves** — or does it just add complexity and cost? "
        "To find out, this project builds a *ladder* of models on the same temporally-split "
        "Elliptic data — from a feature-only **XGBoost control** up to a heterogeneous "
        "temporal GNN — and measures each rung's lift on a future time window it never "
        "trained on. Every model is judged not by how well it fits the past, but by how "
        "well it holds up once the world moves."
    )

    st.markdown("---")

    # ── Dataset stats ────────────────────────────────────────────────────────
    eda = load_eda_stats()
    st.markdown("### Dataset — Elliptic Bitcoin Transaction Graph")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nodes", f"{eda['total_nodes']:,}")
    c2.metric("Edges", f"{eda['total_edges']:,}")
    c3.metric("Time Steps", eda["time_steps"])
    c4.metric("Features / node", eda["feature_count"])
    c5.metric("Illicit (labeled)", f"{eda['illicit_pct_of_labelled']}%")

    st.markdown("---")

    # ── Headline results ─────────────────────────────────────────────────────
    st.markdown("### Headline Results (Test Set, ts 43–49)")
    ev = load_evaluation()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Best Test AUC",
        "0.777",
        "+14% vs XGBoost",
        help="GraphSAGE — best cross-temporal generalization",
    )
    r2.metric(
        "Best Val AUC",
        "0.960",
        "HTGN (Phase 6)",
        help="Heterogeneous Temporal GNN — highest same-distribution AUC",
    )
    r3.metric(
        "Fastest inference",
        "275 ms",
        "full graph (203k nodes)",
        help="GraphSAGE on ARM CPU",
    )
    r4.metric(
        "Concept drift",
        "11.6% → 2.5%",
        "illicit prevalence shift",
        help="Train→test illicit drop is the primary challenge",
        delta_color="inverse",
    )

    st.markdown("---")

    # ── Architecture pipeline ────────────────────────────────────────────────
    st.markdown("### Architecture Pipeline")
    cols = st.columns(5)
    steps = [
        ("Data", "Elliptic CSVs\n203k nodes\n49 snapshots"),
        ("Features", "169 features\n+ structural\n+ temporal"),
        ("Graph", "PyG Data\nEdge-typed\nTemporal slices"),
        ("GNN", "GraphSAGE\nGCN · GAT\nHetero · Temporal"),
        ("Anomaly", "Score + rank\nSHAP explain\nThreshold tune"),
    ]
    for i, (col, (title, body)) in enumerate(zip(cols, steps), start=1):
        col.markdown(
            f"<div class='fd-step'>"
            f"<div class='n'>{i:02d}</div>"
            f"<div class='t'>{title}</div>"
            f"<div class='b'>{body.replace(chr(10), '<br/>')}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Phase summary table ───────────────────────────────────────────────────
    st.markdown("### Model Evolution Across Phases")
    summary_rows = []
    for key in MODEL_ORDER:
        m = ev[key]
        summary_rows.append({
            "Model": m["display_name"],
            "Phase": m["phase"],
            "Val AUC": f"{m['val']['roc_auc']:.3f}",
            "Test AUC": f"{m['test']['roc_auc']:.3f}",
            "Val F1": f"{m['val']['f1']:.3f}",
            "Test F1": f"{m['test']['f1']:.3f}",
        })
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### The Investigation")
    st.markdown(
        "The Elliptic timeline hides a brutal distribution shift: illicit prevalence "
        "collapses from **11.6%** (train) → **9.2%** (val) → **2.5%** (test). Climbing the "
        "model ladder against that shift produced one consistent — and counterintuitive — "
        "story. (The *Concept Drift* page has the per-time-step breakdown.)"
    )

    beats = [
        ("1 · The baseline collapse",
         "XGBoost on node features alone scored an excellent <b>0.914 validation F1</b> "
         "in-distribution — then cratered to <b>0.032 test F1</b> on the later period, "
         "barely above chance. A feature-only model simply can't track fraud once the "
         "patterns themselves evolve over time.",
         C["red"]),
        ("2 · Graph structure helps",
         "Adding the transaction graph lifted GraphSAGE to <b>0.777 test AUC — +14% over "
         "XGBoost's 0.683</b>. Relationships carry signal that survives the drift. But the "
         "<b>type</b> of graph model matters: GCN underperformed because money flows are "
         "<b>directed</b> (A→B), and its symmetric message-passing blurs that asymmetry. "
         "GAT only matched the XGBoost baseline — at appreciably higher compute cost.",
         C["green"]),
        ("3 · Temporal memory hurts",
         "Explicitly modelling the sequence of past snapshots (SnapshotGNN, EvolveGCN) "
         "raised <b>validation</b> AUC — yet <b>lowered</b> test AUC. The added memory "
         "latched onto historical fraud rhythms that simply didn't recur in the test period.",
         C["yellow"]),
        ("4 · Heterogeneous structure, same trap",
         "Direction-typed heterogeneous message passing told the identical story. HTGN "
         "posted the project's <b>best-ever validation AUC (0.960)</b> — and still fell "
         "short of plain GraphSAGE on the test set.",
         C["violet"]),
        ("5 · The throughline",
         "Every added layer of sophistication improved <b>same-distribution</b> "
         "(validation) performance and degraded <b>cross-time</b> (test) generalization. "
         "The simplest model was simultaneously the most robust <b>and</b> the fastest.",
         C["blue"]),
    ]
    for i, (b_title, b_body, b_accent) in enumerate(beats):
        info_card(b_title, b_body, accent=b_accent)
        if i < len(beats) - 1:
            st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    st.markdown("### Structural Discovery")
    info_card(
        "49 disconnected subgraphs — one per time step",
        "The Elliptic graph isn't a single connected network. It decomposes into "
        "<b>49 fully disconnected components, one per time step</b> — transactions never "
        "link across time periods. That made temporal snapshotting the <b>natural</b> unit "
        "of modelling, not an arbitrary design choice.",
        accent=C["sky"],
    )

    st.markdown("### Practitioner Takeaway")
    st.markdown(
        "<div class='fd-card' style='border-left:4px solid #60A5FA;background:#16233b'>"
        "<div class='t'>Prefer simple, robust models — spend complexity elsewhere</div>"
        "<div class='b'>Under severe concept drift, reach for <b>simple inductive "
        "architectures</b> before complex temporal or heterogeneous ones — the extra "
        "capacity tends to memorise patterns that won't persist. Put the engineering "
        "effort into <b>drift detection and retraining triggers</b> instead. And there's "
        "no accuracy-for-speed tradeoff to weigh: every GNN variant ran comfortably under "
        "the <b>100 ms / node</b> latency target (≈1–2 ms per 1,000 nodes), so GraphSAGE "
        "wins on robustness <b>and</b> speed.</div></div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANOMALY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Anomaly Explorer":
    st.title("Anomaly Explorer")
    st.markdown(
        "Explore **GraphSAGE** anomaly scores across all 49 Bitcoin time steps.  \n"
        "Test period = ts 43–49 (live inference). Top-100 ranked anomalies shown below."
    )

    drift = load_drift()
    ts_data = pd.DataFrame(drift["per_time_step"])
    top100 = load_top100()

    # ── Time-step selector ────────────────────────────────────────────────────
    st.markdown("### Select Time Step")
    col_slider, col_info = st.columns([3, 1])
    with col_slider:
        ts = st.slider("Time Step", 1, 49, 43, help="ts 1-34 = train | 35-42 = val | 43-49 = test")
    row = ts_data[ts_data["time_step"] == ts].iloc[0]
    with col_info:
        split = "Train" if ts <= 34 else ("Val" if ts <= 42 else "Test")
        st.metric("Split", split)
        st.metric("Labeled nodes", int(row["n_labeled"]))
        st.metric("Illicit nodes", int(row["n_illicit"]))
        st.metric("Prevalence", f"{row['illicit_prevalence']*100:.1f}%")
        if not np.isnan(row["roc_auc"]):
            st.metric("AUC @ ts", f"{row['roc_auc']:.4f}")

    # ── Per-timestep line chart ───────────────────────────────────────────────
    st.markdown("### AUC & Illicit Prevalence Over Time")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add AUC
    fig.add_trace(
        go.Scatter(
            x=ts_data["time_step"], y=ts_data["roc_auc"],
            name="ROC-AUC", line=dict(color=C["green"], width=2.5),
            hovertemplate="ts=%{x}<br>AUC=%{y:.4f}",
        ),
        secondary_y=False,
    )
    # Add prevalence
    fig.add_trace(
        go.Scatter(
            x=ts_data["time_step"], y=ts_data["illicit_prevalence"],
            name="Illicit Prevalence", line=dict(color=C["red"], width=2, dash="dash"),
            hovertemplate="ts=%{x}<br>Prev=%{y:.3f}",
        ),
        secondary_y=True,
    )
    # Mark selected ts
    fig.add_vline(x=ts, line_dash="dot", line_color=C["ink"], opacity=0.45)
    # Period shading
    fig.add_vrect(x0=1, x1=34.5, fillcolor=C["blue"], opacity=0.06, line_width=0,
                  annotation_text="Train", annotation_position="top left",
                  annotation_font_color=C["muted"])
    fig.add_vrect(x0=34.5, x1=42.5, fillcolor=C["orange"], opacity=0.06, line_width=0,
                  annotation_text="Val", annotation_position="top left",
                  annotation_font_color=C["muted"])
    fig.add_vrect(x0=42.5, x1=49, fillcolor=C["red"], opacity=0.06, line_width=0,
                  annotation_text="Test", annotation_position="top left",
                  annotation_font_color=C["muted"])

    fig.update_yaxes(title_text="ROC-AUC", range=[0.55, 1.02], secondary_y=False)
    fig.update_yaxes(title_text="Illicit Prevalence", range=[0, 0.45], secondary_y=True)
    fig.update_xaxes(title_text="Time Step")
    fig.update_layout(height=350, template="fraud", legend=dict(orientation="h", y=1.12))
    show(fig)

    st.markdown("---")

    # ── Top anomalies for selected time step ──────────────────────────────────
    st.markdown(f"### Top Anomalies in Time Step {ts}")
    ts_anoms = top100[top100["time_step"] == ts].copy()
    if ts_anoms.empty:
        st.info(
            f"No top-100 anomalies fall in ts={ts}. "
            "The global top-100 are concentrated in ts 43-47 (test period). "
            "Try ts 43, 45, 46, or 47."
        )
    else:
        ts_anoms["label_display"] = ts_anoms["label_text"].map(
            {"illicit": "Illicit", "licit": "Licit", "unknown": "Unknown"}
        )
        display_cols = ["rank", "global_node_id", "anomaly_score", "label_display"]
        st.dataframe(
            ts_anoms[display_cols].rename(columns={
                "rank": "Global Rank",
                "global_node_id": "Node ID",
                "anomaly_score": "Anomaly Score",
                "label_display": "True Label",
            }),
            use_container_width=True, hide_index=True,
        )

    st.markdown("---")

    # ── Global top-100 overview ───────────────────────────────────────────────
    st.markdown("### Global Top-100 Anomalies by Time Step")
    label_counts = top100.groupby(["time_step", "label_text"]).size().reset_index(name="count")

    # Pivot to one column per label so we can stack in a deliberate order with the
    # rare labeled segments capped on top (otherwise the licit/illicit slivers are
    # buried under the dominant unknown count and become invisible).
    pivot = (label_counts
             .pivot(index="time_step", columns="label_text", values="count")
             .fillna(0))
    for lab in ("unknown", "licit", "illicit"):
        if lab not in pivot.columns:
            pivot[lab] = 0
    pivot = pivot.sort_index()
    ts_x = pivot.index.tolist()
    totals = pivot.sum(axis=1)
    grand_total = int(totals.sum())
    unknown_total = int(pivot["unknown"].sum())
    peak_ts = int(totals.idxmax())
    peak_val = int(totals.max())

    # unknown (bottom) → licit → illicit (top), so the scarce labeled nodes sit as
    # outlined caps at the top of each bar where they're actually legible.
    fig2 = go.Figure()
    for lab, disp in [("unknown", "Unknown"), ("licit", "Licit"), ("illicit", "Illicit")]:
        fig2.add_trace(go.Bar(
            x=ts_x, y=pivot[lab], name=disp,
            marker_color=LABEL_COLORS[lab],
            marker_line=dict(width=0.6, color=C["bg"]),
            hovertemplate=f"<b>{disp}</b>: %{{y:.0f}} of top-100<extra></extra>",
        ))
    fig2.update_layout(
        barmode="stack", template="fraud", height=320, bargap=0.22,
        hovermode="x unified",
        legend_title="Label",
        legend=dict(orientation="h", y=1.14, x=0),
        xaxis_title="Time Step", yaxis_title="# in Top-100",
    )
    fig2.update_xaxes(dtick=1)
    # Subtle callout on the dominant pattern.
    fig2.add_annotation(
        x=peak_ts, y=peak_val,
        text=f"Unknown-label nodes dominate<br>{unknown_total} / {grand_total} of the top-100",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=C["gray"],
        ax=44, ay=-40,
        align="left", font=dict(color=C["muted"], size=11),
        bgcolor="rgba(30,41,59,0.9)", bordercolor=C["border"], borderwidth=1, borderpad=6,
    )
    show(fig2)
    st.caption(
        "99% of the global top-100 are **unknown**-label nodes — Elliptic only labelled ~23% "
        "of nodes. High-scoring unknown nodes are operationally valid suspicious alerts. "
        "Hover any bar to see the exact licit / illicit / unknown split for that time step."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(
        "Compare all **9 models** across phases 2–6 on val and test metrics, "
        "inference latency, and parameter count."
    )

    ev = load_evaluation()
    lat = load_latency()

    # Build comparison dataframe
    rows = []
    for key in MODEL_ORDER:
        m = ev[key]
        lat_row = lat[lat["Model"] == m["display_name"]]
        lat_ms = lat_row["Full inference (ms)"].values[0] if not lat_row.empty else None
        params = lat_row["Params"].values[0] if not lat_row.empty else None
        rows.append({
            "key": key,
            "Model": m["display_name"],
            "Phase": m["phase"],
            "Val AUC": m["val"]["roc_auc"],
            "Test AUC": m["test"]["roc_auc"],
            "Val AP": m["val"]["avg_precision"],
            "Test AP": m["test"]["avg_precision"],
            "Val F1": m["val"]["f1"],
            "Test F1": m["test"]["f1"],
            "Latency (ms)": lat_ms,
            "Params": params,
        })
    df = pd.DataFrame(rows)

    # ── Metric toggle ─────────────────────────────────────────────────────────
    metric_choice = st.selectbox(
        "Primary metric", ["Test AUC", "Val AUC", "Test AP", "Val AP", "Val F1"]
    )

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig = px.bar(
        df, x="Model", y=metric_choice, color="Phase",
        color_discrete_map=PHASE_COLORS,
        text=metric_choice, template="fraud", height=400,
        labels={"Model": "", metric_choice: metric_choice},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                      textfont=dict(color=C["muted"], size=11))
    fig.update_yaxes(range=[0, df[metric_choice].max() * 1.18])
    fig.update_layout(showlegend=True, legend_title="Phase")
    show(fig)

    # ── Latency vs AUC scatter ────────────────────────────────────────────────
    st.markdown("### Latency vs Test AUC Trade-off")
    df_plot = df[df["Latency (ms)"].notna() & (df["Params"] != "N/A")].copy()
    df_plot["Params_num"] = pd.to_numeric(df_plot["Params"], errors="coerce")
    df_plot = df_plot[df_plot["Params_num"].notna()]
    fig2 = px.scatter(
        df_plot,
        x="Latency (ms)", y="Test AUC",
        color="Phase", size="Params_num",
        text="Model", color_discrete_map=PHASE_COLORS,
        template="fraud", height=400,
        labels={"Latency (ms)": "Full-graph Latency (ms)", "Test AUC": "Test ROC-AUC"},
        size_max=35,
    )
    fig2.update_traces(textposition="top center", textfont=dict(color=C["muted"], size=11))
    fig2.update_layout(legend_title="Phase")
    show(fig2)
    st.caption("Bubble size ∝ parameter count. GraphSAGE achieves the best AUC/latency trade-off.")

    # ── Detailed table ────────────────────────────────────────────────────────
    st.markdown("### Full Metrics Table")
    display_df = df[[
        "Model", "Phase", "Val AUC", "Test AUC",
        "Val AP", "Test AP", "Val F1", "Test F1",
        "Latency (ms)", "Params",
    ]].copy()
    for col in ["Val AUC", "Test AUC", "Val AP", "Test AP", "Val F1", "Test F1"]:
        display_df[col] = display_df[col].map("{:.4f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Radar chart — selected model deep-dive ────────────────────────────────
    st.markdown("### Model Deep-Dive (Radar)")
    selected = st.selectbox("Select model", [m["display_name"] for m in ev.values()])
    sel_key = next(k for k in MODEL_ORDER if ev[k]["display_name"] == selected)
    sel = ev[sel_key]

    categories = ["Val AUC", "Test AUC", "Val AP", "Test AP", "Val F1", "Test F1"]
    values = [
        sel["val"]["roc_auc"], sel["test"]["roc_auc"],
        sel["val"]["avg_precision"], sel["test"]["avg_precision"],
        sel["val"]["f1"], sel["test"]["f1"],
    ]
    accent = PHASE_COLORS.get(sel["phase"], C["blue"])
    fig3 = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=selected,
        line_color=accent,
        fillcolor="rgba(96,165,250,0.12)",
    ))
    fig3.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=C["line"],
                            tickfont=dict(color=C["muted"], size=10)),
            angularaxis=dict(gridcolor=C["line"], tickfont=dict(color=C["muted"])),
        ),
        template="fraud", height=400,
        title=f"{selected} ({sel['phase']})",
    )
    show(fig3)

    # val/test confusion matrices side by side
    col_v, col_t = st.columns(2)
    for col, split in [(col_v, "val"), (col_t, "test")]:
        cm = np.array(sel[split]["confusion_matrix"])
        labels_cm = ["Licit (pred)", "Illicit (pred)"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale=SEQ_BLUE,
            x=labels_cm, y=["Licit (true)", "Illicit (true)"],
            template="fraud", title=f"{split.capitalize()} Confusion Matrix",
        )
        fig_cm.update_layout(height=300, coloraxis_showscale=False)
        with col:
            show(fig_cm)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CONCEPT DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Concept Drift":
    st.title("Concept Drift Analysis")
    st.markdown(
        "The Elliptic dataset has a pronounced **temporal distribution shift**:  \n"
        "illicit prevalence drops 11.6% (train) → 9.2% (val) → 2.5% (test).  \n"
        "This page dissects how that shift impacts model performance over time."
    )

    drift = load_drift()
    ts_df = pd.DataFrame(drift["per_time_step"])

    # ── Period summary metrics ────────────────────────────────────────────────
    p = drift["period_mean_auc"]
    pv = drift["period_mean_prevalence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Train mean AUC", f"{p['train']:.4f}", help="ts 1-34")
    c2.metric("Val mean AUC", f"{p['val']:.4f}", f"Δ {p['val']-p['train']:+.4f}", help="ts 35-42")
    c3.metric("Test mean AUC", f"{p['test']:.4f}", f"Δ {p['test']-p['val']:+.4f}", help="ts 43-49")

    d1, d2, d3 = st.columns(3)
    d1.metric("Train prevalence", f"{pv['train']*100:.1f}%")
    d2.metric("Val prevalence", f"{pv['val']*100:.1f}%", f"Δ {(pv['val']-pv['train'])*100:+.1f}pp",
              delta_color="inverse")
    d3.metric("Test prevalence", f"{pv['test']*100:.1f}%", f"Δ {(pv['test']-pv['val'])*100:+.1f}pp",
              delta_color="inverse")

    st.markdown("---")

    # ── Per-step AUC chart ────────────────────────────────────────────────────
    st.markdown("### Per-Time-Step ROC-AUC (GraphSAGE)")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("ROC-AUC per Time Step", "Illicit Prevalence"),
                        vertical_spacing=0.12)

    colors = [C["blue"] if ts <= 34 else (C["orange"] if ts <= 42 else C["red"])
              for ts in ts_df["time_step"]]
    fig.add_trace(
        go.Bar(x=ts_df["time_step"], y=ts_df["roc_auc"],
               marker_color=colors, name="AUC",
               hovertemplate="ts=%{x}<br>AUC=%{y:.4f}"),
        row=1, col=1,
    )
    fig.add_hline(y=0.85, line_dash="dash", line_color=C["yellow"],
                  annotation_text="SLA floor (0.85)", annotation_font_color=C["yellow"],
                  row=1, col=1)

    fig.add_trace(
        go.Bar(x=ts_df["time_step"], y=ts_df["illicit_prevalence"],
               marker_color=colors, name="Prevalence",
               hovertemplate="ts=%{x}<br>Prevalence=%{y:.3f}"),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="AUC", row=1, col=1, range=[0.55, 1.05])
    fig.update_yaxes(title_text="Prevalence", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_layout(
        height=520, template="fraud", showlegend=False,
        annotations=fig.layout.annotations + (
            dict(x=17, y=1.03, xref="x", yref="y",
                 text="Train (ts 1-34)", showarrow=False, font=dict(color=C["blue"])),
            dict(x=38, y=1.03, xref="x", yref="y",
                 text="Val (35-42)", showarrow=False, font=dict(color=C["orange"])),
            dict(x=46, y=1.03, xref="x", yref="y",
                 text="Test (43-49)", showarrow=False, font=dict(color=C["red"])),
        ),
    )
    show(fig)

    st.markdown("---")

    # ── Retraining recommendation ─────────────────────────────────────────────
    st.markdown("### Drift Mitigation Recommendations")
    rec_cols = st.columns(3)
    recs = [
        ("Trigger-based Retrain",
         "Monitor AUC < 0.85 SLA or prevalence shift > 2×. "
         "Re-train on a rolling window including recent labeled data."),
        ("Threshold Recalibration",
         "Re-tune the decision threshold on the most recent labeled time steps "
         "without full retraining — cheap and effective for prevalence shifts."),
        ("Architecture Choice",
         "Under severe concept drift, prefer simple inductive models (GraphSAGE) "
         "over complex temporal architectures that overfit training dynamics."),
    ]
    for col, (title, body) in zip(rec_cols, recs):
        with col:
            info_card(title, body, accent=C["blue"])

    # ── Adversarial robustness quick summary ──────────────────────────────────
    st.markdown("### Adversarial Robustness Summary")
    rob_cols = st.columns(3)
    robustness = [
        ("Gaussian Noise", "AUC 0.777 → 0.716 at σ=1.0\n(mild degradation — robust)", C["green"]),
        ("Structural Slow-Bleed", "Recall stable with ≤10 licit edges added\n(structurally robust)", C["green"]),
        ("Feature Camouflage", "Illicit recall → 0 at α≥0.75\n(main attack surface)", C["red"]),
    ]
    for col, (title, body, accent) in zip(rob_cols, robustness):
        with col:
            info_card(title, body, accent=accent)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Interpretability":
    st.title("Interpretability & Explanations")
    st.markdown(
        "Understanding **why** GraphSAGE flags a transaction as illicit — "
        "via SHAP feature attribution, GAT attention weights, and neighborhood analysis."
    )

    shap_data = load_shap()
    attn_data = load_attention()

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    st.markdown("### SHAP Feature Attribution (KernelSHAP)")
    st.markdown(
        "**Method:** KernelSHAP over exact 2-hop receptive field. Graph fixed; "
        "only target node's feature row perturbed. Exact (not approximate) for "
        "Elliptic's low-degree graph (mean degree 2.3)."
    )

    # Top SHAP features from the stored values
    if "mean_abs_shap" in shap_data:
        top_features = sorted(
            shap_data["mean_abs_shap"].items(), key=lambda x: x[1], reverse=True
        )[:20]
        feat_names = [f[0] for f in top_features]
        feat_vals = [f[1] for f in top_features]
        fig_shap = px.bar(
            x=feat_vals[::-1], y=feat_names[::-1], orientation="h",
            labels={"x": "Mean |SHAP value|", "y": "Feature"},
            template="fraud", height=440,
            title="Top-20 Features by Mean |SHAP| (over top-20 anomalies)",
            color=feat_vals[::-1], color_continuous_scale=SEQ_BLUE,
        )
        fig_shap.update_layout(coloraxis_showscale=False)
        show(fig_shap)
    else:
        # Fall back to static image
        shap_img = RESULTS / "shap_analysis.png"
        if shap_img.exists():
            st.image(str(shap_img), caption="SHAP analysis (top-20 anomalies)")
        else:
            st.info("SHAP values file found but mean_abs_shap key not present.")

    st.markdown(
        "**Key finding:** The score is driven by the node's **own feature vector** "
        "(feat_76, feat_88, feat_148, feat_79, feat_142 are top drivers). "
        "Neighbor propagation is secondary — confirmed by the GAT self-loop attention below."
    )

    st.markdown("---")

    # ── GAT attention ─────────────────────────────────────────────────────────
    st.markdown("### GAT Attention Weights")
    col_l1, col_l2 = st.columns(2)
    svn_keys = {"Layer 1": "layer1_self_vs_neighbor", "Layer 2": "layer2_self_vs_neighbor"}
    if all(k in attn_data for k in svn_keys.values()):
        for col, (layer_name, svn_key) in zip([col_l1, col_l2], svn_keys.items()):
            svn = attn_data[svn_key]
            self_w = svn["self"]
            neighbor_w = svn["neighbor"]
            fig_attn = go.Figure(go.Pie(
                labels=["Self-loop", "Neighbors"],
                values=[self_w, neighbor_w],
                hole=0.5,
                marker_colors=[C["blue"], C["red"]],
                textinfo="label+percent",
                textfont=dict(color=C["ink"]),
            ))
            fig_attn.update_layout(
                template="fraud", height=300,
                title=f"GAT {layer_name} — Mean Attention Split",
                showlegend=True,
            )
            with col:
                show(fig_attn)
    else:
        attn_img = RESULTS / "attention_heatmaps.png"
        if attn_img.exists():
            st.image(str(attn_img), caption="GAT attention heatmaps (top-20 anomalies)")

    st.markdown(
        "**Key finding:** GAT self-loop attention dominates "
        "(L1 ≈70% self vs ≈26% neighbor). The model heavily relies on a node's own "
        "features in this sparse graph (median degree = 2)."
    )

    st.markdown("---")

    # ── Temporal evolution ────────────────────────────────────────────────────
    st.markdown("### Temporal Evolution of Risk Scores")
    try:
        tev = load_temporal_evolution()
        if "per_time_step" in tev:
            pts = tev["per_time_step"]
            ts_list = pts.get("time_step", [])
            mean_risk = pts.get("mean_risk", [])
            illicit_prev = pts.get("illicit_prevalence", [])
            if ts_list and mean_risk:
                fig_tev = make_subplots(specs=[[{"secondary_y": True}]])
                fig_tev.add_trace(
                    go.Scatter(x=ts_list, y=mean_risk,
                               name="Mean Risk Score", line=dict(color=C["violet"], width=2.5),
                               hovertemplate="ts=%{x}<br>Risk=%{y:.3f}"),
                    secondary_y=False,
                )
                if illicit_prev:
                    fig_tev.add_trace(
                        go.Scatter(x=ts_list, y=illicit_prev,
                                   name="Illicit Prevalence", line=dict(color=C["red"], width=2, dash="dash"),
                                   hovertemplate="ts=%{x}<br>Prev=%{y:.3f}"),
                        secondary_y=True,
                    )
                fig_tev.update_yaxes(title_text="Mean Risk Score", secondary_y=False)
                fig_tev.update_yaxes(title_text="Illicit Prevalence", secondary_y=True)
                fig_tev.update_xaxes(title_text="Time Step")
                fig_tev.update_layout(height=320, template="fraud",
                                      legend=dict(orientation="h", y=1.12))
                show(fig_tev)
                st.caption(
                    "Mean risk stays elevated in the test period even as labeled illicit prevalence "
                    "falls — the model detects feature-anomalous unknown transactions not covered by labels."
                )
    except Exception:
        tev_img = RESULTS / "temporal_evolution_anomalies.png"
        if tev_img.exists():
            st.image(str(tev_img), caption="Temporal evolution of risk scores and embedding drift")

    st.markdown("---")

    # ── Neighborhood insight ───────────────────────────────────────────────────
    st.markdown("### Neighborhood Analysis — Top Anomalies")
    st.markdown(
        "The global top-20 anomalies are **near-isolated leaf transactions** "
        "(degree ≈1, out-degree 0) whose sole neighbor is low-risk (mean 0.068) and non-illicit. "
        "Yet they score ~1.0 → the score is driven by the node's **own feature vector**, "
        "not neighbor propagation. This is independently confirmed by SHAP (above) and "
        "GAT self-loop attention (above)."
    )
    st.markdown(
        "> **Operational implication:** New/isolated transactions can be scored immediately "
        "with near-perfect accuracy (inductive inference — no graph context needed). "
        "Cold-start AUC drops only 0.777 → 0.680 with all edges removed (Phase 10)."
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SCALABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Scalability":
    st.title("Scalability Analysis")
    st.markdown(
        "GraphSAGE full-batch CPU inference scales **sub-linearly** "
        "(fixed overhead amortised at scale). Tested via disjoint k-tiling of the "
        "real Elliptic graph, preserving degree distribution and feature statistics."
    )

    sc = load_scalability()
    sc_df = pd.DataFrame(sc["measurements"])

    # ── Key metrics ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("1× (base)", f"{sc_df['latency_ms_mean'][0]:.0f} ms",
              f"{sc_df['n_nodes'][0]:,} nodes")
    c2.metric("5× scale", f"{sc_df['latency_ms_mean'][2]:.0f} ms",
              f"{sc_df['n_nodes'][2]:,} nodes")
    c3.metric("10× scale", f"{sc_df['latency_ms_mean'][3]:.0f} ms",
              f"{sc_df['n_nodes'][3]:,} nodes")

    st.markdown("---")

    # ── Latency vs nodes chart ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            sc_df, x="n_nodes", y="latency_ms_mean",
            error_y="latency_ms_std",
            labels={"n_nodes": "Nodes", "latency_ms_mean": "Latency (ms)"},
            template="fraud", title="Latency vs Graph Size",
        )
        fig.update_traces(marker=dict(color=C["blue"], size=10))
        # Add linear fit line
        x_fit = np.linspace(sc_df["n_nodes"].min(), sc_df["n_nodes"].max(), 100)
        b = sc["bottleneck"]["latency_vs_nodes"]
        y_fit = b["ms_per_node"] * x_fit + b["intercept_ms"]
        fig.add_trace(go.Scatter(
            x=x_fit, y=y_fit, mode="lines",
            line=dict(dash="dash", color=C["orange"]),
            name=f"Linear fit (R²={b['r2']:.4f})",
        ))
        fig.update_layout(height=340)
        show(fig)

    with col2:
        fig2 = px.bar(
            sc_df, x=sc_df["n_nodes"].astype(str), y="ms_per_1k_nodes",
            labels={"x": "Nodes", "ms_per_1k_nodes": "ms / 1k nodes"},
            template="fraud", title="Per-1k-Nodes Cost (sub-linear)",
            color="ms_per_1k_nodes", color_continuous_scale=SEQ_BLUE,
        )
        fig2.update_layout(height=340, coloraxis_showscale=False)
        show(fig2)

    # ── Memory footprint ──────────────────────────────────────────────────────
    st.markdown("### Memory Footprint")
    fig3 = px.bar(
        sc_df, x=sc_df["n_nodes"].astype(str), y="input_footprint_mb",
        labels={"x": "Nodes", "input_footprint_mb": "Input Tensor (MB)"},
        template="fraud", title="Feature Tensor Memory (N × 169 × 4 bytes)",
        color="input_footprint_mb", color_continuous_scale=SEQ_BLUE,
    )
    fig3.update_layout(height=300, coloraxis_showscale=False)
    show(fig3)
    st.caption(
        "Memory bottleneck = node feature tensor (linear in N). "
        "At 10× (2M nodes) → ~1.35 GB — fits comfortably in free Colab RAM. "
        "Next lever beyond ~10× = mini-batch neighbor sampling."
    )

    st.markdown("---")
    st.markdown("### Scaling Verdict")
    st.success(
        f"**Sub-linear scaling** — fixed overhead amortises at large scale. "
        f"Per-node cost drops from {sc_df['ms_per_1k_nodes'][0]:.2f} ms/1k "
        f"(1×) → {sc_df['ms_per_1k_nodes'][3]:.2f} ms/1k (10×). "
        f"Linear fit R² = {sc['bottleneck']['latency_vs_nodes']['r2']:.4f}."
    )

    # ── Model registry ────────────────────────────────────────────────────────
    st.markdown("### Model Registry (Phase 11.2)")
    try:
        with open(ROOT / "config" / "model_registry.json") as f:
            registry = json.load(f)
        reg_rows = []
        for info in registry.get("models", []):
            reg_rows.append({
                "Model": info.get("name", ""),
                "Phase": info.get("phase", ""),
                "Size (MB)": f"{info.get('size_mb', 0):.3f}",
                "Val AUC": f"{info.get('val', {}).get('roc_auc', ''):.4f}" if info.get("val") else "",
                "Test AUC": f"{info.get('test', {}).get('roc_auc', ''):.4f}" if info.get("test") else "",
                "Production": "Yes" if info.get("is_production") else "",
            })
        if reg_rows:
            st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.info("model_registry.json not found at config/model_registry.json")
