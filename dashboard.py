"""
dashboard.py — Streamlit visualization dashboard for deepfake-data-forge.

Reads dataset_manifest.json and validation_report.json from the outputs/
directory and renders an interactive dataset explorer.

Run:
  streamlit run dashboard.py
  streamlit run dashboard.py -- --manifest outputs/dataset_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="deepfake-data-forge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2e5bba33;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4d8af0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .version-badge {
        font-family: monospace;
        background: #1a1a2e;
        border: 1px solid #2e5bba55;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        color: #4d8af0;
    }
    .stDataFrame { border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #4d8af0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────

@st.cache_data
def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_report(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def manifest_to_df(manifest: dict) -> pl.DataFrame:
    rows = []
    for s in manifest["samples"]:
        detection = s.get("detection_result") or {}
        meta = s.get("metadata", {})
        rows.append({
            "sample_id": s["sample_id"],
            "file_name": meta.get("file_name", ""),
            "media_type": s["media_type"],
            "label": s["label"],
            "label_source": s["label_source"],
            "detection_score": detection.get("detection_score"),
            "model_name": detection.get("model_name", "none"),
            "inference_ms": detection.get("inference_time_ms"),
            "validation_status": s["validation_status"],
            "file_size_kb": round(meta.get("file_size_bytes", 0) / 1024, 1),
            "processed_path": s.get("processed_path", ""),
        })
    return pl.DataFrame(rows)


# ── Chart helpers ──────────────────────────────────────────────────────────

def score_histogram(df: pl.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    real = df.filter(pl.col("label") == "real")["detection_score"].drop_nulls().to_list()
    synthetic = df.filter(pl.col("label") == "synthetic")["detection_score"].drop_nulls().to_list()
    unknown = df.filter(pl.col("label") == "unknown")["detection_score"].drop_nulls().to_list()

    bins = np.linspace(0, 1, 21)

    if real:
        ax.hist(real, bins=bins, alpha=0.75, color="#2ecc71", label="Real", edgecolor="none")
    if synthetic:
        ax.hist(synthetic, bins=bins, alpha=0.75, color="#e74c3c", label="Synthetic", edgecolor="none")
    if unknown:
        ax.hist(unknown, bins=bins, alpha=0.5, color="#95a5a6", label="Unknown", edgecolor="none")

    ax.axvline(0.5, color="#4d8af0", linestyle="--", linewidth=1.5, alpha=0.8, label="Decision boundary")
    ax.set_xlabel("Detection Score  (0 = Real · 1 = Synthetic)", color="#888", fontsize=10)
    ax.set_ylabel("Sample Count", color="#888", fontsize=10)
    ax.tick_params(colors="#666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc", fontsize=9)
    fig.tight_layout()
    return fig


def validation_donut(report: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    sizes = [report["passed"], report["warned"], report["failed"]]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Pass", "Warn", "Fail"]

    valid = [(s, c, l) for s, c, l in zip(sizes, colors, labels) if s > 0]
    if valid:
        s, c, l = zip(*valid)
        wedges, _ = ax.pie(s, colors=c, startangle=90,
                           wedgeprops=dict(width=0.5, edgecolor="#0f1117", linewidth=2))
        ax.legend(
            [mpatches.Patch(color=col, label=f"{lbl}: {cnt}") for col, lbl, cnt in zip(c, l, s)],
            loc="lower center", facecolor="#1a1a2e", edgecolor="#333",
            labelcolor="#ccc", fontsize=9, ncol=3, bbox_to_anchor=(0.5, -0.12)
        )

    ax.text(0, 0, f"{report['pass_rate']:.0%}", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#4d8af0")
    ax.set_title("Validation", color="#ccc", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def media_bar(df: pl.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    counts = df.group_by("media_type").agg(pl.count("sample_id").alias("n")).sort("media_type")
    media_types = counts["media_type"].to_list()
    values = counts["n"].to_list()
    colors = {"image": "#4d8af0", "video": "#9b59b6", "audio": "#1abc9c"}
    bar_colors = [colors.get(m, "#888") for m in media_types]

    bars = ax.bar(media_types, values, color=bar_colors, edgecolor="none", width=0.5)
    ax.bar_label(bars, padding=4, color="#ccc", fontsize=10)
    ax.set_ylabel("Samples", color="#888", fontsize=9)
    ax.set_title("By Media Type", color="#ccc", fontsize=11)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_facecolor("#0f1117")
    fig.tight_layout()
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 deepfake-data-forge")
    st.markdown("---")

    manifest_path = st.text_input("Manifest path", value="outputs/dataset_manifest.json")
    report_path = st.text_input("Validation report path", value="outputs/validation_report.json")

    st.markdown("---")
    st.markdown("**Filters**")

    if Path(manifest_path).exists():
        raw = load_manifest(manifest_path)
        df_full = manifest_to_df(raw)

        label_opts = ["all"] + df_full["label"].unique().sort().to_list()
        media_opts = ["all"] + df_full["media_type"].unique().sort().to_list()
        status_opts = ["all"] + df_full["validation_status"].unique().sort().to_list()

        label_filter = st.selectbox("Label", label_opts)
        media_filter = st.selectbox("Media type", media_opts)
        status_filter = st.selectbox("Validation status", status_opts)
        score_range = st.slider("Detection score range", 0.0, 1.0, (0.0, 1.0), 0.01)
    else:
        st.warning("No manifest found. Run the pipeline first.")
        st.stop()


# ── Main content ───────────────────────────────────────────────────────────

st.markdown("# 🔬 deepfake-data-forge")
st.markdown(
    f'<span class="version-badge">dataset v{raw["dataset_version"]}</span>'
    f'&nbsp;&nbsp;<span style="color:#666;font-size:0.85rem">pipeline v{raw["pipeline_version"]} · {raw["created_at"][:10]}</span>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Top metrics ────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)

label_dist = raw["label_distribution"]
media_dist = raw["media_type_distribution"]

m1.metric("Total Samples", raw["total_samples"])
m2.metric("Real", label_dist.get("real", 0))
m3.metric("Synthetic", label_dist.get("synthetic", 0))
m4.metric("Images", media_dist.get("image", 0))
m5.metric("Video + Audio", media_dist.get("video", 0) + media_dist.get("audio", 0))

st.markdown("---")

# ── Charts row ─────────────────────────────────────────────────────────────
report = load_report(report_path)

col_hist, col_donut, col_bar = st.columns([5, 2.5, 2.5])

with col_hist:
    st.markdown("#### Detection Score Distribution")
    fig = score_histogram(df_full)
    st.pyplot(fig)
    plt.close(fig)

if report:
    with col_donut:
        fig2 = validation_donut(report)
        st.pyplot(fig2)
        plt.close(fig2)

    with col_bar:
        fig3 = media_bar(df_full)
        st.pyplot(fig3)
        plt.close(fig3)

st.markdown("---")

# ── Apply filters ──────────────────────────────────────────────────────────
df = df_full
if label_filter != "all":
    df = df.filter(pl.col("label") == label_filter)
if media_filter != "all":
    df = df.filter(pl.col("media_type") == media_filter)
if status_filter != "all":
    df = df.filter(pl.col("validation_status") == status_filter)
df = df.filter(
    (pl.col("detection_score") >= score_range[0]) &
    (pl.col("detection_score") <= score_range[1])
)

# ── Sample table ───────────────────────────────────────────────────────────
st.markdown(f"#### Sample Browser &nbsp; <span style='color:#666;font-size:0.9rem'>{len(df)} samples</span>", unsafe_allow_html=True)

display_cols = ["sample_id", "file_name", "media_type", "label", "label_source",
                "detection_score", "validation_status", "file_size_kb", "inference_ms"]

st.dataframe(
    df.select([c for c in display_cols if c in df.columns]).to_pandas(),
    use_container_width=True,
    hide_index=True,
    column_config={
        "detection_score": st.column_config.ProgressColumn(
            "Detection Score",
            min_value=0.0,
            max_value=1.0,
            format="%.3f",
        ),
        "inference_ms": st.column_config.NumberColumn("Inference (ms)", format="%.1f"),
        "file_size_kb": st.column_config.NumberColumn("Size (KB)", format="%.1f"),
    }
)

# ── Validation issues ──────────────────────────────────────────────────────
if report and report.get("issues_by_type"):
    st.markdown("---")
    st.markdown("#### Validation Issues")
    issue_df = pl.DataFrame({
        "Issue Type": list(report["issues_by_type"].keys()),
        "Count": list(report["issues_by_type"].values()),
    }).sort("Count", descending=True)
    st.dataframe(issue_df.to_pandas(), use_container_width=True, hide_index=True)

# ── Model info ─────────────────────────────────────────────────────────────
st.markdown("---")
model_names = df_full["model_name"].unique().to_list()
st.markdown(
    f"**Detector:** `{'  ·  '.join(model_names)}`  "
    f"&nbsp;|&nbsp; Avg inference: `{df_full['inference_ms'].mean():.1f}ms`"
    if df_full["inference_ms"].mean() else ""
)
