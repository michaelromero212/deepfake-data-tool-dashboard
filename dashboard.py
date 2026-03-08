"""
dashboard.py — Deepfake Data Tool Dashboard.

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
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Data Tool Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    .dashboard-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .dashboard-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1a202c;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .version-badge {
        display: inline-block;
        font-family: 'SF Mono', 'Fira Code', monospace;
        background: #ebf4ff;
        border: 1px solid #bee3f8;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #2b6cb0;
        margin-right: 8px;
    }
    .meta-info { font-size: 0.8rem; color: #a0aec0; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #2b6cb0; line-height: 1.2; }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; color: #718096;
        text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stat-title {
        font-size: 0.75rem; font-weight: 600; color: #a0aec0;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #2d3748; }
    .stat-sub { font-size: 0.78rem; color: #718096; margin-top: 2px; }

    .section-header {
        font-size: 1rem; font-weight: 600; color: #2d3748;
        border-left: 3px solid #3182ce;
        padding-left: 12px; margin: 1.5rem 0 0.25rem 0;
    }
    .section-desc {
        font-size: 0.84rem; color: #718096; line-height: 1.5;
        margin: 0.25rem 0 1rem 0;
    }

    .stDataFrame { border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }

    div[data-testid="stMetricValue"] { font-size: 1.8rem; color: #2b6cb0; }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .sidebar-title { font-size: 1.1rem; font-weight: 700; color: #1a202c; }
    .sidebar-section {
        font-size: 0.75rem; font-weight: 600; color: #a0aec0;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-top: 1.5rem; margin-bottom: 0.5rem;
    }

    .footer-bar {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 12px 20px; font-size: 0.82rem; color: #718096;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-top: 1.5rem;
    }
    .styled-divider { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme ───────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", color="#4a5568", size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=12),
    xaxis=dict(gridcolor="#f0f4f8", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f0f4f8", zerolinecolor="#e2e8f0", linecolor="#e2e8f0"),
)

LEGEND_BASE = dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font_size=11)
LEGEND_TOP = dict(**LEGEND_BASE, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)

LABEL_COLORS = {"real": "#48bb78", "synthetic": "#fc8181", "unknown": "#a0aec0"}
LABEL_COLORS_ALPHA = {
    "real": "rgba(72,187,120,0.15)",
    "synthetic": "rgba(252,129,129,0.15)",
    "unknown": "rgba(160,174,192,0.15)",
}


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

def score_distribution_chart(df: pl.DataFrame) -> go.Figure:
    """Histogram + KDE overlay per label with decision boundary."""
    fig = go.Figure()
    x_kde = np.linspace(0, 1, 300)

    for label, color in LABEL_COLORS.items():
        scores = df.filter(pl.col("label") == label)["detection_score"].drop_nulls().to_numpy()
        if len(scores) == 0:
            continue

        # Histogram bars
        fig.add_trace(go.Histogram(
            x=scores, xbins=dict(start=0, end=1, size=0.05),
            name=label.capitalize(), marker_color=color, opacity=0.55,
            histnorm="probability density", legendgroup=label,
            hovertemplate=f"<b>{label.capitalize()}</b><br>Score: %{{x:.2f}}<br>Density: %{{y:.3f}}<extra></extra>",
        ))

        # KDE curve — same legend group, no separate entry
        if len(scores) >= 3:
            kde = gaussian_kde(scores, bw_method="scott")
            fig.add_trace(go.Scatter(
                x=x_kde, y=kde(x_kde),
                mode="lines", name=label.capitalize(),
                legendgroup=label, showlegend=False,
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{label.capitalize()} KDE</b><br>Score: %{{x:.3f}}<br>Density: %{{y:.4f}}<extra></extra>",
            ))

    # Decision boundary
    fig.add_vline(x=0.5, line=dict(color="#3182ce", width=1.5, dash="dash"),
                  annotation_text="Decision boundary (0.5)",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#3182ce"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Detection Score  (0 = Real · 1 = Synthetic)",
        yaxis_title="Probability Density",
        barmode="overlay",
        height=320,
        legend=LEGEND_TOP,
    )
    return fig


def score_boxplot(df: pl.DataFrame) -> go.Figure:
    """Box plot of detection scores by label, with individual sample points."""
    fig = go.Figure()

    for label, color in LABEL_COLORS.items():
        scores = df.filter(pl.col("label") == label)["detection_score"].drop_nulls().to_numpy()
        if len(scores) == 0:
            continue

        fig.add_trace(go.Box(
            y=scores, name=label.capitalize(),
            marker_color=color,
            boxmean="sd",   # show mean ± std
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            line_width=2,
            fillcolor=LABEL_COLORS_ALPHA.get(label, "rgba(160,174,192,0.15)"),
            hovertemplate=(
                f"<b>{label.capitalize()}</b><br>"
                "Score: %{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Score Distribution by Label", font_size=13, x=0),
        yaxis_title="Detection Score",
        height=320,
        showlegend=False,
    )
    return fig


def media_label_chart(df: pl.DataFrame) -> go.Figure:
    """Stacked bar showing real/synthetic breakdown per media type."""
    counts = (
        df.group_by(["media_type", "label"])
        .agg(pl.count("sample_id").alias("n"))
        .sort("media_type")
    )
    media_types = counts["media_type"].unique().sort().to_list()
    fig = go.Figure()

    for label, color in LABEL_COLORS.items():
        vals = []
        for mt in media_types:
            row = counts.filter((pl.col("media_type") == mt) & (pl.col("label") == label))
            vals.append(row["n"][0] if len(row) > 0 else 0)

        fig.add_trace(go.Bar(
            name=label.capitalize(), x=media_types, y=vals,
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>{label.capitalize()}</b><br>%{{x}}: %{{y}} samples<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        yaxis_title="Sample Count",
        height=320,
        legend=LEGEND_TOP,
    )
    return fig


def validation_gauge(report: dict) -> go.Figure:
    """Gauge chart showing pass rate + donut breakdown."""
    pass_rate = report["pass_rate"] * 100
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "pie"}]],
        column_widths=[0.5, 0.5],
    )

    # Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pass_rate,
        number=dict(suffix="%", font=dict(size=28, color="#2b6cb0")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#718096"),
            bar=dict(color="#3182ce"),
            bgcolor="#f0f4f8",
            steps=[
                dict(range=[0, 60], color="#fed7d7"),
                dict(range=[60, 85], color="#fefcbf"),
                dict(range=[85, 100], color="#c6f6d5"),
            ],
            threshold=dict(line=dict(color="#e53e3e", width=3), value=60),
        ),
        title=dict(text="Pass Rate", font=dict(size=12, color="#718096")),
    ), row=1, col=1)

    # Donut
    labels = ["Passed", "Warned", "Failed"]
    values = [report["passed"], report["warned"], report["failed"]]
    colors = ["#48bb78", "#ecc94b", "#fc8181"]
    non_zero = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if non_zero:
        l_, v_, c_ = zip(*non_zero)
        fig.add_trace(go.Pie(
            labels=l_, values=v_, marker_colors=c_,
            hole=0.55, textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value} samples (%{percent})<extra></extra>",
        ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#4a5568", size=11),
        margin=dict(l=16, r=16, t=36, b=16),
        height=260,
        showlegend=False,
        title=dict(text="Validation Results", font_size=13, x=0),
    )
    return fig


def inference_scatter(df: pl.DataFrame) -> go.Figure:
    """Scatter of detection score vs inference time, colored by label."""
    scored = df.filter(pl.col("detection_score").is_not_null() & pl.col("inference_ms").is_not_null())
    fig = go.Figure()

    for label, color in LABEL_COLORS.items():
        sub = scored.filter(pl.col("label") == label)
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["detection_score"].to_list(),
            y=sub["inference_ms"].to_list(),
            mode="markers",
            name=label.capitalize(),
            marker=dict(color=color, size=9, opacity=0.8, line=dict(width=1, color="white")),
            text=sub["file_name"].to_list(),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Score: %{x:.4f}<br>"
                "Inference: %{y:.1f}ms<extra></extra>"
            ),
        ))

    fig.add_vline(x=0.5, line=dict(color="#3182ce", width=1.5, dash="dash"),
                  annotation_text="0.5 threshold", annotation_font=dict(size=10, color="#3182ce"))

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "yaxis"},
        xaxis_title="Detection Score",
        yaxis=dict(
            title="Inference Time (ms, log scale)",
            type="log",
            gridcolor="#f0f4f8", zerolinecolor="#e2e8f0", linecolor="#e2e8f0",
        ),
        height=320,
        legend=LEGEND_TOP,
    )
    return fig


# ── Stats helpers ──────────────────────────────────────────────────────────

def score_stats(df: pl.DataFrame) -> dict[str, dict]:
    """Per-label descriptive statistics for detection score."""
    out = {}
    for label in ["real", "synthetic"]:
        scores = df.filter(pl.col("label") == label)["detection_score"].drop_nulls()
        if len(scores) == 0:
            continue
        arr = scores.to_numpy()
        out[label] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def separation_score(df: pl.DataFrame) -> float | None:
    """
    Mean score separation: difference between mean synthetic score
    and mean real score. Ranges 0–1; higher = better model separation.
    """
    stats = score_stats(df)
    if "real" not in stats or "synthetic" not in stats:
        return None
    return stats["synthetic"]["mean"] - stats["real"]["mean"]


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-section">Data Sources</div>', unsafe_allow_html=True)
    manifest_path = st.text_input("Manifest path", value="outputs/dataset_manifest.json")
    report_path = st.text_input("Validation report path", value="outputs/validation_report.json")

    st.markdown('<div class="sidebar-section">Filters</div>', unsafe_allow_html=True)

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


# ── Header ─────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="dashboard-header">
    <div class="dashboard-title">🛡️ Deepfake Data Tool Dashboard</div>
    <div style="margin-top:0.4rem;">
        <span class="version-badge">Dataset v{raw["dataset_version"]}</span>
        <span class="meta-info">Pipeline v{raw["pipeline_version"]} &middot; {raw["created_at"][:10]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────
label_dist = raw["label_distribution"]
report = load_report(report_path)
sep = separation_score(df_full)

pass_rate_val = f"{report['pass_rate']:.0%}" if report else "—"

metrics = [
    ("Total Samples", f"{raw['total_samples']:,}"),
    ("Real", f"{label_dist.get('real', 0):,}"),
    ("Synthetic", f"{label_dist.get('synthetic', 0):,}"),
    ("Validation Pass Rate", pass_rate_val),
]

cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

# ── Charts ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Detection Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">Score distributions by label with KDE curves, '
    'media type composition, and per-sample score vs. inference time.</div>',
    unsafe_allow_html=True,
)

_title_style = "text-align:center;font-size:0.88rem;font-weight:600;color:#2d3748;margin:0 0 0.25rem 0;"

col_dist, col_media, col_scatter = st.columns([3, 2, 3])
with col_dist:
    st.plotly_chart(score_distribution_chart(df_full), width="stretch", config={"displayModeBar": False})
    st.markdown(f'<p style="{_title_style}">Detection Score Distribution</p>', unsafe_allow_html=True)
    st.caption("Probability density of detection scores per label. Real samples cluster near 0, synthetic near 1. The KDE curve smooths the shape of each distribution.")
with col_media:
    st.plotly_chart(media_label_chart(df_full), width="stretch", config={"displayModeBar": False})
    st.markdown(f'<p style="{_title_style}">Samples by Media Type & Label</p>', unsafe_allow_html=True)
    st.caption("Number of real and synthetic samples per media type. Shows whether the dataset is balanced across image, video, and audio.")
with col_scatter:
    st.plotly_chart(inference_scatter(df_full), width="stretch", config={"displayModeBar": False})
    st.markdown(f'<p style="{_title_style}">Score vs. Inference Time</p>', unsafe_allow_html=True)
    st.caption("Detection score vs. model inference time per sample (log scale). Samples to the right of the threshold line are classified as synthetic.")

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

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
st.markdown(
    f'<div class="section-header">Sample Browser &nbsp; '
    f'<span style="font-weight:400;color:#a0aec0;font-size:0.85rem">{len(df)} samples</span></div>',
    unsafe_allow_html=True,
)
st.caption("Each row is a single sample. Detection Score shows model confidence (0 = real, 1 = synthetic). Use the sidebar filters to narrow by label, media type, or validation status.")

display_cols = ["sample_id", "file_name", "media_type", "label", "label_source",
                "detection_score", "validation_status", "file_size_kb", "inference_ms"]

st.dataframe(
    df.select([c for c in display_cols if c in df.columns]).to_pandas(),
    width="stretch",
    hide_index=True,
    column_config={
        "detection_score": st.column_config.ProgressColumn(
            "Detection Score", min_value=0.0, max_value=1.0, format="%.4f",
        ),
        "inference_ms": st.column_config.NumberColumn("Inference (ms)", format="%.1f"),
        "file_size_kb": st.column_config.NumberColumn("Size (KB)", format="%.1f"),
    },
)

# ── Validation issues ──────────────────────────────────────────────────────
if report and report.get("issues_by_type"):
    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Validation Issues</div>', unsafe_allow_html=True)
    st.caption(
        "Samples that failed one or more validation checks during the pipeline run. "
        "**label_score_mismatch** means the model's detection score contradicts the ground-truth label — "
        "e.g. a real sample scored above 0.5, or a synthetic sample scored below 0.5. "
        "These samples may indicate model uncertainty, mislabelled data, or edge cases worth reviewing."
    )
    issue_df = pl.DataFrame({
        "Issue Type": list(report["issues_by_type"].keys()),
        "Count": list(report["issues_by_type"].values()),
    }).sort("Count", descending=True)
    st.dataframe(issue_df.to_pandas(), width="stretch", hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────
model_names = df_full["model_name"].unique().to_list()
avg_inference = df_full["inference_ms"].mean()
footer_parts = [f"<strong>Detector:</strong> {' · '.join(model_names)}"]
if avg_inference:
    footer_parts.append(f"<strong>Avg Inference:</strong> {avg_inference:.1f}ms")
if sep is not None:
    footer_parts.append(f"<strong>Score Separation (Δ mean):</strong> {sep:.3f}")
st.markdown(
    f'<div class="footer-bar">'
    f'{"&nbsp;&nbsp;|&nbsp;&nbsp;".join(footer_parts)}'
    f'<span style="float:right;color:#cbd5e0;font-size:0.78rem">'
    f'Score Separation = mean(synthetic) − mean(real). Higher is better. &nbsp;·&nbsp; '
    f'Avg Inference includes all media types.'
    f'</span></div>',
    unsafe_allow_html=True,
)
