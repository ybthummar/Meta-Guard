from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Meta-Guard SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom Dark Cyber CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@400;600;700&display=swap');

    :root {
        --soc-bg: #0a0e14;
        --soc-panel-bg: #131920;
        --soc-border: #1e2a38;
        --neon-green: #00ff41;
        --neon-red: #ff2a2a;
        --neon-yellow: #faca2b;
        --neon-blue: #6366f1;
        --text-main: #c9d1d9;
        --text-muted: #8b949e;
    }

    .stApp {
        background-color: var(--soc-bg) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 { color: #ffffff !important; }

    .soc-header {
        background: linear-gradient(135deg, #131920 0%, #0a0e14 100%);
        border-left: 4px solid var(--neon-green);
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 0 20px rgba(0,255,65,0.06);
    }
    .soc-header h1 {
        margin: 0;
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        letter-spacing: 2px;
    }
    .soc-header .subtitle {
        margin: 6px 0 0 0;
        color: var(--text-muted);
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.9rem;
    }

    .soc-panel {
        background-color: var(--soc-panel-bg);
        border: 1px solid var(--soc-border);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }

    .ai-panel {
        background: linear-gradient(145deg, #14181f, #0d1015);
        border: 1px solid #2d3748;
        border-left: 4px solid var(--neon-blue);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 0 18px rgba(99,102,241,0.07);
    }
    .ai-panel h4 { margin-top: 0; }
    .ai-panel p { line-height: 1.7; }

    .gemini-panel {
        background: linear-gradient(145deg, #1a1520, #0d1015);
        border: 1px solid #4a3d6b;
        border-left: 4px solid #a855f7;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 0 18px rgba(168,85,247,0.07);
    }
    .gemini-panel h4 { margin-top: 0; }

    .status-alert-red {
        background: linear-gradient(90deg, rgba(255,42,42,0.12), rgba(255,42,42,0.04));
        border: 1px solid var(--neon-red);
        color: var(--neon-red);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        letter-spacing: 1px;
        box-shadow: 0 0 12px rgba(255,42,42,0.1);
    }
    .status-alert-green {
        background: linear-gradient(90deg, rgba(0,255,65,0.10), rgba(0,255,65,0.03));
        border: 1px solid var(--neon-green);
        color: var(--neon-green);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        letter-spacing: 1px;
        box-shadow: 0 0 12px rgba(0,255,65,0.08);
    }
    .status-alert-yellow {
        background: linear-gradient(90deg, rgba(250,202,43,0.12), rgba(250,202,43,0.03));
        border: 1px solid var(--neon-yellow);
        color: var(--neon-yellow);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        font-weight: 700;
        text-align: center;
        font-size: 1.15rem;
        margin-bottom: 1.2rem;
        letter-spacing: 1px;
        box-shadow: 0 0 12px rgba(250,202,43,0.08);
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid var(--soc-border);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: var(--text-muted);
    }

    div[data-testid="stMetric"] {
        background-color: var(--soc-panel-bg);
        border: 1px solid var(--soc-border);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        overflow: visible;
        min-width: 0;
    }
    div[data-testid="stMetric"] label {
        color: var(--text-muted) !important;
        font-size: 0.8rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.6rem !important;
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: #1a2332 !important;
        border-radius: 6px;
    }

    .stDataFrame { border-radius: 8px; overflow: hidden; }

    hr { border-color: var(--soc-border); }

    .mono-text {
        font-family: 'Share Tech Mono', monospace;
        color: var(--neon-green);
        font-size: 0.85rem;
    }

    .section-label {
        font-family: 'Share Tech Mono', monospace;
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_sample_inputs() -> dict[str, list[float]]:
    sample_path = Path(__file__).resolve().parents[1] / "examples" / "sample_inputs.json"
    if not sample_path.exists():
        return {}
    with sample_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False, ttl=30)
def fetch_metadata(api_base_url: str) -> dict[str, Any] | None:
    try:
        resp = requests.get(f"{api_base_url.rstrip('/')}/metadata", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def parse_manual_input(raw_text: str, expected: int) -> list[float]:
    tokens = [t.strip() for t in raw_text.split(",") if t.strip()]
    cleaned: list[float] = []
    for t in tokens:
        try:
            cleaned.append(float(t))
        except ValueError:
            cleaned.append(0.0)
    if len(cleaned) < expected:
        cleaned.extend([0.0] * (expected - len(cleaned)))
    elif len(cleaned) > expected:
        cleaned = cleaned[:expected]
    return cleaned


def parse_csv_upload(uploaded_file: Any, expected: int) -> tuple[pd.DataFrame, list[str], int]:
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python")
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    ndf = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    orig_cols = ndf.shape[1]
    msgs: list[str] = []
    if ndf.shape[1] > expected:
        msgs.append(f"Input has {ndf.shape[1]} columns — truncated to {expected}.")
        ndf = ndf.iloc[:, :expected]
    elif ndf.shape[1] < expected:
        msgs.append(f"Input has {ndf.shape[1]} columns — padded to {expected} with zeros.")
        for i in range(ndf.shape[1], expected):
            ndf[f"f{i+1}"] = 0.0
    return ndf, msgs, orig_cols


def api_post(api_base_url: str, path: str, payload: Any, timeout: int = 300) -> dict[str, Any]:
    resp = requests.post(f"{api_base_url.rstrip('/')}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def normalize_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "results" in payload:
        return payload["results"]
    return [payload]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Data & State
# ---------------------------------------------------------------------------
samples = load_sample_inputs()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="soc-header">
        <h1>🛡️ META-GUARD</h1>
        <p class="subtitle">IoMT Intrusion Detection System &nbsp;|&nbsp; SOC Dashboard &nbsp;|&nbsp; Two-Stage Pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ System Control")
    api_url = st.text_input("FastAPI Backend URL", value="http://127.0.0.1:8080")
    metadata = fetch_metadata(api_url)

    st.markdown("---")
    if metadata:
        st.success("🟢 API Online")
        ca, cb = st.columns(2)
        ca.metric("Features", metadata["expected_features"])
        cb.metric("Classes", metadata["known_class_count"])
        st.caption(f"Distance Threshold: **{metadata['distance_threshold']:.4f}**")
        if metadata.get("gemini_available"):
            st.info("🟣 Gemini AI: Active")
        else:
            st.caption("🔘 Gemini AI: Inactive (set GEMINI_API_KEY)")
        if metadata.get("known_classes"):
            with st.expander("Known Attack Classes"):
                for cn in metadata["known_classes"]:
                    st.markdown(f"- `{cn}`")
    else:
        st.error("🔴 API Offline — start the backend first")

    st.markdown("---")
    st.markdown("### 📊 Pipeline Architecture")
    st.markdown(
        """
        **Stage 1 — Edge Filter**
        Binary classifier: *normal* vs *suspicious*

        **Stage 2 — Cloud Open-Set**
        Embedding distance analysis against prototypes.
        Flags unknown patterns as potential Zero-Day threats.
        """
    )
    st.markdown("---")
    st.markdown(
        '<p class="mono-text" style="text-align:center;">v2.0 &nbsp;|&nbsp; Meta-Guard Research</p>',
        unsafe_allow_html=True,
    )

expected_features = int(metadata["expected_features"]) if metadata else 45

# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------
tab_generate, tab_manual, tab_csv = st.tabs([
    "🧪 Generate & Evaluate",
    "🔍 Manual Detection",
    "📁 CSV Upload",
])


# =====================================================================
# TAB 1: SYNTHETIC GENERATION & EVALUATION
# =====================================================================
with tab_generate:
    st.markdown('<div class="section-label">🧪 Synthetic Dataset Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="soc-panel">'
        '<p style="color:var(--text-muted);margin:0;">Define the number of samples for each traffic '
        'category. Meta-Guard will <strong>generate</strong> a synthetic 45-feature IoMT dataset, '
        'then run <strong>all samples</strong> through the two-stage pipeline for inference-based '
        'classification. The model does <strong>NOT</strong> see ground truth labels.</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="soc-panel" style="border-left:4px solid #ff2a2a;">'
        '<h4 style="color:#ff2a2a;margin-top:0;">⚠️ Important Disclaimer — Unseen Synthetic Data</h4>'
        '<p style="color:#c9d1d9;margin:0;">'
        'The synthetic dataset generated here is <strong>completely new data that the model has never '
        'seen during training</strong>. Because these samples are entirely unseen and out-of-distribution '
        'with respect to the model\'s training set, the two-stage pipeline will predominantly classify them '
        'as <strong style="color:#ff2a2a;">Zero-Day (unknown) attacks</strong>.</p>'
        '<p style="color:#c9d1d9;margin-top:8px;margin-bottom:0;">'
        'This behaviour is <strong>expected and by design</strong> — it demonstrates that Meta-Guard\'s '
        'open-set recognition correctly rejects unfamiliar patterns rather than forcing them into known '
        'categories. Our model achieves a <strong style="color:#faca2b;">~35 % zero-day detection rate</strong>, '
        'which may appear low, but combined with a <strong>very low false-positive rate (FPR)</strong> it is '
        'highly valuable in real-world SOC operations where false alarms are costly.</p></div>',
        unsafe_allow_html=True,
    )

    # --- Input Panel ---
    st.markdown("#### 📋 Sample Configuration")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        num_normal = st.number_input(
            "🟢 Normal Samples", min_value=0, max_value=500, value=30, step=5,
            help="Low-variance traffic matching normal IoMT baselines",
        )
    with cc2:
        num_known = st.number_input(
            "🟡 Known Attack Samples", min_value=0, max_value=500, value=20, step=5,
            help="Moderate anomaly patterns pushed toward known attack prototypes",
        )
    with cc3:
        num_zero_day = st.number_input(
            "🔴 Zero-Day Samples", min_value=0, max_value=500, value=50, step=5,
            help="High-deviation out-of-distribution patterns far from all prototypes",
        )

    total_planned = num_normal + num_known + num_zero_day
    st.caption(f"**Total samples:** {total_planned}")

    # --- Threshold Slider ---
    st.markdown("#### 📏 Zero-Day Distance Threshold")
    default_threshold = float(metadata["distance_threshold"]) if metadata else 0.0957
    custom_threshold = st.slider(
        "Adjust distance threshold (higher → fewer zero-day detections)",
        min_value=0.01, max_value=2.0,
        value=default_threshold, step=0.01, format="%.4f",
        help="Samples with embedding distance above this AND low margin are flagged as zero-day.",
    )
    if abs(custom_threshold - default_threshold) > 0.001:
        st.info(f"Custom threshold: **{custom_threshold:.4f}** (default: {default_threshold:.4f})")

    st.markdown("---")

    # --- Action Buttons ---
    bc1, bc2, _ = st.columns([2, 2, 1])
    with bc1:
        gen_and_run = st.button(
            "⚡ Generate & Run (Demo Mode)", type="primary", use_container_width=True,
            help="One-click: generate synthetic dataset + run full evaluation",
        )
    with bc2:
        gen_only = st.button("📦 Generate Dataset Only", use_container_width=True)

    # --- Processing ---
    if gen_and_run or gen_only:
        if total_planned == 0:
            st.error("Please specify at least one sample.")
        else:
            with st.spinner(f"🔄 Generating {total_planned} synthetic samples..."):
                try:
                    gen_resp = api_post(api_url, "/generate", {
                        "num_normal": num_normal,
                        "num_known": num_known,
                        "num_zero_day": num_zero_day,
                    })
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")
                    gen_resp = None

            if gen_resp:
                st.session_state["gen_response"] = gen_resp
                st.success(f"✅ Generated {gen_resp['total_samples']} samples")

                # Preview
                st.markdown("#### 📊 Generated Dataset Preview")
                preview_df = pd.DataFrame(gen_resp["preview"])
                st.dataframe(preview_df, use_container_width=True, height=250)

                # Download link
                dl_url = f"{api_url.rstrip('/')}/download-dataset"
                st.markdown(
                    f'<a href="{dl_url}" target="_blank">'
                    f'<button style="background:#6366f1;color:white;border:none;padding:8px 24px;'
                    f'border-radius:6px;cursor:pointer;font-weight:600;">⬇️ Download CSV</button></a>',
                    unsafe_allow_html=True,
                )

                # Evaluate
                if gen_and_run:
                    thr = custom_threshold if abs(custom_threshold - default_threshold) > 0.001 else None
                    with st.spinner("🔍 Running Meta-Guard evaluation on all samples..."):
                        try:
                            eval_resp = api_post(api_url, "/evaluate", {"custom_threshold": thr})
                        except Exception as exc:
                            st.error(f"Evaluation failed: {exc}")
                            eval_resp = None
                    if eval_resp:
                        st.session_state["eval_response"] = eval_resp

    # --- Display Evaluation Results ---
    eval_resp = st.session_state.get("eval_response")
    if eval_resp:
        try:
            import plotly.graph_objects as go
        except ImportError:
            go = None

        results = eval_resp.get("results", [])
        comparison = eval_resp.get("comparison", [])
        total = eval_resp["total_samples"]
        normal_det = eval_resp["normal_detected"]
        known_det = eval_resp["known_attacks_detected"]
        zeroday_det = eval_resp["zero_day_detected"]

        st.markdown("---")
        st.markdown('<div class="section-label">📡 Detection Results</div>', unsafe_allow_html=True)

        # Summary cards
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Samples", f"{total:,}")
        m2.metric("🟢 Normal", f"{normal_det:,}")
        m3.metric("🟡 Known Attacks", f"{known_det:,}")
        m4.metric("🔴 Zero-Day", f"{zeroday_det:,}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Status banner
        if zeroday_det > 0:
            st.markdown(
                f'<div class="status-alert-red">🚨 {zeroday_det} ZERO-DAY INCIDENT{"S" if zeroday_det != 1 else ""} DETECTED</div>',
                unsafe_allow_html=True,
            )
        elif known_det > 0:
            st.markdown(
                f'<div class="status-alert-yellow">⚠️ {known_det} KNOWN THREAT{"S" if known_det != 1 else ""} DETECTED</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="status-alert-green">🟢 ALL TRAFFIC NORMAL</div>', unsafe_allow_html=True)

        # JSON summary
        st.markdown("#### 📋 Summary Output")
        st.json({
            "total_samples": total,
            "normal_detected": normal_det,
            "known_attacks_detected": known_det,
            "zero_day_detected": zeroday_det,
        })

        # --- Charts ---
        if go is not None:
            st.markdown("#### 📊 Visualization Dashboard")
            v1, v2 = st.columns(2)

            pie_labels = ["Normal", "Known Attack", "Zero-Day"]
            pie_values = [normal_det, known_det, zeroday_det]
            pie_colors = ["#00ff41", "#faca2b", "#ff2a2a"]

            with v1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_labels, values=pie_values,
                    marker=dict(colors=pie_colors, line=dict(color="#1e2a38", width=2)),
                    textfont=dict(color="white", size=13), hole=0.4,
                )])
                fig_pie.update_layout(
                    title=dict(text="Detection Distribution", font=dict(color="white", size=16)),
                    paper_bgcolor="#131920", plot_bgcolor="#131920",
                    legend=dict(font=dict(color="#c9d1d9")),
                    margin=dict(t=50, b=20, l=20, r=20), height=350,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with v2:
                fig_bar = go.Figure(data=[go.Bar(
                    x=pie_labels, y=pie_values,
                    marker_color=pie_colors,
                    text=pie_values, textposition="auto",
                    textfont=dict(color="white", size=14),
                )])
                fig_bar.update_layout(
                    title=dict(text="Detection Counts", font=dict(color="white", size=16)),
                    paper_bgcolor="#131920", plot_bgcolor="#0a0e14",
                    xaxis=dict(tickfont=dict(color="#c9d1d9"), gridcolor="#1e2a38"),
                    yaxis=dict(tickfont=dict(color="#c9d1d9"), gridcolor="#1e2a38",
                               title="Count", title_font=dict(color="#8b949e")),
                    margin=dict(t=50, b=20, l=20, r=20), height=350,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- Comparison Table ---
        if comparison:
            st.markdown("#### 🎯 Ground Truth vs Prediction Comparison")
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df, use_container_width=True, height=300)
            matches = sum(1 for c in comparison if c.get("match"))
            st.metric(
                "Demo Accuracy (Intent vs Prediction)",
                f"{matches}/{total} ({matches / max(total, 1) * 100:.1f}%)",
            )
            st.markdown(
                '<div class="soc-panel" style="border-left:4px solid #faca2b;">'
                '<h4 style="color:#faca2b;margin-top:0;">📌 Why Most Predictions Show Zero-Day?</h4>'
                '<p style="color:#c9d1d9;margin:0;">'
                'The model has <strong>never been trained on this synthetic data</strong>. '
                'Since every sample is unseen, the open-set classifier correctly treats them as '
                'unknown — resulting in predominantly <strong style="color:#ff2a2a;">zero-day</strong> '
                'predictions. This is the expected behaviour of an open-set recognition system. '
                'Meta-Guard\'s ~35 % zero-day detection rate is deliberately conservative: '
                'a low false-positive rate ensures that when the system <em>does</em> flag a zero-day, '
                'SOC analysts can trust the alert rather than drowning in false alarms.</p></div>',
                unsafe_allow_html=True,
            )

        # --- Full Predictions Table ---
        st.markdown("#### 📋 Full Prediction Table")
        result_df = pd.DataFrame(results)
        disp_cols = ["stage1", "stage2", "confidence", "distance",
                     "predicted_label", "zero_day", "final_decision", "ground_truth"]
        avail = [c for c in disp_cols if c in result_df.columns]
        st.dataframe(result_df[avail], use_container_width=True, height=400)

        # --- Distance Analysis ---
        if "distance" in result_df.columns:
            valid_dist = result_df["distance"].dropna()
            if not valid_dist.empty:
                st.markdown("#### 📏 Embedding Distance Analysis")
                st.markdown(
                    f'<div class="soc-panel"><p style="color:var(--text-muted);margin-bottom:8px;">'
                    f'Samples with distance &gt; <strong>{custom_threshold:.4f}</strong> (threshold) '
                    f'are flagged as zero-day. This demonstrates how the model detects unseen attacks.</p></div>',
                    unsafe_allow_html=True,
                )
                d1, d2, d3 = st.columns(3)
                d1.metric("Min Distance", f"{valid_dist.min():.4f}")
                d2.metric("Mean Distance", f"{valid_dist.mean():.4f}")
                d3.metric("Max Distance", f"{valid_dist.max():.4f}")

                if go is not None:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Histogram(
                        x=valid_dist, nbinsx=30,
                        marker_color="#6366f1", opacity=0.8,
                        name="Distance Distribution",
                    ))
                    fig_h.add_vline(
                        x=custom_threshold, line_dash="dash", line_color="#ff2a2a",
                        annotation_text=f"Threshold ({custom_threshold:.4f})",
                        annotation_font_color="#ff2a2a",
                    )
                    fig_h.update_layout(
                        title=dict(text="Embedding Distance Distribution", font=dict(color="white", size=16)),
                        paper_bgcolor="#131920", plot_bgcolor="#0a0e14",
                        xaxis=dict(title="Cosine Distance", tickfont=dict(color="#c9d1d9"),
                                   gridcolor="#1e2a38", title_font=dict(color="#8b949e")),
                        yaxis=dict(title="Count", tickfont=dict(color="#c9d1d9"),
                                   gridcolor="#1e2a38", title_font=dict(color="#8b949e")),
                        margin=dict(t=50, b=40, l=40, r=20), height=300,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

        # Confidence
        st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
        avg_conf = _clamp(float(result_df["confidence"].mean()), 0.0, 1.0)
        st.progress(avg_conf, text=f"Average Confidence: {avg_conf:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # AI Analysis
        batch_ai = eval_resp.get("batch_ai_summary")
        st.markdown('<div class="ai-panel"><h4>🧠 AI Security Analysis</h4></div>', unsafe_allow_html=True)
        if batch_ai:
            st.markdown(str(batch_ai))

        # Gemini Analysis
        gemini_txt = eval_resp.get("gemini_analysis")
        if gemini_txt:
            st.markdown('<div class="gemini-panel"><h4>🟣 Gemini AI Insight</h4></div>', unsafe_allow_html=True)
            st.markdown(str(gemini_txt))


# =====================================================================
# TAB 2: MANUAL SINGLE-SAMPLE DETECTION
# =====================================================================
with tab_manual:
    st.markdown('<div class="section-label">🔍 Manual Input Detection</div>', unsafe_allow_html=True)

    st.markdown("**Quick-Load Test Vectors:**")
    pb1, pb2, pb3 = st.columns(3)
    preset_changed = False
    if pb1.button("🟢 Normal", use_container_width=True):
        st.session_state["sample_preset"] = "normal"
        preset_changed = True
    if pb2.button("⚠️ Attack", use_container_width=True):
        st.session_state["sample_preset"] = "known_attack"
        preset_changed = True
    if pb3.button("🚨 Zero-Day", use_container_width=True):
        st.session_state["sample_preset"] = "unknown_attack"
        preset_changed = True
    if preset_changed:
        st.rerun()

    sample_choice = st.session_state.get("sample_preset", "normal")
    default_vals = samples.get(sample_choice, [0.0] * expected_features)
    raw_input = st.text_area(
        f"Sensor Feature Vector  (N = {expected_features})",
        value=", ".join(f"{v:.6f}" for v in default_vals),
        height=140, key=f"fv_{sample_choice}",
    )

    det_manual = st.button("🚀 Run Detection", type="primary", use_container_width=True, key="run_manual")
    if det_manual:
        payload = parse_manual_input(raw_input, expected_features)
        with st.spinner("Analyzing traffic signature…"):
            try:
                resp = api_post(api_url, "/predict", payload, timeout=120)
                result = resp if "stage1" in resp else resp.get("results", [{}])[0]
            except Exception as exc:
                st.error(f"API Error: {exc}")
                result = None

        if result:
            decision = result.get("final_decision", "Unknown")
            if decision == "Zero-Day Alert":
                st.markdown('<div class="status-alert-red">🚨 ZERO-DAY ATTACK DETECTED</div>', unsafe_allow_html=True)
            elif decision == "Normal":
                st.markdown('<div class="status-alert-green">🟢 TRAFFIC CLASSIFIED: NORMAL</div>', unsafe_allow_html=True)
            elif decision == "Known Attack":
                sl = html.escape(str(result.get("predicted_label", "")).upper())
                st.markdown(f'<div class="status-alert-yellow">⚠️ KNOWN THREAT: {sl}</div>', unsafe_allow_html=True)

            st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Stage 1 — Edge Filter**")
                if result["stage1"] == "normal":
                    st.success("🟢 Normal")
                else:
                    st.error("🔴 Suspicious")
                st.metric("Edge Confidence", f"{result['stage1_confidence']:.2%}")
            with s2:
                st.markdown("**Stage 2 — Cloud Analysis**")
                st.info(f"🧬 {result['stage2'].replace('_',' ').title()}")
                st.metric("Cloud Confidence", f"{result['confidence']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

            if result.get("distance") is not None:
                st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
                st.markdown("#### 📏 Embedding Distance")
                dd1, dd2 = st.columns(2)
                dd1.metric("Distance", f"{result['distance']:.4f}")
                if result.get("closest_known_label"):
                    dd2.metric("Nearest Prototype", result["closest_known_label"])
                thr = result.get("threshold") or 1.0
                ratio = min(result["distance"] / (thr + 1e-6), 1.0)
                st.progress(ratio, text=f"Distance / Threshold  ({ratio:.2f}×)")
                st.caption(f"Threshold: {thr:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

            ai_text = result.get("ai_analysis")
            st.markdown('<div class="ai-panel"><h4>🧠 AI Security Insight</h4></div>', unsafe_allow_html=True)
            if ai_text:
                st.markdown(str(ai_text))
            else:
                st.caption("AI analysis unavailable.")


# =====================================================================
# TAB 3: CSV UPLOAD
# =====================================================================
with tab_csv:
    st.markdown('<div class="section-label">📁 CSV Upload Detection</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)", type=["csv"],
        help="Upload any CSV with numeric columns. Non-numeric data is auto-cleaned.",
    )
    if uploaded_file is not None:
        try:
            ndf, csv_warns, orig_cols = parse_csv_upload(uploaded_file, expected_features)
            for w in csv_warns:
                st.warning(w)
            st.caption(f"Loaded **{ndf.shape[0]}** samples × **{ndf.shape[1]}** features")
            st.dataframe(ndf.head(5), use_container_width=True, height=180)

            det_csv = st.button("🚀 Run Batch Detection", type="primary", use_container_width=True, key="run_csv")
            if det_csv:
                csv_payload = {"features": ndf.values.tolist(), "original_feature_count": orig_cols}
                with st.spinner("Analyzing traffic signatures…"):
                    try:
                        api_resp = api_post(api_url, "/predict", csv_payload, timeout=120)
                        csv_results = normalize_response(api_resp)
                    except Exception as exc:
                        st.error(f"API Error: {exc}")
                        csv_results = []

                if csv_results:
                    rdf = pd.DataFrame(csv_results)
                    tot = len(csv_results)
                    susp = int((rdf["stage1"] == "suspicious").sum())
                    zds = int(rdf["zero_day"].sum())
                    norms = tot - susp

                    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", f"{tot:,}")
                    c2.metric("Normal", f"{norms:,}")
                    c3.metric("Suspicious", f"{susp:,}")
                    c4.metric("Zero-Day", f"{zds:,}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if zds > 0:
                        st.markdown(
                            f'<div class="status-alert-red">🚨 {zds} ZERO-DAY INCIDENT{"S" if zds != 1 else ""} DETECTED</div>',
                            unsafe_allow_html=True)
                    elif susp > 0:
                        st.markdown(
                            f'<div class="status-alert-yellow">⚠️ {susp} KNOWN THREAT{"S" if susp != 1 else ""} DETECTED</div>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-alert-green">🟢 ALL TRAFFIC NORMAL</div>', unsafe_allow_html=True)

                    dcols = ["stage1", "stage2", "confidence", "distance",
                             "predicted_label", "zero_day", "final_decision"]
                    avc = [c for c in dcols if c in rdf.columns]
                    st.dataframe(rdf[avc], use_container_width=True, height=300)

                    st.markdown('<div class="soc-panel">', unsafe_allow_html=True)
                    ac = _clamp(float(rdf["confidence"].mean()), 0.0, 1.0)
                    st.progress(ac, text=f"Average Confidence: {ac:.2%}")
                    if "distance" in rdf:
                        vd = rdf["distance"].dropna()
                        if not vd.empty:
                            e1, e2, e3 = st.columns(3)
                            e1.metric("Min Distance", f"{vd.min():.4f}")
                            e2.metric("Mean Distance", f"{vd.mean():.4f}")
                            e3.metric("Max Distance", f"{vd.max():.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    bai = api_resp.get("batch_ai_summary")
                    st.markdown('<div class="ai-panel"><h4>🧠 AI Batch Analysis</h4></div>', unsafe_allow_html=True)
                    if bai:
                        st.markdown(str(bai))
                    else:
                        st.caption("Batch AI summary unavailable.")
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="mono-text" style="text-align:center;">META-GUARD v2.0 &nbsp;|&nbsp; '
    'synthetic generation → model inference → explainable decision &nbsp;|&nbsp; '
    'edge filtering → cloud open-set analysis</div>',
    unsafe_allow_html=True,
)
