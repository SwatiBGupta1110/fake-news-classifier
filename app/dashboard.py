"""
app/dashboard.py
----------------
Streamlit live-demo dashboard for the Fake vs Real News classifier.

REQUIRES: models must be trained first.
    python src/train.py

Run:
    streamlit run app/dashboard.py

To disable LLM tab (no API key):
    NO_LLM=1 streamlit run app/dashboard.py
"""
import os
import sys
import json
import time
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
sys.path.insert(0, BASE_DIR)

USE_LLM = os.environ.get("NO_LLM", "0") != "1"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake vs Real News Classifier | AMEX GBT",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #6c757d; margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 1.2rem 1.5rem;
        color: white; text-align: center; margin: 0.3rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; margin-top: 0.2rem; }
    .fake-badge {
        background: #e74c3c; color: white; padding: 0.4rem 1.2rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
        display: inline-block;
    }
    .real-badge {
        background: #2ecc71; color: white; padding: 0.4rem 1.2rem;
        border-radius: 20px; font-weight: 700; font-size: 1.1rem;
        display: inline-block;
    }
    .reason-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 0.6rem 1rem; border-radius: 4px; margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
    div[data-testid="stSidebar"] { background: #1a1a2e; }
    div[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import joblib
    models = {}
    for name, fname in [("lr", "lr_pipeline.joblib"), ("svc", "svc_pipeline.joblib")]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_metrics():
    path = os.path.join(METRICS_DIR, "track1_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_comparison():
    path = os.path.join(METRICS_DIR, "track1_vs_track2.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_robustness():
    path = os.path.join(METRICS_DIR, "track1_robustness.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


models = load_models()
MODELS_READY = bool(models)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📰 Fake News Detector")
    st.markdown("**AMEX GBT — ML Take-Home**")
    st.markdown("---")

    st.markdown("### Model Status")
    for label, key in [("Logistic Regression", "lr"), ("LinearSVC", "svc")]:
        status = "✅ Loaded" if key in models else "❌ Not found"
        st.markdown(f"**{label}:** {status}")

    st.markdown("---")
    st.markdown("### Navigation")
    st.markdown("Use the tabs above to explore:")
    st.markdown("- **Live Demo** — classify any article")
    st.markdown("- **EDA** — dataset insights")
    st.markdown("- **Model Performance** — metrics & curves")
    st.markdown("- **Business Case** — recommendations")
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("[FastAPI Docs](http://localhost:8000/docs)")
    st.markdown("[GitHub](https://github.com)")
    st.markdown("---")
    st.caption("Built with scikit-learn + OpenAI + Streamlit")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📰 Fake vs Real News Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">American Express Global Business Travel — ML Engineer Assignment</p>', unsafe_allow_html=True)

if not MODELS_READY:
    st.error(
        "Models not found. Run `python src/train.py` first, then restart this app.",
        icon="🚨"
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_demo, tab_eda, tab_perf, tab_biz = st.tabs([
    "🔍 Live Demo", "📊 EDA & Insights", "📈 Model Performance", "💼 Business Case"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### Classify an Article in Real Time")
    st.markdown("Paste any news article below and see the prediction, confidence, and — optionally — GPT-4o-mini's reasoning.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        sample_articles = {
            "Choose an example…": "",
            "Real — Reuters style": (
                "WASHINGTON (Reuters) - The Senate passed the $1.9 trillion COVID-19 "
                "relief bill on Saturday in a party-line vote, clearing it for President "
                "Biden's signature. The legislation, the American Rescue Plan, includes "
                "$1,400 direct payments to most Americans, $350 billion for state and "
                "local governments, and billions for vaccine distribution."
            ),
            "Fake — Sensational style": (
                "BREAKING: Deep state operatives have been exposed planting fake evidence "
                "at the White House! Sources say the president was NEVER informed. Patriots "
                "are rising up! Share this before it gets deleted! "
                "The mainstream media won't touch this story!!! #TruthBombs"
            ),
        }

        example = st.selectbox("Load an example article:", list(sample_articles.keys()))
        if example != "Choose an example…":
            default_text = sample_articles[example]
        else:
            default_text = ""

        title_input = st.text_input("Article Title (optional):", placeholder="e.g. Senate passes relief bill")
        text_input  = st.text_area(
            "Article Text:",
            value=default_text,
            height=200,
            placeholder="Paste article text here…",
        )

        col_m, col_t = st.columns(2)
        with col_m:
            model_choice = st.selectbox("Baseline Model:", ["LinearSVC (fastest)", "Logistic Regression"])
        with col_t:
            use_llm = st.checkbox(
                "Use GPT-4o-mini for uncertain articles",
                value=False,
                disabled=not USE_LLM,
                help="Routes low-confidence predictions to GPT-4o-mini for a second opinion."
            )
            if use_llm:
                llm_threshold = st.slider("Confidence threshold for LLM escalation:", 0.5, 0.99, 0.80, 0.01)
            else:
                llm_threshold = 0.80

        classify_btn = st.button("🔍 Classify Article", type="primary", disabled=not MODELS_READY)

    with col_right:
        if classify_btn and text_input.strip():
            model_key = "svc" if "LinearSVC" in model_choice else "lr"
            pipeline  = models[model_key]
            content   = ((title_input.strip() + " ") if title_input else "") + text_input.strip()
            content   = content.lower()

            with st.spinner("Classifying…"):
                t0    = time.time()
                proba = pipeline.predict_proba([content])[0]
                conf  = float(max(proba))
                pred  = int(proba.argmax())
                label = "Real" if pred == 1 else "Fake"
                source = "baseline"
                reasons = []
                llm_conf = None
                needs_review = False

                if use_llm and conf < llm_threshold:
                    try:
                        from src.llm_inference import classify_with_llm
                        llm_result = classify_with_llm(content)
                        label      = llm_result["label"]
                        llm_conf   = llm_result["confidence"]
                        reasons    = llm_result["reasons"]
                        needs_review = llm_result["needs_human_review"]
                        source     = "gpt-4o-mini"
                    except Exception as e:
                        st.warning(f"LLM call failed: {e}. Using baseline result.")

                elapsed_ms = (time.time() - t0) * 1000

            # ── Result display ──────────────────────────────────────────────
            badge = f'<span class="{"real-badge" if label == "Real" else "fake-badge"}">{label.upper()}</span>'
            st.markdown(f"**Prediction:** {badge}", unsafe_allow_html=True)
            st.markdown(f"**Source:** `{source}` | **Latency:** `{elapsed_ms:.1f} ms`")

            # Confidence gauge
            gauge_val   = llm_conf if llm_conf is not None else conf
            gauge_color = "#2ecc71" if label == "Real" else "#e74c3c"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_val * 100,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": "Confidence", "font": {"size": 14}},
                gauge={
                    "axis"      : {"range": [0, 100]},
                    "bar"       : {"color": gauge_color},
                    "steps"     : [
                        {"range": [0, 50],  "color": "#fadbd8"},
                        {"range": [50, 80], "color": "#fdebd0"},
                        {"range": [80, 100],"color": "#d5f5e3"},
                    ],
                    "threshold" : {
                        "line"  : {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value" : 80,
                    },
                },
            ))
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # LLM reasons
            if reasons:
                st.markdown("**GPT-4o-mini reasoning:**")
                for r in reasons:
                    st.markdown(f'<div class="reason-card">• {r}</div>', unsafe_allow_html=True)

            if needs_review:
                st.warning("Low confidence — human review recommended.", icon="⚠️")

            # Confidence breakdown
            if source == "baseline":
                st.markdown(f"**Baseline confidence (Fake):** `{proba[0]:.2%}` | **Real:** `{proba[1]:.2%}`")

        elif classify_btn:
            st.info("Please paste an article to classify.")
        else:
            st.info("Enter or paste an article text on the left, then click **Classify Article**.")
            st.markdown("**What this demo does:**")
            st.markdown("- Runs your article through a trained TF-IDF + LinearSVC pipeline")
            st.markdown("- Shows a confidence gauge and prediction label")
            st.markdown("- Optionally escalates uncertain articles to GPT-4o-mini")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA & INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">29,204</div><div class="metric-label">Total Articles</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">7,787</div><div class="metric-label">Fake Articles</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">21,417</div><div class="metric-label">Real Articles</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">26.7%</div><div class="metric-label">Fake (minority class)</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Load and show saved figures
    figure_files = {
        "Class Distribution": "01_class_distribution.png",
        "Word Count by Class": "02_word_count_by_class.png",
        "Top Words per Class": "03_top_words_per_class.png",
        "TF-IDF Discriminative Words": "04_tfidf_discriminative_words.png",
    }

    col_a, col_b = st.columns(2)
    items = list(figure_files.items())

    for i, (title, fname) in enumerate(items):
        col = col_a if i % 2 == 0 else col_b
        path = os.path.join(FIGURES_DIR, fname)
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.info(f"Run Track 1 notebook or `python src/train.py` to generate this figure.")

    st.markdown("---")
    st.markdown("### Key EDA Insights")

    insights = [
        ("📏 Length difference", "Real articles average ~600 words; Fake articles ~350 words. Real news includes full context and attributed quotes."),
        ("❗ Emotional language", "Fake articles use significantly more exclamation marks — a strong signal of sensationalist framing."),
        ("🏛️ Vocabulary divergence", "Real news vocabulary: *reuters, government, officials, minister, said*. Fake: *just, image, twitter, like, share*."),
        ("⚠️ Subject column leakage", "The `subject` column contains values like `politicsNews` only in Real articles — a near-perfect label proxy. Dropped from features."),
        ("⚖️ Class imbalance", "27% Fake / 73% Real. Moderate imbalance — handled via `class_weight='balanced'` in both models."),
    ]

    for icon_title, detail in insights:
        with st.expander(icon_title):
            st.write(detail)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown("### Model Evaluation Results")

    metrics_df = load_metrics()
    if metrics_df is not None:
        # Summary metrics in columns
        cols = st.columns(len(metrics_df))
        for col, (model_name, row) in zip(cols, metrics_df.iterrows()):
            with col:
                st.markdown(f"**{model_name}**")
                for metric, val in row.items():
                    delta_color = "normal"
                    st.metric(metric, f"{val:.4f}")

        # Bar chart comparison
        fig = px.bar(
            metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            title="Model Performance Comparison",
            color_discrete_sequence=["#667eea", "#764ba2"],
            range_y=[0.95, 1.0],
        )
        fig.update_layout(height=400, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to generate metrics.")

    st.markdown("---")

    # Confusion matrix + ROC images
    col_cm, col_roc = st.columns(2)
    with col_cm:
        p = os.path.join(FIGURES_DIR, "05_confusion_matrices.png")
        st.markdown("**Confusion Matrices**")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("Run training to generate.")

    with col_roc:
        p = os.path.join(FIGURES_DIR, "06_roc_pr_curves.png")
        st.markdown("**ROC & PR Curves**")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("Run training to generate.")

    st.markdown("---")
    st.markdown("### Robustness Under Input Degradation")

    rob_df = load_robustness()
    if rob_df is not None:
        fig = px.bar(
            rob_df,
            x="Condition", y="F1-Macro",
            title="LinearSVC — F1-Macro Under Degraded Input",
            color="Condition",
            color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
            text="F1-Macro",
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=380, showlegend=False, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_column_width=True)

        p = os.path.join(FIGURES_DIR, "07_robustness_comparison.png")
        if os.path.exists(p):
            st.image(p, use_column_width=True)
    else:
        st.info("Run Track 1 notebook to generate robustness data.")

    st.markdown("---")
    st.markdown("### Track 1 vs Track 2 Comparison")

    comp_df = load_comparison()
    if comp_df is not None:
        st.dataframe(comp_df.set_index("Track"), use_container_width=True)
    else:
        st.info("Run Track 2 notebook to generate comparison data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BUSINESS CASE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_biz:
    st.markdown("### Business Case & Deployment Recommendations")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("#### Track 1 — Production Baseline")
        st.markdown("""
**TF-IDF + LinearSVC Pipeline**

✅ **Best for:** High-volume, cost-sensitive, latency-critical deployments

| Dimension | Value |
|---|---|
| Inference latency | < 1 ms / article |
| Infrastructure | Single `joblib` file |
| GPU required | No |
| API dependency | None |
| Cost per 1M articles | ~$0 |
| Interpretability | Low (weights) |

**Deployment:** Containerise as a REST microservice.
Single binary (`svc_pipeline.joblib`) — no GPU, no network call.
        """)

    with col_t2:
        st.markdown("#### Track 2 — Hybrid LLM Pipeline")
        st.markdown("""
**Baseline + GPT-4o-mini for uncertain articles**

✅ **Best for:** Analyst-facing tools where explainability matters

| Dimension | Value |
|---|---|
| Inference latency (baseline) | < 1 ms |
| Inference latency (LLM) | 1–3 s |
| LLM API dependency | OpenAI |
| Cost per LLM call | ~$0.001 |
| LLM routing rate | ~15–25% |
| Interpretability | High (reasons) |

**Deployment:** Same baseline + OpenAI API call for escalated cases.
Routing threshold tunable without retraining.
        """)

    st.markdown("---")
    st.markdown("### Decision Matrix")

    decision_data = {
        "Use Case": [
            "Real-time feed moderation (high volume)",
            "Editorial review tool",
            "Analyst dashboard with explanations",
            "Offline batch scoring",
            "Privacy-sensitive deployment",
        ],
        "Recommended Track": [
            "Track 1 (LinearSVC)",
            "Track 2 (Hybrid LLM)",
            "Track 2 (Hybrid LLM)",
            "Track 1 (LinearSVC)",
            "Track 1 or local LLM",
        ],
        "Reason": [
            "Latency & cost critical — sub-ms needed",
            "Explanations build editor trust",
            "Textual reasons for ambiguous articles",
            "No latency constraint — accuracy priority",
            "Article text stays on-premises",
        ],
    }
    st.dataframe(pd.DataFrame(decision_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Failure Cases & Improvement Roadmap")

    roadmap = {
        "Area": ["Data", "Data", "Modeling", "Modeling", "Evaluation", "Monitoring"],
        "Issue": [
            "Limited topic diversity (US politics only)",
            "Dataset may be dated (2016–2018 news)",
            "Lexical model fails on style-mimic fakes",
            "No calibration curve computed",
            "Robustness under paraphrasing not tested",
            "No production drift detection",
        ],
        "Improvement": [
            "Add multi-topic, multilingual sources",
            "Curate recent news data",
            "Fine-tune DistilBERT or RoBERTa on domain data",
            "Add Platt scaling + reliability diagram",
            "Add paraphrase robustness benchmark",
            "Add confidence distribution monitoring + alerting",
        ],
        "Priority": ["Medium", "High", "Medium", "Low", "Medium", "High"],
    }
    st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### GenAI Privacy & Cost Considerations")
    st.info(
        "**Privacy**: Article text is sent to OpenAI's API when `use_llm=True`. "
        "For confidential documents, replace with a locally hosted model (e.g. Mistral 7B via Ollama).\n\n"
        "**Cost (GPT-4o-mini)**: $0.15/1M input tokens. At 20% LLM routing, classifying 1M articles costs ~$24.\n\n"
        "**Reproducibility**: `temperature=0.1` + strict Pydantic schema minimises output variance across runs.",
        icon="ℹ️"
    )
