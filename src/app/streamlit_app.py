"""
src/app/streamlit_app.py
------------------------
Interactive Web Application for Explainable AI Phishing URL Detection.
Designed with a clean, modern, human-centered UI.
Zero network requests: All evaluations are strictly lexical/structural.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.features.url_features import extract_features, get_feature_names, batch_extract_features
from src.models.predict import load_model, load_feature_schema, predict_url
from src.explain.shap_utils import get_shap_explainer, compute_shap_local
from src.explain.lime_utils import get_lime_explainer, explain_url_lime
from src.explain.reasons import generate_reasons


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Phishing URL Detector — Explainable AI",
    page_icon="https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Clean, Human-Centered UI Styling ──────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #1f2937;
    }

    /* Page Titles & Headers */
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #4b5563;
        margin-bottom: 20px;
        line-height: 1.5;
    }

    /* Simple Info Box */
    .info-banner {
        background-color: #f3f4f6;
        border-left: 3px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #374151;
        margin-bottom: 24px;
        line-height: 1.4;
    }

    /* Tabs Styling - Simple, clean underline tabs */
    div[data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0px;
    }
    div[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 500;
        padding: 8px 12px;
        color: #6b7280;
        border: none;
        background: transparent;
        transition: color 0.15s ease;
    }
    div[data-baseweb="tab"]:hover {
        color: #111827;
        background: transparent;
    }
    div[aria-selected="true"] {
        color: #2563eb !important;
        background: transparent !important;
        border-bottom: 2px solid #2563eb !important;
    }

    /* Verdict Cards - Friendly and clear */
    .card-safe {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .card-danger {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .badge-safe {
        display: inline-block;
        background-color: #16a34a;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .badge-danger {
        display: inline-block;
        background-color: #dc2626;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
    }
    .verdict-headline-safe {
        font-size: 1.4rem;
        font-weight: 700;
        color: #15803d;
        margin: 0 0 6px 0;
    }
    .verdict-headline-danger {
        font-size: 1.4rem;
        font-weight: 700;
        color: #b91c1c;
        margin: 0 0 6px 0;
    }
    .verdict-description {
        font-size: 0.95rem;
        color: #374151;
        margin: 0;
        line-height: 1.5;
    }

    /* Section Subheads */
    .subhead-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #111827;
        margin: 24px 0 12px 0;
    }

    /* Factor items */
    .factor-item-safe {
        background-color: #f0fdf4;
        border-left: 3px solid #22c55e;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: #1f2937;
    }
    .factor-item-danger {
        background-color: #fef2f2;
        border-left: 3px solid #ef4444;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 0.9rem;
        color: #1f2937;
    }

    /* Status dot */
    .status-text {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: #374151;
        margin-bottom: 12px;
    }
    .status-dot-green {
        width: 9px;
        height: 9px;
        background-color: #16a34a;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot-red {
        width: 9px;
        height: 9px;
        background-color: #dc2626;
        border-radius: 50%;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Resource Loaders ───────────────────────────────────────────────────

@st.cache_resource
def get_cached_model():
    """Load and cache the trained champion model pipeline."""
    try:
        pipeline = load_model(artifacts_dir=PROJECT_ROOT / "artifacts")
        schema = load_feature_schema(artifacts_dir=PROJECT_ROOT / "artifacts")
        return pipeline, schema
    except Exception:
        return None, None


@st.cache_resource
def get_cached_explainers(_pipeline, _feature_names):
    """Load background data and create cached SHAP and LIME explainers."""
    if _pipeline is None or _feature_names is None:
        return None, None
    try:
        model_name = getattr(_pipeline.named_steps.get("clf", _pipeline), "__class__", {}).__name__
        proc_dir = PROJECT_ROOT / "data" / "processed"
        X_tr_path = proc_dir / "X_train.npy"
        if X_tr_path.exists():
            bg_data = np.load(X_tr_path)[:200]
        else:
            bg_data = np.zeros((50, len(_feature_names)))

        shap_exp = get_shap_explainer(_pipeline, bg_data, model_name)
        lime_exp = get_lime_explainer(bg_data, _feature_names)
        return shap_exp, lime_exp
    except Exception:
        return None, None


# ── Clean Risk Gauge Chart ────────────────────────────────────────────────────

def create_risk_gauge(probability: float, threshold: float = 0.5) -> go.Figure:
    """Create a simple, clean gauge indicator for phishing risk."""
    bar_color = "#dc2626" if probability >= threshold else "#16a34a"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100.0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 14, 'color': '#4b5563'}},
        number={'suffix': "%", 'font': {'size': 32, 'color': '#111827'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
            'bar': {'color': bar_color, 'thickness': 0.35},
            'bgcolor': "#f8fafc",
            'borderwidth': 1,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 35], 'color': "#dcfce7"},
                {'range': [35, 65], 'color': "#fef9c3"},
                {'range': [65, 100], 'color': "#fee2e2"}
            ],
            'threshold': {
                'line': {'color': "#64748b", 'width': 2},
                'thickness': 0.75,
                'value': threshold * 100.0
            }
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=15),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# ── Main Application ──────────────────────────────────────────────────────────

def main():
    pipeline, feature_names = get_cached_model()
    model_loaded = pipeline is not None

    # ── Sidebar ──
    with st.sidebar:
        st.subheader("Phishing URL Detector")
        st.caption("Explainable machine learning framework")
        st.divider()

        # Simple status indicator
        if model_loaded:
            clf_name = getattr(pipeline.named_steps.get("clf", pipeline), "__class__", {}).__name__
            st.markdown(f"""
            <div class="status-text">
                <span class="status-dot-green"></span>
                <span><b>Model active:</b> {clf_name}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-text">
                <span class="status-dot-red"></span>
                <span><b>Model offline:</b> Please run <code>make train</code></span>
            </div>
            """, unsafe_allow_html=True)

        threshold = st.slider(
            "Detection Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Higher values reduce false alarms; lower values catch more subtle threats."
        )

        st.divider()
        st.markdown("**About this system**")
        st.caption(
            "• Extracts 52 lexical & structural URL features\n"
            "• Zero network requests (safe offline analysis)\n"
            "• Explanations powered by SHAP and LIME\n"
            "• Tested with domain-grouped cross-validation"
        )

        st.divider()
        st.markdown("[API Documentation (FastAPI)](http://localhost:8000/docs)")
        st.caption("Version 1.0.0")

    # ── Main Tabs (Human, Simple, Clear) ──
    tab_single, tab_batch, tab_performance, tab_about = st.tabs([
        "Check a URL",
        "Batch Check",
        "Model Performance",
        "How It Works"
    ])

    # ── TAB 1: Check a URL ────────────────────────────────────────────────────
    with tab_single:
        st.markdown('<div class="app-title">Check a website link</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Enter any web address to check if it looks safe or suspicious, and see the exact reasons why.</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-banner">
            <b>Safe offline inspection:</b> We only inspect the text structure of the URL. 
            The target website is <b>never opened, visited, or contacted</b> over the internet.
        </div>
        """, unsafe_allow_html=True)

        example_urls = [
            "Select an example URL...",
            "https://www.paypal.com/signin",
            "http://192.168.1.50/paypal-update/login.php",
            "https://www.google.com/search?q=machine+learning",
            "http://paypal-account-verify.xyz/login.php",
            "https://github.com/security/advisories",
            "http://www.paypa1.com/login"
        ]
        selected_example = st.selectbox("Try an example URL:", example_urls)

        default_input = "" if selected_example == example_urls[0] else selected_example
        input_url = st.text_input("Enter URL to check:", value=default_input, placeholder="https://example.com/page")

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            check_clicked = st.button("Check URL", type="primary", use_container_width=True)

        if check_clicked or (selected_example != example_urls[0] and input_url):
            if not input_url.strip():
                st.warning("Please enter a URL to analyze.")
            elif not model_loaded:
                st.error("The machine learning model is not loaded yet. Run `python scripts/train.py`.")
            else:
                with st.spinner("Analyzing URL structure and generating explanation..."):
                    pred_res = predict_url(input_url, pipeline, feature_names, threshold=threshold)
                    prob = pred_res["probability"]
                    is_phish = pred_res["is_phishing"]

                    shap_exp, lime_exp = get_cached_explainers(pipeline, feature_names)
                    model_name = getattr(pipeline.named_steps.get("clf", pipeline), "__class__", {}).__name__
                    vec = np.array(pred_res["feature_vector"]).reshape(1, -1)

                    s_loc = compute_shap_local(shap_exp, vec, feature_names, model_name) if shap_exp else None
                    l_res = explain_url_lime(lime_exp, vec, pipeline.predict_proba, num_features=10) if lime_exp else None

                    reasons = generate_reasons(
                        shap_local=s_loc,
                        lime_result=l_res,
                        url_features=pred_res["features"],
                        prediction_label=pred_res["label"],
                        probability=prob,
                        feature_names=feature_names,
                        top_k=5
                    )

                # Clean Verdict Card
                col_verdict, col_gauge = st.columns([3, 2])
                with col_verdict:
                    if is_phish:
                        st.markdown(f"""
                        <div class="card-danger">
                            <span class="badge-danger">Phishing Alert</span>
                            <div class="verdict-headline-danger">This URL looks like phishing</div>
                            <p class="verdict-description">
                                <b>Confidence:</b> {prob * 100.0:.1f}% probability of being a fake or malicious website.<br>
                                This web address contains structural patterns frequently seen in phishing and deceptive scams.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card-safe">
                            <span class="badge-safe">Safe Link</span>
                            <div class="verdict-headline-safe">This URL looks legitimate</div>
                            <p class="verdict-description">
                                <b>Confidence:</b> {(1.0 - prob) * 100.0:.1f}% probability of being legitimate.<br>
                                The domain structure, protocol, and path conform to verified benign website standards.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('<div class="subhead-title">Summary Explanation</div>', unsafe_allow_html=True)
                    st.info(reasons["plain_english"])

                with col_gauge:
                    st.plotly_chart(create_risk_gauge(prob, threshold=threshold), use_container_width=True)

                # Factor Explanations
                st.divider()
                st.markdown('<div class="subhead-title">Why did the model make this decision?</div>', unsafe_allow_html=True)

                col_reasons, col_lime = st.columns(2)

                with col_reasons:
                    st.markdown("**Factors that increase phishing risk:**")
                    if reasons["risk_factors"]:
                        for rf in reasons["risk_factors"]:
                            st.markdown(f"""
                            <div class="factor-item-danger">
                                <b>{rf['feature']}</b>: {rf['explanation']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No significant risk factors detected.")

                    st.markdown("**Factors that indicate this link is safe:**")
                    if reasons["protective_factors"]:
                        for pf in reasons["protective_factors"]:
                            st.markdown(f"""
                            <div class="factor-item-safe">
                                <b>{pf['feature']}</b>: {pf['explanation']}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.write("No specific mitigating factors detected.")

                with col_lime:
                    st.markdown("**Top influential features (LIME):**")
                    if l_res and "feature_weights" in l_res:
                        lime_data = l_res["feature_weights"][:8]
                        feats_l = [x[0] for x in lime_data]
                        weights_l = [x[1] for x in lime_data]
                        colors_l = ["#dc2626" if w > 0 else "#16a34a" for w in weights_l]

                        fig_lime = go.Figure(go.Bar(
                            x=weights_l[::-1],
                            y=feats_l[::-1],
                            orientation='h',
                            marker_color=colors_l[::-1]
                        ))
                        fig_lime.update_layout(
                            title={'text': "Feature Weights", 'font': {'size': 13, 'color': '#6b7280'}},
                            xaxis_title="Contribution (Green = Safe, Red = Phishing)",
                            height=280,
                            margin=dict(l=10, r=10, t=30, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(gridcolor="#f1f5f9")
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)

                # Feature Table
                with st.expander("View all 52 extracted URL features"):
                    feat_items = list(pred_res["features"].items())
                    df_feats = pd.DataFrame(feat_items, columns=["Feature Name", "Extracted Value"])
                    st.dataframe(df_feats, use_container_width=True)

    # ── TAB 2: Batch Check ────────────────────────────────────────────────────
    with tab_batch:
        st.markdown('<div class="app-title">Batch Check URLs</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Upload a CSV file with a column named <code>url</code> to inspect multiple links at once.</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} rows from uploaded file.")

            url_col_candidates = [c for c in batch_df.columns if "url" in c.lower()]
            if not url_col_candidates:
                st.error("Could not find a column containing 'url' in your CSV. Please check the column header.")
            else:
                target_col = url_col_candidates[0]
                urls_to_test = batch_df[target_col].dropna().astype(str).tolist()[:1000]

                if st.button(f"Analyze {len(urls_to_test)} URLs", type="primary"):
                    with st.spinner("Extracting features and scoring batch..."):
                        t_start = time.time()
                        feats_batch = batch_extract_features(urls_to_test)
                        for f in feature_names:
                            if f not in feats_batch.columns:
                                feats_batch[f] = 0.0

                        X_batch = feats_batch[feature_names].values
                        probs = pipeline.predict_proba(X_batch)[:, 1]

                        results_data = []
                        for u, p in zip(urls_to_test, probs):
                            v = "Phishing" if p >= threshold else "Legitimate"
                            lvl = "High Risk" if p >= 0.70 else ("Medium Risk" if p >= 0.50 else "Safe")
                            results_data.append({
                                "URL": u,
                                "Verdict": v,
                                "Phishing Probability": f"{p * 100.0:.1f}%",
                                "Risk Level": lvl
                            })

                        res_df = pd.DataFrame(results_data)
                        proc_sec = time.time() - t_start

                    phish_total = (res_df["Verdict"] == "Phishing").sum()
                    legit_total = len(res_df) - phish_total
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Analyzed", len(res_df))
                    c2.metric("Flagged Phishing", phish_total)
                    c3.metric("Verified Safe", legit_total)
                    c4.metric("Speed", f"{len(res_df) / proc_sec:.1f} URLs/sec")

                    st.dataframe(res_df, use_container_width=True)

                    csv_buf = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_buf,
                        file_name="url_analysis_results.csv",
                        mime="text/csv"
                    )

    # ── TAB 3: Model Performance ──────────────────────────────────────────────
    with tab_performance:
        st.markdown('<div class="app-title">Model Performance & Benchmarks</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Comparison of 6 machine learning models evaluated on unseen registered domains.</div>', unsafe_allow_html=True)

        metrics_csv = PROJECT_ROOT / "artifacts" / "metrics" / "model_comparison.csv"
        if metrics_csv.exists():
            st.markdown('<div class="subhead-title">Model Comparison Table</div>', unsafe_allow_html=True)
            comp_df = pd.read_csv(metrics_csv)
            st.dataframe(comp_df, use_container_width=True)
        else:
            st.info("Run `python scripts/train.py` to populate performance metrics.")

        st.markdown('<div class="subhead-title">Evaluation Charts</div>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns(2)

        roc_plot = PROJECT_ROOT / "artifacts" / "plots" / "roc_curves.png"
        pr_plot = PROJECT_ROOT / "artifacts" / "plots" / "pr_curves.png"
        beeswarm_plot = PROJECT_ROOT / "artifacts" / "plots" / "shap_summary_beeswarm.png"
        perm_plot = PROJECT_ROOT / "artifacts" / "plots" / "permutation_importance.png"
        cal_plot = PROJECT_ROOT / "artifacts" / "plots" / "calibration.png"
        th_plot = PROJECT_ROOT / "artifacts" / "plots" / "threshold_analysis.png"

        with p_col1:
            if roc_plot.exists():
                st.image(str(roc_plot), caption="ROC Curves (6 Models on Grouped Test Set)")
            if beeswarm_plot.exists():
                st.image(str(beeswarm_plot), caption="Global SHAP Feature Attribution")
            if cal_plot.exists():
                st.image(str(cal_plot), caption="Model Calibration Curves")

        with p_col2:
            if pr_plot.exists():
                st.image(str(pr_plot), caption="Precision-Recall Curves")
            if perm_plot.exists():
                st.image(str(perm_plot), caption="Permutation Feature Importance")
            if th_plot.exists():
                st.image(str(th_plot), caption="Decision Threshold Optimization")

    # ── TAB 4: How It Works ───────────────────────────────────────────────────
    with tab_about:
        st.markdown('<div class="app-title">How It Works</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">An overview of the machine learning pipeline and explainability methods.</div>', unsafe_allow_html=True)

        st.markdown("""
        ### System Flow
        1. **Feature Extraction:** Extracts 52 lexical and structural indicators from the raw URL string (length, special characters, entropy, sensitive words, typosquatting signals).
        2. **Machine Learning Model:** Classifies the feature vector into Legitimate or Phishing using trained tree-based and neural network models.
        3. **Explainability Layer:** SHAP (Shapley Additive Explanations) and LIME decompose the decision into human-readable reasons explaining why a URL was flagged or cleared.

        ### Key Technical Highlights
        - **Zero Network Risk:** URLs are evaluated entirely offline as character sequences. The target server is never contacted.
        - **Leak-Free Splitting:** The models were evaluated using `GroupShuffleSplit` on registered domains so that URLs from the same domain never appear in both train and test sets.
        - **Dual Explainability:** Combines SHAP for mathematically consistent feature attribution and LIME for local surrogate verification.

        ### Realistic Limitations
        - **Compromised Legitimate Sites:** If an attacker compromises a legitimate site (such as a hacked school or company website), the URL structure will look legitimate and cannot be detected through URL strings alone.
        - **Adversarial URLs:** Attackers who intentionally structure URLs to match benign patterns can reduce detection confidence.
        - **Defense in Depth:** This system is intended as a fast, privacy-preserving first line of triage, best combined with DNS telemetry and SSL certificate checks.
        """)


if __name__ == "__main__":
    main()
