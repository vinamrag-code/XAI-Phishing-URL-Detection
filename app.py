import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import shap
import lime
import lime.lime_tabular
import re
import math
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background-color: #0e1117; }

    .title-block {
        background: linear-gradient(135deg, #1a1f2e, #0d1117);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .title-block h1 {
        font-family: 'Space Mono', monospace;
        color: #58a6ff;
        font-size: 2.2rem;
        margin: 0;
    }
    .title-block p {
        color: #8b949e;
        margin-top: 0.5rem;
        font-size: 1rem;
    }

    .result-phishing {
        background: linear-gradient(135deg, #3d1a1a, #2d1010);
        border: 1px solid #f85149;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-legitimate {
        background: linear-gradient(135deg, #1a3d2b, #102d1e);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-text {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .confidence-text {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    .feature-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.3rem 0;
    }
    .feature-name {
        color: #58a6ff;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
    }
    .feature-value {
        color: #e6edf3;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        color: #58a6ff;
        font-size: 1rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    .info-box {
        background: #1c2128;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #8b949e;
        font-size: 0.9rem;
    }

    div[data-testid="stTextInput"] input {
        background-color: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
        font-family: 'Space Mono', monospace;
        border-radius: 8px;
        font-size: 0.95rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #58a6ff;
        box-shadow: 0 0 0 2px rgba(88,166,255,0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        padding: 0.6rem 2rem;
        font-size: 0.95rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388bfd, #58a6ff);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace;
        color: #8b949e;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf = joblib.load('random_forest_model.pkl')
        lr = joblib.load('logistic_regression_model.pkl')
        return rf, lr
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Make sure .pkl files are in the same folder as app.py")
        st.stop()

rf, lr = load_models()


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
SUSPICIOUS_WORDS = ['login', 'verify', 'update', 'secure', 'account',
                    'banking', 'confirm', 'password', 'credit', 'paypal',
                    'signin', 'ebay', 'amazon', 'free', 'lucky', 'service', 'access']

def calculate_entropy(text):
    if not text:
        return 0
    freq = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())

def extract_features(url):
    url_lower = url.lower()
    parsed_path = re.sub(r'https?://', '', url_lower)

    features = {
        'url_length':         len(url),
        'num_dots':           url.count('.'),
        'has_https':          1 if url_lower.startswith('https') else 0,
        'has_ip':             1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        'num_subdirs':        parsed_path.count('/'),
        'num_params':         url.count('?') + url.count('&'),
        'suspicious_words':   sum(1 for word in SUSPICIOUS_WORDS if word in url_lower),
        'special_char_count': sum(1 for c in url if c in '@#%^*~[]{}|\\<>'),
        'digits_count':       sum(1 for c in url if c.isdigit()),
        'entropy':            calculate_entropy(url),
    }
    return features

FEATURE_NAMES = ['url_length', 'num_dots', 'has_https', 'has_ip',
                 'num_subdirs', 'num_params', 'suspicious_words',
                 'special_char_count', 'digits_count', 'entropy']

FEATURE_DESCRIPTIONS = {
    'url_length':         'Total length of the URL',
    'num_dots':           'Number of dots (.) in the URL',
    'has_https':          'Whether URL uses HTTPS (1=yes, 0=no)',
    'has_ip':             'Whether URL contains an IP address',
    'num_subdirs':        'Number of subdirectories (slashes)',
    'num_params':         'Number of query parameters',
    'suspicious_words':   'Count of suspicious keywords found',
    'special_char_count': 'Count of special characters (@, #, % etc.)',
    'digits_count':       'Number of digits in the URL',
    'entropy':            'Randomness/complexity of the URL string',
}


# ─────────────────────────────────────────────
# SHAP EXPLAINER (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_rf_model):
    return shap.TreeExplainer(_rf_model)

shap_explainer = get_shap_explainer(rf)


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🔐 Phishing URL Detector</h1>
    <p>ML-powered detection with Explainable AI — SHAP & LIME transparency layer</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — INPUT
# ─────────────────────────────────────────────
col_input, col_btn = st.columns([4, 1])
with col_input:
    url_input = st.text_input(
        label="URL",
        placeholder="https://example.com/login?verify=true",
        label_visibility="collapsed"
    )
with col_btn:
    st.write("")
    analyze_btn = st.button("Analyze")


# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyze_btn and url_input.strip():
    url = url_input.strip()

    # Extract features
    features = extract_features(url)
    X_input = pd.DataFrame([features])[FEATURE_NAMES]

    # Predictions
    rf_pred  = rf.predict(X_input)[0]
    rf_prob  = rf.predict_proba(X_input)[0]
    lr_pred  = lr.predict(X_input)[0]
    lr_prob  = lr.predict_proba(X_input)[0]

    # ── RESULT BANNER ──
    st.markdown('<p class="section-header">// DETECTION RESULT</p>', unsafe_allow_html=True)

    col_rf, col_lr = st.columns(2)

    with col_rf:
        if rf_pred == 1:
            st.markdown(f"""
            <div class="result-phishing">
                <div class="result-text" style="color:#f85149;">⚠️ PHISHING</div>
                <div class="confidence-text">Random Forest — {rf_prob[1]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legitimate">
                <div class="result-text" style="color:#3fb950;">✅ LEGITIMATE</div>
                <div class="confidence-text">Random Forest — {rf_prob[0]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

    with col_lr:
        if lr_pred == 1:
            st.markdown(f"""
            <div class="result-phishing">
                <div class="result-text" style="color:#f85149;">⚠️ PHISHING</div>
                <div class="confidence-text">Logistic Regression — {lr_prob[1]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legitimate">
                <div class="result-text" style="color:#3fb950;">✅ LEGITIMATE</div>
                <div class="confidence-text">Logistic Regression — {lr_prob[0]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

    # ── EXTRACTED FEATURES ──
    st.markdown('<p class="section-header">// EXTRACTED URL FEATURES</p>', unsafe_allow_html=True)

    feat_cols = st.columns(5)
    for i, (feat, val) in enumerate(features.items()):
        with feat_cols[i % 5]:
            display_val = f"{val:.3f}" if isinstance(val, float) else str(val)
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-name">{feat}</div>
                <div class="feature-value">{display_val}</div>
            </div>""", unsafe_allow_html=True)

    # ── XAI TABS ──
    st.markdown('<p class="section-header">// EXPLAINABILITY LAYER</p>', unsafe_allow_html=True)

    tab_shap, tab_lime = st.tabs(["📊 SHAP Explanation", "🔬 LIME Explanation"])

    # ── SHAP ──
    with tab_shap:
        st.markdown('<div class="info-box">SHAP shows how each feature <strong>pushed</strong> the prediction toward phishing or legitimate. Red = pushed toward phishing, Blue = pushed toward legitimate.</div>', unsafe_allow_html=True)

        try:
            shap_vals = shap_explainer.shap_values(X_input)

            # Handle different shap_values formats
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
                base_val = shap_explainer.expected_value[1]
            else:
                sv = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]
                base_val = (shap_explainer.expected_value[1]
                            if hasattr(shap_explainer.expected_value, '__len__')
                            else shap_explainer.expected_value)

            # Waterfall plot
            fig_shap, ax = plt.subplots(figsize=(9, 5))
            fig_shap.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#161b22')

            colors = ['#f85149' if v > 0 else '#3fb950' for v in sv]
            bars = ax.barh(FEATURE_NAMES, sv, color=colors, edgecolor='none', height=0.6)

            ax.set_xlabel('SHAP Value (impact on prediction)', color='#8b949e', fontsize=9)
            ax.set_title('Feature Contributions — Random Forest', color='#58a6ff',
                         fontsize=11, fontfamily='monospace', pad=12)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines[:].set_color('#30363d')
            ax.axvline(0, color='#30363d', linewidth=1)

            for bar, val in zip(bars, sv):
                ax.text(val + (0.002 if val >= 0 else -0.002),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}',
                        va='center',
                        ha='left' if val >= 0 else 'right',
                        color='#e6edf3', fontsize=7.5)

            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

        except Exception as e:
            st.error(f"SHAP error: {e}")

    # ── LIME ──
    with tab_lime:
        st.markdown('<div class="info-box">LIME explains this specific prediction by testing small variations around the input URL. It shows which features were most influential for <strong>this exact URL</strong>.</div>', unsafe_allow_html=True)

        try:
            # Need training data for LIME — generate synthetic reference data
            # Using feature value ranges typical for the dataset
            np.random.seed(42)
            n_bg = 200
            bg_data = np.column_stack([
                np.random.randint(20, 200, n_bg),    # url_length
                np.random.randint(1, 8, n_bg),        # num_dots
                np.random.randint(0, 2, n_bg),        # has_https
                np.random.randint(0, 2, n_bg),        # has_ip
                np.random.randint(0, 10, n_bg),       # num_subdirs
                np.random.randint(0, 5, n_bg),        # num_params
                np.random.randint(0, 4, n_bg),        # suspicious_words
                np.random.randint(0, 10, n_bg),       # special_char_count
                np.random.randint(0, 30, n_bg),       # digits_count
                np.random.uniform(2.0, 5.5, n_bg),   # entropy
            ])

            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=bg_data,
                feature_names=FEATURE_NAMES,
                class_names=['Legitimate', 'Phishing'],
                mode='classification'
            )

            exp = lime_explainer.explain_instance(
                data_row=X_input.values[0],
                predict_fn=rf.predict_proba,
                num_features=10
            )

            lime_vals = exp.as_list()
            lime_features = [x[0] for x in lime_vals]
            lime_weights  = [x[1] for x in lime_vals]

            fig_lime, ax = plt.subplots(figsize=(9, 5))
            fig_lime.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#161b22')

            colors = ['#f85149' if w > 0 else '#3fb950' for w in lime_weights]
            bars = ax.barh(lime_features, lime_weights, color=colors, edgecolor='none', height=0.6)

            ax.set_xlabel('Weight (positive = phishing, negative = legitimate)', color='#8b949e', fontsize=9)
            ax.set_title('LIME Local Explanation — Random Forest', color='#58a6ff',
                         fontsize=11, fontfamily='monospace', pad=12)
            ax.tick_params(colors='#8b949e', labelsize=8)
            ax.spines[:].set_color('#30363d')
            ax.axvline(0, color='#30363d', linewidth=1)

            for bar, val in zip(bars, lime_weights):
                ax.text(val + (0.002 if val >= 0 else -0.002),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}',
                        va='center',
                        ha='left' if val >= 0 else 'right',
                        color='#e6edf3', fontsize=7.5)

            plt.tight_layout()
            st.pyplot(fig_lime)
            plt.close()

        except Exception as e:
            st.error(f"LIME error: {e}")

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="info-box" style="margin-top:2rem; border-left-color:#d29922;">
        ⚠️ <strong>Disclaimer:</strong> This model analyzes URL structure only — not webpage content.
        It was trained on a 2020 dataset. Use as a first-pass filter alongside other security tools.
    </div>
    """, unsafe_allow_html=True)

elif analyze_btn and not url_input.strip():
    st.warning("Please enter a URL to analyze.")

else:
    # ── LANDING STATE ──
    st.markdown("""
    <div class="info-box">
        👆 Enter any URL above and click <strong>Analyze</strong> to get an instant phishing detection
        with full SHAP and LIME explainability.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-name">// WHAT THIS DETECTS</div>
            <br>
            <div style="color:#8b949e; font-size:0.9rem; line-height:1.7">
            • Suspicious URL length & structure<br>
            • IP addresses used instead of domains<br>
            • Phishing keywords (login, verify, secure...)<br>
            • Unusual special characters<br>
            • High entropy (randomized URLs)<br>
            • Missing HTTPS
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-name">// HOW IT WORKS</div>
            <br>
            <div style="color:#8b949e; font-size:0.9rem; line-height:1.7">
            • <strong style="color:#58a6ff">Layer 1</strong>: Random Forest + Logistic Regression classify the URL<br>
            • <strong style="color:#58a6ff">Layer 2</strong>: SHAP explains global feature importance<br>
            • <strong style="color:#58a6ff">Layer 2</strong>: LIME explains this specific prediction<br>
            • Both models trained on 159K+ phishing URLs
            </div>
        </div>
        """, unsafe_allow_html=True)