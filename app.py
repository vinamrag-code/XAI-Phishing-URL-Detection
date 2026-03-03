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
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# PROFESSIONAL DARK UI CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f172a;
    color: #e2e8f0;
}

.main {
    background-color: #0f172a;
}

/* HEADER */
.title-block {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 0 40px rgba(37,99,235,0.08);
}

.title-block h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0;
}

.title-block p {
    color: #94a3b8;
    margin-top: 0.6rem;
    font-size: 0.95rem;
}

/* INPUT */
div[data-testid="stTextInput"] input {
    background-color: #111827;
    border: 1px solid #334155;
    color: #e2e8f0;
    border-radius: 10px;
    font-size: 0.95rem;
    padding: 0.75rem;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #2563eb;
    box-shadow: 0 0 0 2px rgba(37,99,235,0.3);
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.65rem 2rem;
    border: none;
    width: 100%;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(59,130,246,0.4);
}

/* RESULT CARDS */
.result-card {
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: #111827;
    border: 1px solid #1f2937;
}

.result-phishing {
    border-left: 4px solid #ef4444;
}

.result-legitimate {
    border-left: 4px solid #22c55e;
}

.result-text {
    font-size: 1.2rem;
    font-weight: 600;
}

.confidence-text {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* FEATURE CARDS */
.feature-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.4rem 0;
    transition: all 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-3px);
    border-color: #2563eb;
}

.feature-name {
    color: #94a3b8;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.feature-value {
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.3rem;
}

/* SECTION HEADER */
.section-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #60a5fa;
    margin: 2rem 0 1rem 0;
}

/* INFO BOX */
.info-box {
    background: #111827;
    border-left: 3px solid #3b82f6;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
    color: #cbd5e1;
    font-size: 0.9rem;
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
    rf = joblib.load('random_forest_model.pkl')
    lr = joblib.load('logistic_regression_model.pkl')
    return rf, lr

rf, lr = load_models()

# ─────────────────────────────────────────────
# SHAP EXPLAINER (Cached)
# ─────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(model):
    return shap.TreeExplainer(model)

shap_explainer = get_shap_explainer(rf)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
SUSPICIOUS_WORDS = ['login','verify','update','secure','account',
                    'banking','confirm','password','credit',
                    'paypal','signin','ebay','amazon','free',
                    'lucky','service','access']

def calculate_entropy(text):
    if not text:
        return 0
    freq = Counter(text)
    length = len(text)
    return -sum((c/length)*math.log2(c/length) for c in freq.values())

def extract_features(url):
    url_lower = url.lower()
    parsed = re.sub(r'https?://', '', url_lower)

    return {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'has_https': 1 if url_lower.startswith('https') else 0,
        'has_ip': 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0,
        'num_subdirs': parsed.count('/'),
        'num_params': url.count('?') + url.count('&'),
        'suspicious_words': sum(1 for w in SUSPICIOUS_WORDS if w in url_lower),
        'special_char_count': sum(1 for c in url if c in '@#%^*~[]{}|\\<>'),
        'digits_count': sum(1 for c in url if c.isdigit()),
        'entropy': calculate_entropy(url),
    }

FEATURE_NAMES = list(extract_features("test").keys())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
<h1>🛡️ Phishing URL Detection System</h1>
<p>Machine Learning based URL risk assessment with SHAP & LIME explainability.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col1:
    url_input = st.text_input("", placeholder="https://example.com/login?verify=true")
with col2:
    analyze = st.button("Analyze")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyze and url_input.strip():

    features = extract_features(url_input.strip())
    X_input = pd.DataFrame([features])[FEATURE_NAMES]

    rf_pred = rf.predict(X_input)[0]
    rf_prob = rf.predict_proba(X_input)[0]
    lr_pred = lr.predict(X_input)[0]
    lr_prob = lr.predict_proba(X_input)[0]

    st.markdown('<p class="section-header">Detection Results</p>', unsafe_allow_html=True)

    col_rf, col_lr = st.columns(2)

    def render_result(pred, prob, model):
        if pred == 1:
            return f"""
            <div class="result-card result-phishing">
                <div class="result-text">🚨 Phishing Detected</div>
                <div class="confidence-text">{model} • {prob[1]*100:.2f}% confidence</div>
            </div>"""
        else:
            return f"""
            <div class="result-card result-legitimate">
                <div class="result-text">✅ Likely Legitimate</div>
                <div class="confidence-text">{model} • {prob[0]*100:.2f}% confidence</div>
            </div>"""

    with col_rf:
        st.markdown(render_result(rf_pred, rf_prob, "Random Forest"), unsafe_allow_html=True)

    with col_lr:
        st.markdown(render_result(lr_pred, lr_prob, "Logistic Regression"), unsafe_allow_html=True)

    # Features
    st.markdown('<p class="section-header">Extracted Features</p>', unsafe_allow_html=True)
    cols = st.columns(5)

    for i, (k,v) in enumerate(features.items()):
        with cols[i%5]:
            val = f"{v:.3f}" if isinstance(v,float) else v
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-name">{k}</div>
                <div class="feature-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    # SHAP
    st.markdown('<p class="section-header">Explainability</p>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["SHAP", "LIME"])

    with tab1:
        shap_vals = shap_explainer.shap_values(X_input)[1][0]

        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#111827')

        colors = ['#ef4444' if v>0 else '#22c55e' for v in shap_vals]
        ax.barh(FEATURE_NAMES, shap_vals, color=colors)
        ax.axvline(0)
        ax.set_title("SHAP Feature Contribution")
        st.pyplot(fig)
        plt.close()

    with tab2:
        np.random.seed(42)
        bg = np.random.rand(200, len(FEATURE_NAMES))
        explainer = lime.lime_tabular.LimeTabularExplainer(
            bg,
            feature_names=FEATURE_NAMES,
            class_names=['Legit','Phishing'],
            mode='classification'
        )
        exp = explainer.explain_instance(X_input.values[0], rf.predict_proba, num_features=10)
        vals = exp.as_list()

        features_l = [v[0] for v in vals]
        weights = [v[1] for v in vals]

        fig2, ax2 = plt.subplots(figsize=(9,5))
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#111827')

        colors = ['#ef4444' if w>0 else '#22c55e' for w in weights]
        ax2.barh(features_l, weights, color=colors)
        ax2.axvline(0)
        ax2.set_title("LIME Local Explanation")
        st.pyplot(fig2)
        plt.close()

elif analyze:
    st.warning("Please enter a URL to analyze.")