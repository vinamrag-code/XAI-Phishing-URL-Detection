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
    page_icon="PD",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f5f5f7;
    }

    .title-block {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: left;
    }
    .title-block h1 {
        color: #111827;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }
    .title-block p {
        color: #4b5563;
        margin-top: 0.5rem;
        font-size: 0.95rem;
    }

    .result-phishing {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: left;
        margin: 1rem 0;
    }
    .result-legitimate {
        background: #ecfdf3;
        border: 1px solid #6ee7b7;
        border-radius: 8px;
        padding: 1.25rem;
        text-align: left;
        margin: 1rem 0;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .confidence-text {
        color: #4b5563;
        font-size: 0.9rem;
        margin-top: 0.1rem;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.85rem;
        margin: 0.3rem 0;
    }
    .feature-name {
        color: #6b7280;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.1rem;
    }
    .feature-value {
        color: #111827;
        font-size: 1rem;
        font-weight: 500;
    }

    .section-header {
        color: #374151;
        font-size: 0.95rem;
        font-weight: 600;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .info-box {
        background: #f9fafb;
        border-left: 3px solid #2563eb;
        border-radius: 0 6px 6px 0;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
        color: #4b5563;
        font-size: 0.9rem;
    }

    div[data-testid="stTextInput"] input {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #111827;
        border-radius: 6px;
        font-size: 0.95rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.15);
    }

    .stButton > button {
        background: #2563eb;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.55rem 2rem;
        font-size: 0.95rem;
        width: 100%;
        transition: background 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background: #1d4ed8;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb;
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
    <h1>Phishing URL Detector</h1>
    <p>Machine learning based risk assessment with SHAP and LIME explanations.</p>
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
    analyze_btn = st.button("🔍 Analyze")


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
    st.markdown('<p class="section-header">Detection Result</p>', unsafe_allow_html=True)

    col_rf, col_lr = st.columns(2)

    with col_rf:
        if rf_pred == 1:
            st.markdown(f"""
            <div class="result-phishing">
                <div class="result-text" style="color:#b91c1c;">Phishing detected</div>
                <div class="confidence-text">Random Forest · {rf_prob[1]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legitimate">
                <div class="result-text" style="color:#166534;">Likely legitimate</div>
                <div class="confidence-text">Random Forest · {rf_prob[0]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

    with col_lr:
        if lr_pred == 1:
            st.markdown(f"""
            <div class="result-phishing">
                <div class="result-text" style="color:#b91c1c;">Phishing detected</div>
                <div class="confidence-text">Logistic Regression · {lr_prob[1]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-legitimate">
                <div class="result-text" style="color:#166534;">Likely legitimate</div>
                <div class="confidence-text">Logistic Regression · {lr_prob[0]*100:.1f}% confidence</div>
            </div>""", unsafe_allow_html=True)

    # ── EXTRACTED FEATURES ──
    st.markdown('<p class="section-header">Extracted URL Features</p>', unsafe_allow_html=True)

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
    st.markdown('<p class="section-header">Explainability</p>', unsafe_allow_html=True)

    tab_shap, tab_lime = st.tabs(["SHAP explanation", "LIME explanation"])

    # ── SHAP ──
    with tab_shap:
        st.markdown('<div class="info-box">SHAP shows how each feature influences the model towards a phishing or legitimate decision. Positive values indicate higher phishing risk.</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="info-box">LIME explains this specific prediction by testing small variations around the input URL. It highlights which features were most influential for this exact URL.</div>', unsafe_allow_html=True)

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
    <div class="info-box" style="margin-top:2rem; border-left-color:#d97706;">
        <strong>Disclaimer:</strong> This model analyzes URL structure only, not webpage content.
        It was trained on a 2020 dataset. Use as an initial screening tool together with your existing security controls.
    </div>
    """, unsafe_allow_html=True)

elif analyze_btn and not url_input.strip():
    st.warning("Please enter a URL to analyze.")

else:
    # ── LANDING STATE ──
    st.markdown("""
    <div class="info-box">
        Enter a URL above and select <strong>Analyze</strong> to generate a phishing risk assessment with SHAP and LIME explanations.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-name">What this detects</div>
            <br>
            <div style="color:#4b5563; font-size:0.9rem; line-height:1.7">
            • Suspicious URL length and structure<br>
            • IP addresses used instead of domains<br>
            • Phishing-related keywords (login, verify, secure, etc.)<br>
            • Unusual special characters<br>
            • High entropy (randomized-looking URLs)<br>
            • Missing HTTPS
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-name">How it works</div>
            <br>
            <div style="color:#4b5563; font-size:0.9rem; line-height:1.7">
            • Random Forest and Logistic Regression classify the URL<br>
            • SHAP summarises how each feature affects the model decision<br>
            • LIME explains why this specific URL received its score<br>
            • Models trained on a large phishing and legitimate URL dataset
            </div>
        </div>
        """, unsafe_allow_html=True)