"""Streamlit interface for the trained fake-review detection model.

The app loads the artifacts produced by FRD.ipynb, applies the same text and
metadata transformations, predicts Fake (CG) or Genuine (OR), and displays
the saved evaluation results.
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
import os
from streamlit_option_menu import option_menu

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Page configuration
st.set_page_config(
    page_title="Fake Review Detection System",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 16.5px;
        font-weight: 400;
    }

    /* Main App Background - Dark Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1422 25%, #13182a 50%, #0a0e1a 100%);
        background-attachment: fixed;
    }

    /* Main Container - Glassmorphism Effect */
    .main .block-container {
        background: rgba(18, 22, 35, 0.75);
        backdrop-filter: blur(12px);
        border-radius: 24px;
        border: 1px solid rgba(72, 85, 120, 0.25);
        padding: 2rem 2.5rem 3rem !important;
        box-shadow: 0 25px 45px -12px rgba(0, 0, 0, 0.4);
        margin-top: 1rem;
    }

    /* Header Styles */
    .main-header {
        text-align: center;
        padding: 2rem 2rem 1.5rem;
        background: linear-gradient(135deg, rgba(56, 68, 168, 0.16) 0%, rgba(255, 75, 75, 0.10) 100%);
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin-bottom: 2rem;
        backdrop-filter: blur(5px);
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0 0 0.5rem;
        letter-spacing: 0;
    }
    .main-header .shield-icon {
        font-size: 2.7rem;
        line-height: 1;
        vertical-align: -0.08em;
        margin-right: 0.2rem;
    }
    .main-header p {
        font-size: 1rem;
        color: #cbd5e1;
        margin: 0;
    }

    /* Labels - Larger Font */
    .stTextArea label, .stSelectbox label, .stCheckbox label {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        margin-bottom: 10px !important;
    }

    /* Textarea - Larger Font */
    textarea {
        font-size: 14px !important;
        font-family: 'Inter', monospace !important;
        border-radius: 12px !important;
        border: 1.5px solid rgba(72, 85, 120, 0.4) !important;
        background: rgba(15, 20, 35, 0.9) !important;
        color: #e2e8f0 !important;
        line-height: 1.6 !important;
    }
    textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
    }

    /* Selectbox - Larger Font */
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 1.5px solid rgba(72, 85, 120, 0.4) !important;
        background: rgba(15, 20, 35, 0.9) !important;
        font-size: 16px !important;
        color: #e2e8f0 !important;
    }

    /* Checkbox - Larger Font */
    .stCheckbox label span {
        font-size: 16px !important;
        color: #e2e8f0 !important;
    }

    /* Star Rating */
    [data-testid="stFeedback"] button {
        transform: scale(2.2);
        margin: 0 10px;
    }
    [data-testid="stFeedback"] {
        gap: 25px !important;
        margin: 20px 0 15px 0;
    }
    [data-testid="stFeedback"] button svg {
        width: 32px !important;
        height: 32px !important;
    }
    [data-testid="stFeedback"] button svg path {
        fill: #fbbf24 !important;
        stroke: #fbbf24 !important;
    }

    /* Rating Label */
    .rating-label {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        margin: 0 0 10px 0;
        line-height: 1.4;
        display: block;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.45);
    }
    .stButton > button:not([kind="primary"]) {
        background: rgba(30, 35, 55, 0.8) !important;
        border: 1.5px solid rgba(72, 85, 120, 0.4) !important;
        color: #a5b4fc !important;
    }
    .st-key-demo_genuine_button button {
        background: #16a34a !important;
        background-image: linear-gradient(135deg, #15803d 0%, #22c55e 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(34, 197, 94, 0.75) !important;
        box-shadow: inset 0 0 0 999px rgba(22, 163, 74, 0.95), 0 4px 14px rgba(34, 197, 94, 0.28) !important;
    }
    .st-key-demo_fake_button button {
        background: #dc2626 !important;
        background-image: linear-gradient(135deg, #991b1b 0%, #dc2626 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(248, 113, 113, 0.85) !important;
        box-shadow: inset 0 0 0 999px rgba(220, 38, 38, 0.94), 0 4px 14px rgba(220, 38, 38, 0.3) !important;
    }
    .st-key-detect_button button {
        background: #4f46e5 !important;
        background-image: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(129, 140, 248, 0.8) !important;
        box-shadow: inset 0 0 0 999px rgba(79, 70, 229, 0.92), 0 4px 14px rgba(79, 70, 229, 0.35) !important;
    }
    .st-key-demo_genuine_button button *,
    .st-key-demo_fake_button button *,
    .st-key-detect_button button * {
        color: #ffffff !important;
    }
    .st-key-demo_genuine_button button:hover,
    .st-key-demo_fake_button button:hover,
    .st-key-detect_button button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
    }

    /* Result Cards - Icon Beside Text */
    .result-card {
        border-radius: 20px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .result-left {
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }
    .result-icon {
        font-size: 2.8rem;
    }
    .result-text h2 {
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .result-text p {
        font-size: 0.95rem;
        margin: 0.2rem 0 0 0;
        opacity: 0.85;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.4rem 1.3rem;
        border-radius: 30px;
        font-size: 1.0rem;
        font-weight: 600;
        text-align: center;
    }
    .fake-result {
        background: linear-gradient(135deg, rgba(185, 28, 28, 0.15) 0%, rgba(153, 27, 27, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .fake-result .result-text h2 { color: #f87171; }
    .genuine-result {
        background: linear-gradient(135deg, rgba(22, 163, 74, 0.15) 0%, rgba(21, 128, 61, 0.1) 100%);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }
    .genuine-result .result-text h2 { color: #4ade80; }

    /* Feature Cards - Larger Font */
    .feature-card {
        background: rgba(15, 23, 42, 0.86);
        border: 1px solid rgba(125, 211, 252, 0.22);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
        transition: border-color 0.2s ease;
    }
    .feature-card:hover {
        border-color: rgba(125, 211, 252, 0.45);
    }
    .feature-card .feat-name {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7dd3fc;
        margin-bottom: 0.5rem;
    }
    .feature-card .feat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        font-family: 'Inter', monospace;
    }
    .feature-card .feat-status {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        margin-bottom: 0.65rem;
        border: 1px solid rgba(125, 211, 252, 0.35);
        border-radius: 999px;
        background: rgba(14, 116, 144, 0.18);
        color: #bae6fd;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .feature-card .feat-desc {
        font-size: 0.9rem;
        color: #cbd5e1;
        line-height: 1.4;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080b12 0%, #111827 52%, #0b1019 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.16) !important;
        box-shadow: 12px 0 30px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.14) !important;
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #a5b4fc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Headings - Larger */
    h2, h3, .stMarkdown h3 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        letter-spacing: 0;
    }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.3rem !important; }

    /* Expander */
    details summary {
        font-weight: 600 !important;
        color: #7dd3fc !important;
        font-size: 1rem !important;
    }
    details {
        background: rgba(25, 30, 48, 0.5);
        border-radius: 14px;
        padding: 0.5rem;
        border: 1px solid rgba(72, 85, 120, 0.2);
    }

    /* About Page Box - Neutral Dark Color */
    .about-box {
        background: rgba(25, 30, 48, 0.6);
        border: 1px solid rgba(72, 85, 120, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Table Styles */
    table {
        background: rgba(18, 22, 35, 0.6);
        border-radius: 12px;
        overflow: hidden;
    }
    th {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: white !important;
        padding: 12px 15px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    td {
        color: #cbd5e1 !important;
        padding: 10px 15px;
        border-bottom: 1px solid rgba(72, 85, 120, 0.2);
    }

    /* Model Performance Dashboard */
    .best-model-banner {
        background: linear-gradient(135deg, rgba(21, 128, 61, 0.18), rgba(15, 118, 110, 0.12));
        border: 1px solid rgba(74, 222, 128, 0.35);
        border-radius: 8px;
        padding: 1.15rem 1.4rem;
        margin: 1.1rem 0 1.8rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .best-model-icon { font-size: 2rem; line-height: 1; }
    .best-model-label {
        color: #86efac;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .best-model-name {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .best-model-note, .section-note {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .section-note {
        color: #94a3b8;
        margin: -0.35rem 0 1rem;
    }
    .insight-box {
        background: rgba(30, 41, 59, 0.72);
        border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        color: #dbeafe;
        font-size: 0.92rem;
        line-height: 1.55;
        padding: 0.9rem 1rem;
        margin: 0.8rem 0 1.2rem;
    }
    .error-note { border-left-color: #f87171; }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 0.85rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
    }
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
        font-weight: 600;
        padding: 0.7rem 1rem;
    }
    .stTabs [aria-selected="true"] { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# Download required NLTK data
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4',
                'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        nltk.download(pkg, quiet=True)

download_nltk()

# Load saved model files
@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    try:
        model = joblib.load('frd_model/best_model.pkl')
        tfidf = joblib.load('frd_model/tfidf.pkl')
        ohe = joblib.load('frd_model/ohe.pkl')
        rating_scaler = joblib.load('frd_model/rating_scaler.pkl')
        meta_scaler = joblib.load('frd_model/meta_scaler.pkl')
        le = joblib.load('frd_model/label_encoder.pkl')
        category_avg = joblib.load('frd_model/category_avg_map.pkl')
        meta_features = joblib.load('frd_model/meta_features.pkl')
        return model, tfidf, ohe, rating_scaler, meta_scaler, le, category_avg, meta_features
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first.")
        st.stop()

@st.cache_resource(show_spinner="Loading language model...")
def load_gpt2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)
    model.eval()
    if device.type == 'cuda':
        model = model.half()
    return tok, model, device

@st.cache_resource
def load_metrics():
    metrics_dir = 'frd_model/metrics'
    if os.path.exists(metrics_dir):
        results = pd.read_csv(f'{metrics_dir}/all_results.csv') if os.path.exists(f'{metrics_dir}/all_results.csv') else None
        best_model = pd.read_csv(f'{metrics_dir}/best_model_info.csv').iloc[0] if os.path.exists(f'{metrics_dir}/best_model_info.csv') else None
        cm = pd.read_csv(f'{metrics_dir}/confusion_matrix.csv', index_col=0) if os.path.exists(f'{metrics_dir}/confusion_matrix.csv') else None
        return results, best_model, cm
    return None, None, None

# Load model components
model, tfidf, ohe, rating_scaler, meta_scaler, le, category_avg, meta_features = load_model()
gpt2_tok, gpt2_model, gpt2_device = load_gpt2()
sia = SentimentIntensityAnalyzer()
metrics_results, best_model_info, confusion_matrix_data = load_metrics()

# Format model names
def display_model_name(raw_name):
    if raw_name is None:
        return raw_name
    name = str(raw_name)
    abbrev = {'SVC': 'SVM', 'RF': 'Random Forest', 'LR': 'Logistic Regression'}
    for short, full in abbrev.items():
        if name.startswith(short + ' ') or name == short:
            name = full + name[len(short):]
            break

    name = name.replace('[', '— ').replace(']', '')
    return name

# Calculate the confidence for the predicted class
def get_prediction_confidence(model, X, prediction):
    if hasattr(model, 'predict_proba'):
        proba = np.asarray(model.predict_proba(X)[0], dtype=float)
        proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)

        if proba.size:
            total = proba.sum()
            if total > 0 and (proba.max() > 1.0 or total > 1.5):
                proba = proba / total

            classes = list(getattr(model, 'classes_', []))
            if prediction in classes:
                confidence = proba[classes.index(prediction)]
            else:
                confidence = proba.max()

            return float(np.clip(confidence, 0.0, 1.0))

    return 0.95

# Text preprocessing and feature extraction
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

@st.cache_data(ttl=3600, max_entries=100)
def compute_perplexity(text):
    try:
        enc = gpt2_tok(str(text), return_tensors='pt', truncation=True, max_length=512)
        enc = {k: v.to(gpt2_device) for k, v in enc.items()}
        with torch.no_grad():
            loss = gpt2_model(**enc, labels=enc['input_ids']).loss
        return min(torch.exp(loss).item(), 387.265358)
    except:
        return 100.0

def adj_frequency(text):
    tokens = word_tokenize(str(text).lower())
    tags = pos_tag(tokens)
    total = len(tokens)
    adjs = sum(1 for _, tag in tags if tag.startswith('JJ'))
    return adjs / (total + 1e-5)

def validate_review(text):
    if not text or not text.strip():
        return False, "Please enter a review text."
    words_only = re.sub(r'[^a-zA-Z\s]', '', text.strip()).split()
    if len(words_only) == 0:
        return False, "Please enter a valid review containing actual words. Numbers and symbols alone are not accepted."
    if len(words_only) < 3:
        return False, "Review too short. Please enter at least 3 words."
    return True, ""

# Category mapping
RAW_CATEGORIES = list(ohe.categories_[0])

DISPLAY_NAMES = {
    'Books_5': 'Books',
    'Clothing_Shoes_and_Jewelry_5': 'Clothing, Shoes & Jewellery',
    'Electronics_5': 'Electronics',
    'Home_and_Kitchen_5': 'Home & Kitchen',
    'Kindle_Store_5': 'Kindle Store',
    'Movies_and_TV_5': 'Movies & TV',
    'Pet_Supplies_5': 'Pet Supplies',
    'Sports_and_Outdoors_5': 'Sports & Outdoors',
    'Tools_and_Home_Improvement_5': 'Tools & Home Improvement',
    'Toys_and_Games_5': 'Toys & Games',
}

CATEGORIES = [DISPLAY_NAMES.get(cat, cat.replace('_5', '').replace('_', ' ')) for cat in RAW_CATEGORIES]
CATEGORY_MAP = dict(zip(CATEGORIES, RAW_CATEGORIES))

# Examples used for system demonstration
DEMO_EXAMPLES = {
    "genuine": {
        "text": "For the price I am not sure it can be beat. I have used it during several rounds and it is within a few yards of much higher priced range finders and GPS watches that my friends use. Pin lock will at times lock right on with the same reading each time. Other times, it will take a bit and give me a different reading each time but never more than 3-7 yards of variance. Different readings each time is a pain but I am not that accurate with my irons anyway so it is not a deal breaker for me. As long as this unit lasts I consider this a huge win.",
        "category": "Sports_and_Outdoors_5",
        "rating": 4
    },
    "fake": {
        "text": "We have 2 dogs, these are the best quality food. I will keep feeding them this stuff and they will be fine. The only thing I can say is that if you have a large dog, this is the best food. I will keep feeding them this and they will be fine. My dog loves this bed! I will keep buying it. I have a dog who is obsessed with this food. This has given him a great amount of energy. It's very healthy for him and his digestive system.",
        "category": "Pet_Supplies_5",
        "rating": 3
    }
}

# Session state
for key, default in [('review_text', ''), ('rating', None), ('prediction_result', None), ('use_metadata', True), ('selected_category', 0), ('form_reset_counter', 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    selected = option_menu(
        menu_title="NAVIGATION",
        options=["Fake Review Detection", "Model Performance Dashboard", "About"],
        icons=["search", "bar-chart", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "10px 8px",
                "background-color": "rgba(15, 23, 42, 0.72)",
                "border": "1px solid rgba(148, 163, 184, 0.14)",
                "border-radius": "14px",
            },
            "menu-title": {
                "font-size": "11px",
                "font-weight": "800",
                "color": "#ffffff",
                "letter-spacing": "0.12em",
                "padding": "12px 14px 8px",
            },
            "icon": {"color": "#ffffff", "font-size": "16px"},
            "nav-link": {
                "font-size": "13px",
                "font-weight": "600",
                "color": "#ffffff",
                "padding": "11px 14px",
                "border-radius": "10px",
                "margin": "5px 4px",
                "background-color": "rgba(30, 41, 59, 0.38)",
                "--hover-color": "rgba(255, 75, 75, 0.13)",
            },
            "nav-link-selected": {
                "background-color": "rgba(255, 75, 75, 0.18)",
                "color": "#ffffff",
                "font-weight": "700",
                "border-left": "4px solid #ff4b4b",
            },
        }
    )
    st.markdown("---")
    

# ============================================================================
# PAGE 1: FAKE REVIEW DETECTION
# ============================================================================
if selected == "Fake Review Detection":
    st.markdown(
        '<div class="main-header">'
        '<h1><span class="shield-icon">🛡️</span>Fake Review Detection System</h1>'
        '<p>Hybrid Machine Learning Model for Authenticity Verification</p>'
        '</div>',
        unsafe_allow_html=True
    )

    form_key_suffix = st.session_state.form_reset_counter

    # Load example reviews
    demo_col1, demo_col2 = st.columns([1, 1])
    with demo_col1:
        if st.button("Try Genuine Example", use_container_width=True, key="demo_genuine_button"):
            example = DEMO_EXAMPLES["genuine"]
            next_form_key_suffix = st.session_state.form_reset_counter + 1
            st.session_state.review_text = example["text"]
            st.session_state.use_metadata = True
            st.session_state.selected_category = RAW_CATEGORIES.index(example["category"])
            st.session_state.rating = example["rating"]
            st.session_state[f"rating_input_{next_form_key_suffix}"] = example["rating"] - 1
            st.session_state.prediction_result = None
            st.session_state.form_reset_counter = next_form_key_suffix
            st.rerun()

    with demo_col2:
        if st.button("Try Fake Example", use_container_width=True, key="demo_fake_button"):
            example = DEMO_EXAMPLES["fake"]
            next_form_key_suffix = st.session_state.form_reset_counter + 1
            st.session_state.review_text = example["text"]
            st.session_state.use_metadata = True
            st.session_state.selected_category = RAW_CATEGORIES.index(example["category"])
            st.session_state.rating = example["rating"]
            st.session_state[f"rating_input_{next_form_key_suffix}"] = example["rating"] - 1
            st.session_state.prediction_result = None
            st.session_state.form_reset_counter = next_form_key_suffix
            st.rerun()

    # Review input form
    col1, col2 = st.columns([2, 1])

    with col1:
        review_text = st.text_area(
            "Review Text",
            value=st.session_state.review_text,
            height=200,
            placeholder="Enter the product review text here for analysis...",
            key=f"review_text_input_{form_key_suffix}"
        )

    with col2:
        use_metadata = st.checkbox(
            "Include Category & Rating",
            value=st.session_state.use_metadata,
            key=f"use_metadata_input_{form_key_suffix}"
        )

        rating = st.session_state.rating

        if use_metadata:
            selected_category_display = st.selectbox(
                "Product Category",
                CATEGORIES,
                index=st.session_state.selected_category,
                key=f"category_input_{form_key_suffix}"
            )
            st.session_state.selected_category = CATEGORIES.index(selected_category_display)
            category = CATEGORY_MAP[selected_category_display]

            st.markdown('<div class="rating-label">Rating</div>', unsafe_allow_html=True)
            rating_mapping = ["1 Star", "2 Stars", "3 Stars", "4 Stars", "5 Stars"]

            selected_rating = st.feedback("stars", key=f"rating_input_{form_key_suffix}")
            if selected_rating is not None:
                rating = selected_rating + 1
                st.session_state.rating = rating
                st.caption(f"Selected: {rating_mapping[selected_rating]}")
            else:
                rating = None
                st.session_state.rating = None
                st.caption("Selected: None")
        else:
            category = None
            rating = None

    col_btn1, col_btn2 = st.columns([1, 1])

    with col_btn1:
        detect_clicked = st.button("Detect Review", type="primary", use_container_width=True, key="detect_button")

    with col_btn2:
        clear_clicked = st.button("Clear", use_container_width=True)

    # Process the submitted review
    if detect_clicked:
        st.session_state.review_text = review_text
        st.session_state.use_metadata = use_metadata
        if use_metadata and category:
            st.session_state.selected_category = CATEGORIES.index(selected_category_display)
            st.session_state.rating = rating

        is_valid, error_msg = validate_review(review_text)
        if not is_valid:
            st.error(error_msg)
        elif use_metadata and rating is None:
            st.error("Please select a rating before detecting the review.")
        else:
            with st.spinner("Analyzing review..."):
                try:
                    # Transform the review text into TF-IDF features
                    clean_text = preprocess_text(review_text)
                    X_text = tfidf.transform([clean_text])

                    # Transform category and rating when provided
                    if use_metadata and category:
                        X_cat = ohe.transform([[category]])
                        X_rating = csr_matrix(rating_scaler.transform([[rating]]))
                        cat_avg = category_avg.get(category, 3)
                        rating_dev = rating - cat_avg
                    else:
                        X_cat = csr_matrix((1, sum(len(cats) for cats in ohe.categories_)))
                        X_rating = csr_matrix((1, 1))
                        rating_dev = 0.0

                    # Extract review characteristics
                    words = review_text.split()
                    review_len = len(words)
                    upper_ratio = sum(1 for c in review_text if c.isupper()) / (len(review_text) + 1e-5)
                    lex_div = len(set(review_text.lower().split())) / (len(words) + 1e-5)
                    adj_freq_val = adj_frequency(review_text)
                    sentiment = sia.polarity_scores(review_text)['compound']
                    perplexity = compute_perplexity(review_text)

                    raw_meta = [[review_len, upper_ratio, lex_div, adj_freq_val, sentiment, perplexity, rating_dev]]
                    X_meta = csr_matrix(meta_scaler.transform(raw_meta))

                    # Combine text and metadata features
                    X_hybrid = hstack([X_text, X_cat, X_rating, X_meta])

                    # Generate the prediction and confidence score
                    prediction = model.predict(X_hybrid)[0]
                    label = le.inverse_transform([prediction])[0]
                    confidence = get_prediction_confidence(model, X_hybrid, prediction)

                    st.session_state.prediction_result = {
                        'label': label, 'confidence': confidence,
                        'use_metadata': use_metadata,
                        'features': {
                            'review_length': review_len, 'uppercase_ratio': upper_ratio,
                            'lexical_diversity': lex_div, 'adjective_frequency': adj_freq_val,
                            'sentiment': sentiment, 'perplexity': perplexity, 'rating_deviation': rating_dev
                        }
                    }
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                

    if clear_clicked:
        # Reset all session state values
        st.session_state.review_text = ''
        st.session_state.prediction_result = None
        st.session_state.rating = None
        st.session_state.selected_category = 0
        st.session_state.use_metadata = True
        st.session_state.form_reset_counter += 1
        st.rerun()

    # Display prediction result
    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        label = result['label']
        confidence = result['confidence']

        st.markdown("---")

        if label == 'CG':
            st.markdown(f'''
            <div class="result-card fake-result">
                <div class="result-left">
                    <div class="result-icon">⚠️</div>
                    <div class="result-text">
                        <h2>Fake Review Detected</h2>
                        <p>This review appears to be AI-generated or inauthentic.</p>
                        <p><strong>Confidence Score: {confidence:.1%}</strong></p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="result-card genuine-result">
                <div class="result-left">
                    <div class="result-icon">✅</div>
                    <div class="result-text">
                        <h2>Genuine Review</h2>
                        <p>This review appears to be authentic and human-written.</p>
                        <p><strong>Confidence Score: {confidence:.1%}</strong></p>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        if not result['use_metadata']:
            st.info("Category and rating were not provided. Include them to allow the model to use the complete set of hybrid features.")

        with st.expander("Feature Analysis — Observed Review Signals"):
            f = result['features']

            # Dataset reference ranges
            def signal_status(value, low_limit, high_limit):
                if value < low_limit:
                    return "Low"
                if value > high_limit:
                    return "High"
                return "Typical"

            length_status = signal_status(f['review_length'], 21, 85)
            diversity_status = signal_status(f['lexical_diversity'], 0.688, 0.909)
            perplexity_status = signal_status(f['perplexity'], 16.3, 57.2)
            adjective_status = signal_status(f['adjective_frequency'], 0.068, 0.125)
            uppercase_status = signal_status(f['uppercase_ratio'], 0.019, 0.033)

            if f['sentiment'] < -0.05:
                sentiment_status = "Negative"
            elif f['sentiment'] > 0.05:
                sentiment_status = "Positive"
            else:
                sentiment_status = "Neutral"

            if not result['use_metadata']:
                rating_status = "Not provided"
            elif abs(f['rating_deviation']) <= 0.5:
                rating_status = "Close to average"
            elif f['rating_deviation'] > 0:
                rating_status = "Above average"
            else:
                rating_status = "Below average"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="feature-card"><div class="feat-name">Review Length</div><div class="feat-value">{f['review_length']} words</div><div class="feat-status">{length_status}</div><div class="feat-desc">Represents the amount of text in the review. The typical reference range is 21–85 words.</div></div>
                <div class="feature-card"><div class="feat-name">Lexical Diversity</div><div class="feat-value">{f['lexical_diversity']:.3f}</div><div class="feat-status">{diversity_status}</div><div class="feat-desc">Represents the variety of unique words used. The typical reference range is 0.688–0.909.</div></div>
                <div class="feature-card"><div class="feat-name">GPT-2 Perplexity</div><div class="feat-value">{f['perplexity']:.1f}</div><div class="feat-status">{perplexity_status}</div><div class="feat-desc">Represents how predictable the wording is to GPT-2. Lower values are more predictable and higher values are less predictable.</div></div>
                <div class="feature-card"><div class="feat-name">Adjective Frequency</div><div class="feat-value">{f['adjective_frequency']:.3f}</div><div class="feat-status">{adjective_status}</div><div class="feat-desc">Represents the proportion of descriptive words. The typical reference range is 0.068–0.125.</div></div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="feature-card"><div class="feat-name">Sentiment Score</div><div class="feat-value">{f['sentiment']:.3f}</div><div class="feat-status">{sentiment_status}</div><div class="feat-desc">Represents the review's emotional tone on a scale from −1 (negative) to +1 (positive).</div></div>
                <div class="feature-card"><div class="feat-name">Uppercase Ratio</div><div class="feat-value">{f['uppercase_ratio'] * 100:.1f}%</div><div class="feat-status">{uppercase_status}</div><div class="feat-desc">Represents the percentage of letters written in uppercase. The typical reference range is 1.9%–3.3%.</div></div>
                <div class="feature-card"><div class="feat-name">Rating Deviation</div><div class="feat-value">{f['rating_deviation']:.3f}</div><div class="feat-status">{rating_status}</div><div class="feat-desc">Represents how far the selected rating is above or below the average rating for its product category.</div></div>
                """, unsafe_allow_html=True)

            st.info(
                "Feature statuses show whether each value is low, typical, or high compared "
                "with the project dataset. These values describe review characteristics only. "
                "The final prediction is based on the combined text and metadata features."
            )

# ============================================================================
# PAGE 2: MODEL PERFORMANCE DASHBOARD
# ============================================================================
elif selected == "Model Performance Dashboard":
    st.markdown(
        '<div class="main-header"><h1>&#128202; Model Performance Dashboard</h1>'
        '<p>Test-set evaluation results for the trained fake review detection models.</p></div>',
        unsafe_allow_html=True
    )

    if metrics_results is None or best_model_info is None:
        st.warning("Performance metrics are not available. Please run the training notebook first.")
    else:
        # Load saved evaluation results
        display_df = metrics_results.copy()
        display_df['Model'] = display_df['Model'].apply(display_model_name)
        best_name = display_model_name(best_model_info.get('best_model_name', ''))
        roc_curve_path = 'frd_model/metrics/roc_curve.csv'
        roc_auc_path = 'frd_model/metrics/roc_auc.csv'
        roc_available = os.path.exists(roc_curve_path) and os.path.exists(roc_auc_path)
        auc_score = pd.read_csv(roc_auc_path)['auc_score'].iloc[0] if roc_available else None

        # Selected model performance summary
        summary_columns = st.columns(5)
        summary_metrics = [
            ("Accuracy", best_model_info['best_accuracy']),
            ("Precision", best_model_info['best_precision']),
            ("Recall", best_model_info['best_recall']),
            ("F1 Score", best_model_info['best_f1_score']),
            ("AUC", auc_score)
        ]
        for column, (label, value) in zip(summary_columns, summary_metrics):
            with column:
                st.metric(label, f"{value * 100:.2f}%" if value is not None else "N/A")

        st.markdown(f"""
        <div class="best-model-banner">
            <div class="best-model-icon">&#127942;</div>
            <div>
                <div class="best-model-label">Selected Best Model</div>
                <div class="best-model-name">{best_name}</div>
                <div class="best-model-note">This model achieved the highest F1 score among the tuned hybrid models and is used by the review detection system.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dashboard analysis sections
        comparison_tab, feature_tab, matrix_tab, roc_tab = st.tabs([
            "Model Comparison", "Feature Impact", "Confusion Matrix", "ROC Curve"
        ])

        with comparison_tab:
            # Compare all models using the selected metric
            st.subheader("Model Performance Ranking")
            st.markdown(
                '<p class="section-note">Select a metric to compare all trained models. '
                'A longer bar indicates better performance.</p>',
                unsafe_allow_html=True
            )
            selected_metric = st.selectbox(
                "Performance Metric",
                ["F1 Score", "Accuracy", "Precision", "Recall"],
                key="dashboard_metric"
            )
            ranking_df = metrics_results.copy()
            ranking_df['Group'] = np.select(
                [
                    ranking_df['Model'].str.contains('Hybrid Tuned', na=False),
                    ranking_df['Model'].str.contains('Text-Only Tuned', na=False),
                    ranking_df['Model'].str.contains('Hybrid', na=False),
                    ranking_df['Model'].str.contains('Text-Only', na=False)
                ],
                [
                    'Tuned Hybrid',
                    'Tuned Text-Only',
                    'Baseline Hybrid',
                    'Baseline Text-Only'
                ],
                default='Other'
            )
            group_order = {
                'Tuned Hybrid': 0,
                'Tuned Text-Only': 1,
                'Baseline Hybrid': 2,
                'Baseline Text-Only': 3,
                'Other': 4
            }
            model_order = {
                'Random Forest': 0,
                'SVM': 1,
                'Logistic Regression': 2
            }
            ranking_df['Algorithm'] = ranking_df['Model'].str.split(' [', regex=False).str[0]
            ranking_df['Group Order'] = ranking_df['Group'].map(group_order)
            ranking_df['Model Order'] = ranking_df['Algorithm'].map(model_order).fillna(99)
            ranking_df = ranking_df.sort_values(['Group Order', 'Model Order'])
            ranking_df['Chart Label'] = ranking_df['Group'] + ' | ' + ranking_df['Algorithm']
            ranking_df['Performance'] = ranking_df[selected_metric] * 100
            ranking_df['Result'] = np.where(
                ranking_df['Model'].apply(display_model_name) == best_name,
                'Selected Model',
                'Other Models'
            )
            metric_colors = {
                'Accuracy': '#3b82f6',
                'Precision': '#ec4899',
                'Recall': '#a855f7',
                'F1 Score': '#22c55e'
            }
            fig = px.bar(
                ranking_df,
                x='Performance',
                y='Chart Label',
                orientation='h',
                color='Result',
                color_discrete_map={
                    'Selected Model': '#facc15',
                    'Other Models': metric_colors[selected_metric]
                },
                text='Performance',
                height=560
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', cliponaxis=False)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#cbd5e1',
                xaxis_title=f'{selected_metric} (%)',
                yaxis_title=None,
                legend_title=None,
                legend=dict(orientation='h', y=1.08, x=0),
                margin=dict(l=10, r=55, t=35, b=20)
            )
            fig.update_xaxes(
                range=[max(0, ranking_df['Performance'].min() - 5), 100],
                gridcolor='rgba(148,163,184,0.14)',
                ticksuffix='%'
            )
            fig.update_yaxes(
                showgrid=False,
                categoryorder='array',
                categoryarray=list(reversed(ranking_df['Chart Label'].tolist()))
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f'<div class="insight-box"><strong>Key finding:</strong> {best_name} has the '
                f'highest F1 score at {best_model_info["best_f1_score"] * 100:.2f}%.</div>',
                unsafe_allow_html=True
            )

        with feature_tab:
            # Compare tuned text-only and hybrid models
            st.subheader("Text-Only vs Hybrid Features")
            st.markdown(
                '<p class="section-note">Text-only models use TF-IDF features extracted from '
                'the review text. Hybrid models combine TF-IDF text features with category, '
                'rating and engineered review signals.</p>',
                unsafe_allow_html=True
            )
            tuned_results = metrics_results[
                metrics_results['Model'].str.contains('Tuned', na=False)
            ]
            text_models = tuned_results[
                tuned_results['Model'].str.contains('Text-Only', na=False)
            ]
            hybrid_models = tuned_results[
                tuned_results['Model'].str.contains('Hybrid', na=False)
            ]

            if len(text_models) > 0 and len(hybrid_models) > 0:
                text_f1 = text_models['F1 Score'].mean()
                hybrid_f1 = hybrid_models['F1 Score'].mean()
                feature_improvement = ((hybrid_f1 - text_f1) / text_f1) * 100
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Average Text-Only F1", f"{text_f1 * 100:.2f}%")
                with c2:
                    st.metric("Average Hybrid F1", f"{hybrid_f1 * 100:.2f}%")
                with c3:
                    st.metric("Relative Improvement", f"+{feature_improvement:.2f}%")

                feature_df = pd.DataFrame({
                    'Feature Set': ['Text-Only', 'Hybrid'],
                    'Average F1': [text_f1 * 100, hybrid_f1 * 100]
                })
                fig = px.bar(
                    feature_df,
                    x='Feature Set',
                    y='Average F1',
                    color='Feature Set',
                    color_discrete_map={'Text-Only': '#f59e0b', 'Hybrid': '#22c55e'},
                    text='Average F1',
                    height=390
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside', width=0.45)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1',
                    showlegend=False,
                    xaxis_title=None,
                    yaxis_title='Average F1 Score (%)',
                    margin=dict(t=30, b=20)
                )
                fig.update_yaxes(
                    range=[max(0, feature_df['Average F1'].min() - 8), 100],
                    gridcolor='rgba(148,163,184,0.14)',
                    ticksuffix='%'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    '<div class="insight-box"><strong>Interpretation:</strong> Across the '
                    'three tuned algorithms, the hybrid feature set achieved a higher average F1 '
                    'score than the text-only feature set. This indicates that category, rating '
                    'and engineered review signals provide useful information alongside the '
                    'review text.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Text-only and hybrid results are required for this comparison.")

            # Compare baseline and tuned models
            st.subheader("Effect of Hyperparameter Tuning")
            st.markdown(
                '<p class="section-note">Each tuned model is compared directly with its own '
                'baseline version.</p>',
                unsafe_allow_html=True
            )
            baseline = metrics_results[
                ~metrics_results['Model'].str.contains('Tuned', na=False)
            ].copy()
            tuned = metrics_results[
                metrics_results['Model'].str.contains('Tuned', na=False)
            ].copy()
            baseline['Pair'] = baseline['Model']
            tuned['Pair'] = tuned['Model'].str.replace(' Tuned]', ']', regex=False)
            tuning_df = baseline[['Pair', 'F1 Score']].merge(
                tuned[['Pair', 'F1 Score']],
                on='Pair',
                suffixes=(' Baseline', ' Tuned')
            )

            if not tuning_df.empty:
                tuning_df['Model'] = tuning_df['Pair'].apply(display_model_name)
                tuning_long = tuning_df.melt(
                    id_vars='Model',
                    value_vars=['F1 Score Baseline', 'F1 Score Tuned'],
                    var_name='Version',
                    value_name='F1 Score'
                )
                tuning_long['Version'] = tuning_long['Version'].str.replace(
                    'F1 Score ', '', regex=False
                )
                tuning_long['F1 Score'] *= 100
                fig = px.bar(
                    tuning_long,
                    x='Model',
                    y='F1 Score',
                    color='Version',
                    barmode='group',
                    color_discrete_map={'Baseline': '#3b82f6', 'Tuned': '#22c55e'},
                    height=440
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1',
                    xaxis_title=None,
                    yaxis_title='F1 Score (%)',
                    legend_title=None,
                    legend=dict(orientation='h', y=1.08, x=0),
                    margin=dict(t=35, b=90)
                )
                fig.update_xaxes(tickangle=-25, showgrid=False)
                fig.update_yaxes(
                    range=[max(0, tuning_long['F1 Score'].min() - 6), 100],
                    gridcolor='rgba(148,163,184,0.14)',
                    ticksuffix='%'
                )
                st.plotly_chart(fig, use_container_width=True)

                tuning_df['Improvement'] = (
                    tuning_df['F1 Score Tuned'] - tuning_df['F1 Score Baseline']
                ) * 100
                improved_count = int((tuning_df['Improvement'] > 0).sum())
                best_gain = tuning_df.loc[tuning_df['Improvement'].idxmax()]
                st.markdown(
                    f'<div class="insight-box"><strong>Interpretation:</strong> Hyperparameter '
                    f'tuning improved {improved_count} of the {len(tuning_df)} model '
                    f'configurations and maintained the performance of the remaining models. '
                    f'The largest improvement was achieved by {best_gain["Model"]}, with an '
                    f'F1-score increase of {best_gain["Improvement"]:.2f} % points.</div>',
                    unsafe_allow_html=True
                )

        with matrix_tab:
            # Display correct and incorrect test predictions
            st.subheader("Confusion Matrix")
            st.markdown(
                '<p class="section-note">This shows how many fake and genuine reviews were '
                'classified correctly or incorrectly by the selected model.</p>',
                unsafe_allow_html=True
            )
            if confusion_matrix_data is not None:
                cm_values = confusion_matrix_data.values
                cm_display = pd.DataFrame(
                    [[cm_values[1, 1], cm_values[1, 0]],
                     [cm_values[0, 1], cm_values[0, 0]]],
                    index=['Actual Fake', 'Actual Genuine'],
                    columns=['Predicted Fake', 'Predicted Genuine']
                )
                fig = px.imshow(
                    cm_display.values,
                    x=cm_display.columns,
                    y=cm_display.index,
                    text_auto=',d',
                    aspect='auto',
                    color_continuous_scale=[
                        [0, '#f3e8ff'], [0.5, '#c084fc'], [1, '#f59e0b']
                    ]
                )
                fig.update_layout(
                    height=410,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1',
                    margin=dict(t=20, b=20)
                )
                fig.update_xaxes(side='bottom')
                fig.update_traces(textfont_size=18, textfont_color='#111827')
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)

                tp, fn = cm_values[1, 1], cm_values[1, 0]
                fp, tn = cm_values[0, 1], cm_values[0, 0]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("True Positive (TP)", f"{tp:,}", help="Fake reviews detected as fake")
                with c2:
                    st.metric("True Negative (TN)", f"{tn:,}", help="Genuine reviews detected as genuine")
                with c3:
                    st.metric("False Positive (FP)", f"{fp:,}", help="Genuine reviews predicted as fake")
                with c4:
                    st.metric("False Negative (FN)", f"{fn:,}", help="Fake reviews predicted as genuine")
                st.markdown(
                    f'<div class="insight-box"><strong>Interpretation:</strong> The model '
                    f'correctly identified {tp:,} fake reviews and {tn:,} genuine reviews. '
                    f'It missed {fn:,} fake reviews and incorrectly flagged {fp:,} genuine '
                    f'reviews as fake.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("Confusion matrix data is not available.")

        with roc_tab:
            # Display the ROC curve for the selected model
            st.subheader("ROC Curve")
            st.markdown(
                '<p class="section-note">The ROC curve measures how well the selected model '
                'separates fake reviews from genuine reviews across different thresholds.</p>',
                unsafe_allow_html=True
            )
            if roc_available:
                roc_df = pd.read_csv(roc_curve_path)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=roc_df['fpr'],
                    y=roc_df['tpr'],
                    mode='lines',
                    name=f'Selected model (AUC = {auc_score:.3f})',
                    line=dict(color='#facc15', width=4)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random classifier',
                    line=dict(color='#94a3b8', width=2, dash='dash')
                ))
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#cbd5e1',
                    legend=dict(orientation='h', y=1.08, x=0, bgcolor='rgba(0,0,0,0)'),
                    xaxis=dict(gridcolor='rgba(148,163,184,0.14)', range=[0, 1], dtick=0.2),
                    yaxis=dict(gridcolor='rgba(148,163,184,0.14)', range=[0, 1], dtick=0.2)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f'<div class="insight-box"><strong>Interpretation:</strong> The selected '
                    f'model achieved an AUC of {auc_score * 100:.1f}%, indicating excellent '
                    'ability to distinguish fake reviews from genuine reviews across different '
                    'classification thresholds. This is well above the approximately 50% '
                    'performance of a random classifier.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.info("ROC curve data is not available. Please re-run the training notebook.")

# ============================================================================
# PAGE 3: ABOUT
# ============================================================================
else:
    st.markdown('<div class="main-header"><h1>ℹ️ About This Project</h1></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <h3 style="color: #e2e8f0; margin-top: 0;">Final Year Project: Fake Review Detection using Machine Learning with Hybrid Text and Metadata Features</h3>
    <p><strong>Aim:</strong> To develop a practical and lightweight fake review detection system that combines text and metadata features for real-time detection.</p>
    </div>
    
    <div class="about-box">
    <h3 style="color: #e2e8f0;">Dataset Information</h3>
    <table>
        <tr><th>Aspect</th><th>Details</th></tr>
        <tr><td>Source</td><td>Kaggle: Fake Reviews Dataset</td></tr>
        <tr><td>Total Data</td><td>40,432 reviews</td></tr>
        <tr><td>Fake Reviews (CG)</td><td>20,216 (50%)</td></tr>
        <tr><td>Genuine Reviews (OR)</td><td>20,216 (50%)</td></tr>
        <tr><td>Categories</td><td>10 product categories</td></tr>
        <tr><td>Train/Test Split</td><td>80% / 20%</td></tr>
    </table>
    </div>
    
    <div class="about-box">
    <h3 style="color: #e2e8f0;">Methodology</h3>
    <table>
        <tr><th>Component</th><th>Description</th></tr>
        <tr><td>Text Features</td><td>TF-IDF vectorization</td></tr>
        <tr><td>Metadata Features</td><td>Product category, rating, review length, uppercase ratio, lexical diversity, adjective frequency, sentiment score, GPT-2 perplexity, and rating deviation</td></tr>
        <tr><td>Algorithms</td><td>Support Vector Machine (SVM), Random Forest (RF), Logistic Regression (LR)</td></tr>
        <tr><td>Optimization</td><td>GridSearchCV with stratified 5-fold cross-validation</td></tr>
    </table>
    </div>
    """, unsafe_allow_html=True)

    if best_model_info is not None:
        best_name = display_model_name(best_model_info.get('best_model_name', ''))
        st.markdown(f"""
        <div class="about-box">
        <h3 style="color: #e2e8f0;">Results Summary</h3>
        <ul>
            <li><strong>Best Model:</strong> {best_name}</li>
            <li><strong>Accuracy:</strong> {best_model_info['best_accuracy']:.4f}</li>
            <li><strong>F1 Score:</strong> {best_model_info['best_f1_score']:.4f}</li>
            <li><strong>Precision:</strong> {best_model_info['best_precision']:.4f}</li>
            <li><strong>Recall:</strong> {best_model_info['best_recall']:.4f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <h3 style="color: #e2e8f0;">Key Findings</h3>
    <ol>
        <li>Hyperparameter tuning improved model performance.</li>
        <li>Hybrid text-metadata models outperformed text-only models.</li>
        <li>Tuned hybrid Random Forest was the best-performing model.</li>
        <li>A Streamlit web-based prototype was successfully developed.</li>
    </ol>
    </div>
    
    <div class="about-box">
    <h3 style="color: #e2e8f0;">Technologies Used</h3>
    <ul>
        <li><strong>Python</strong> - Core programming language</li>
        <li><strong>Scikit-learn</strong> - Machine learning models and preprocessing</li>
        <li><strong>PyTorch &amp; Transformers</strong> - GPT-2 perplexity computation</li>
        <li><strong>NLTK &amp; VADER</strong> - Natural language processing and sentiment analysis</li>
        <li><strong>Streamlit</strong> - Web application framework</li>
        <li><strong>Plotly</strong> - Interactive data visualizations</li>
        <li><strong>GPU (RTX 4050)</strong> - Hardware acceleration for GPT-2 perplexity computation</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
