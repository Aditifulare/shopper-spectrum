import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Custom CSS (Light Sky Blue Theme) ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background-color: #f7fafc;
        color: #1e293b;
    }

    section[data-testid="stSidebar"] {
        background-color: #eaf4fb;
        border-right: 1px solid #bfe3f5;
    }

    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
        font-size: 16px;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #1e293b !important;
    }

    h1 {
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    p, label, .stMarkdown, .stMarkdown p {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] span {
        color: #1e293b !important;
    }

    [data-testid="stMarkdownContainer"] p {
        color: #1e293b;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #38bdf8, #0ea5e9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #0ea5e9, #0284c7);
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4);
        transform: translateY(-1px);
    }

    .stTextInput input, .stNumberInput input {
        background-color: #ffffff;
        color: #1e293b;
        border: 1px solid #bfe3f5;
        border-radius: 6px;
    }

    .segment-card {
        background: linear-gradient(135deg, #ffffff, #eaf4fb);
        border: 1px solid #bfe3f5;
        border-left: 4px solid #38bdf8;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.08);
    }

    .product-card {
        background: linear-gradient(135deg, #ffffff, #eaf4fb);
        border: 1px solid #bfe3f5;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        font-family: 'Inter', sans-serif;

        color: #1e293b;
        border-left: 3px solid #38bdf8;
    }

    .metric-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #bfe3f5;
        box-shadow: 0 2px 8px rgba(56, 189, 248, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------- Load Models (cached) ----------------------
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("rfm_scaler.pkl")
    label_map = joblib.load("cluster_label_map.pkl")
    similarity_df = joblib.load("product_similarity.pkl")
    return kmeans, scaler, label_map, similarity_df


try:
    kmeans, scaler, label_map, similarity_df = load_artifacts()
    products_list = sorted(similarity_df.index.tolist())
    models_loaded = True
except FileNotFoundError as e:
    models_loaded = False
    load_error = str(e)


# Segment descriptions for context
SEGMENT_INFO = {
    "High-Value": {
        "emoji": "💎",
        "desc": "Recent, frequent, and big-spending customers. Your best customers — reward and retain them.",
        "color": "#22c55e"
    },
    "Regular": {
        "emoji": "🔵",
        "desc": "Steady purchasers with moderate frequency and spend. Reliable but not premium.",
        "color": "#3b82f6"
    },
    "Occasional": {
        "emoji": "🟠",
        "desc": "Rare, occasional purchasers with lower spend. Potential to nurture into Regular customers.",
        "color": "#f97316"
    },
    "At-Risk": {
        "emoji": "🔴",
        "desc": "Haven't purchased in a long time. Target with re-engagement and win-back campaigns.",
        "color": "#ef4444"
    }
}


# ---------------------- Sidebar Navigation ----------------------
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Clustering", "🔁 Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("E-Commerce Customer Segmentation & Product Recommendation System")


# ---------------------- Home Page ----------------------
if page == "🏠 Home":
    st.title("Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendations in E-Commerce")

    st.markdown("""
    Welcome! This app uses transaction data from an online retail business to:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="segment-card">
        <h3>📊 Customer Segmentation</h3>
        <p>Enter a customer's <b>Recency</b>, <b>Frequency</b>, and <b>Monetary</b> values
        to predict which segment they belong to — High-Value, Regular, Occasional, or At-Risk —
        using RFM analysis and KMeans clustering.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="segment-card">
        <h3>🔁 Product Recommendation</h3>
        <p>Enter a product name to get the top 5 similar products, based on
        item-based collaborative filtering using cosine similarity on
        customer purchase history.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Customer Segments")

    cols = st.columns(4)
    for col, (seg, info) in zip(cols, SEGMENT_INFO.items()):
        with col:
            st.markdown(f"""
            <div class="metric-box">
            <div style="font-size:28px;">{info['emoji']}</div>
            <div style="font-weight:600; color:{info['color']}; margin-top:4px;">{seg}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("👈 Use the sidebar to navigate to Clustering or Recommendation modules.")


# ---------------------- Clustering Page ----------------------
elif page == "📊 Clustering":
    st.title("Customer Segmentation")
    st.markdown("Enter customer purchase behavior to predict their segment.")

    if not models_loaded:
        st.error(f"Could not load model files: {load_error}")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)",
            min_value=0, max_value=1000, value=30, step=1
        )
        frequency = st.number_input(
            "Frequency (number of purchases)",
            min_value=1, max_value=1000, value=5, step=1
        )
        monetary = st.number_input(
            "Monetary (total spend, £)",
            min_value=0.0, max_value=1000000.0, value=500.0, step=10.0
        )

        predict_btn = st.button("Predict Segment", use_container_width=True)

    with col2:
        if predict_btn:
            # Apply the same transformation pipeline used during training: log1p -> scale
            input_df = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            })
            input_log = input_df.apply(lambda x: np.log1p(x))
            input_scaled = scaler.transform(input_log)
            input_scaled_df = pd.DataFrame(input_scaled, columns=["Recency", "Frequency", "Monetary"])

            cluster = kmeans.predict(input_scaled_df)[0]
            segment = label_map[cluster]
            info = SEGMENT_INFO[segment]

            st.markdown(f"""
            <div class="segment-card" style="border-left-color:{info['color']};">
            <div style="font-size:40px;">{info['emoji']}</div>
            <h2 style="margin-top:0;">{segment} Customer</h2>
            <p>{info['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="segment-card">
            <p style="color:#64748b;">Enter values and click <b>Predict Segment</b> to see the result.</p>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("ℹ️ How segments are defined"):
        st.markdown("""
        | Segment | Characteristics |
        |---|---|
        | 💎 **High-Value** | Recent, frequent, and big spenders |
        | 🔵 **Regular** | Steady purchasers but not premium |
        | 🟠 **Occasional** | Rare, occasional purchases |
        | 🔴 **At-Risk** | Haven't purchased in a long time |
        """)


# ---------------------- Recommendation Page ----------------------
elif page == "🔁 Recommendation":
    st.title("Product Recommender")
    st.markdown("Enter a product name to get 5 similar product recommendations.")

    if not models_loaded:
        st.error(f"Could not load model files: {load_error}")
        st.stop()

    product_input = st.selectbox(
        "Select or search for a product",
        options=products_list,
        index=None,
        placeholder="Start typing a product name..."
    )

    recommend_btn = st.button("Recommend", use_container_width=False)

    if recommend_btn:
        if not product_input:
            st.warning("Please select a product first.")
        else:
            similar_scores = similarity_df[product_input].sort_values(ascending=False)
            similar_scores = similar_scores.drop(product_input)
            top_5 = similar_scores.head(5)

            st.markdown(f"#### Products similar to: *{product_input}*")
            for i, (prod, score) in enumerate(top_5.items(), 1):
                st.markdown(f"""
                <div class="product-card">
                <b>{i}. {prod}</b>
                <span style="color:#0ea5e9; float:right;">{score:.2f} similarity</span>
                </div>
                """, unsafe_allow_html=True)