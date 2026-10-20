from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"
model = joblib.load(MODEL_DIR / "Random_Forest_House_Price.pkl")
columns = joblib.load(MODEL_DIR / "columns.pkl")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Overall page */
    .main {
        padding-top: 1.5rem;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        color: white;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.2rem;
        margin-bottom: 0.3rem;
        color: white;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.95;
        margin: 0;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1f4037;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid #1f4037;
        padding-left: 0.6rem;
    }

    /* Card container look for input blocks */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }

    /* Predict button */
    div[data-testid="stFormSubmitButton"] {
        margin-top: 0.8rem;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
    width: 220px;
    background: linear-gradient(135deg, #1f4037 0%, #2c7a5b 100%);
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 0.7rem 0;
    border-radius: 10px;
    border: none;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    display: block;
    margin: 0 auto;
}
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(31, 64, 55, 0.35);
        color: white;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #e9fbf1 0%, #d4f5e3 100%);
        border: 1px solid #a9e8c4;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.2rem;
    }
    .result-card .label {
        font-size: 1rem;
        color: #2c7a5b;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .result-card .price {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1f4037;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Prediction</h1>
    <p>Fill in the property details and get an instant AI-powered price estimate</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# INPUT FORM
# ============================================================
with st.form("prediction_form"):

    st.markdown('<div class="section-title">📐 Property Size</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            area = st.number_input(
                "Area (sq.ft)", min_value=0, max_value=20000, value=5000, step=50
            )
            stories = st.select_slider("Stories", options=[1, 2, 3, 4], value=1)
        with c2:
            bedrooms = st.select_slider("Bedrooms", options=[1, 2, 3, 4, 5, 6], value=3)
            bathrooms = st.select_slider("Bathrooms", options=[1, 2, 3, 4, 5], value=2)

    st.markdown('<div class="section-title">🚗 Parking & Furnishing</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            parking = st.select_slider("Parking Spaces", options=[0, 1, 2, 3], value=1)
        with c2:
            furnishingstatus = st.radio(
                "Furnishing Status", ["furnished", "unfurnished"], horizontal=True
            )

    st.markdown('<div class="section-title">✨ Amenities</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            mainroad = "yes" if st.checkbox("Main Road", value=True) else "no"
        with c2:
            guestroom = "yes" if st.checkbox("Guest Room") else "no"
        with c3:
            basement = "yes" if st.checkbox("Basement") else "no"
        with c4:
            airconditioning = "yes" if st.checkbox("Air Conditioning") else "no"
        with c5:
            prefarea = "yes" if st.checkbox("Preferred Area") else "no"

    submitted = st.form_submit_button("🔮 Predict Price")

# ============================================================
# PREDICTION
# ============================================================
if submitted:
    # Area must be a realistic minimum before we even attempt a prediction
    MIN_VALID_AREA = 200

    if area < MIN_VALID_AREA:
        st.error(
            f"⚠️ Area value ({area:,} sq.ft) is too low. Please enter an area "
            f"of at least {MIN_VALID_AREA} sq.ft.",
            icon="🚨",
        )
        st.stop()

    with st.spinner("Crunching the numbers..."):
        data = {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "parking": [parking],
            "mainroad": [mainroad],
            "guestroom": [guestroom],
            "basement": [basement],
            "airconditioning": [airconditioning],
            "prefarea": [prefarea],
            "furnishingstatus": [furnishingstatus],
        }

        df = pd.DataFrame(data)
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        raw_prediction = model.predict(df)[0]

    # The model can occasionally extrapolate to unrealistic values
    # (including negative numbers) for unusual input combinations.
    # Anything below this floor is treated as an invalid estimate.
    MIN_VALID_PRICE = 200

    if raw_prediction < MIN_VALID_PRICE:
        st.error(
            f"⚠️ The estimated price (₹ {raw_prediction:,.0f}) is below the minimum "
            f"valid threshold of ₹ {MIN_VALID_PRICE:,}. This usually happens with "
            "unusual input combinations (e.g. very small area with many "
            "bedrooms/stories). Please adjust the details and try again.",
            icon="🚨",
        )
    else:
        prediction = raw_prediction
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated House Price</div>
            <div class="price">₹ {prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("📋 View Submitted Details"):
        summary = pd.DataFrame({
            "Feature": ["Area (sq.ft)", "Bedrooms", "Bathrooms", "Stories", "Parking",
                        "Main Road", "Guest Room", "Basement", "Air Conditioning",
                        "Preferred Area", "Furnishing"],
            "Value": [area, bedrooms, bathrooms, stories, parking,
                      mainroad, guestroom, basement, airconditioning,
                      prefarea, furnishingstatus],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)