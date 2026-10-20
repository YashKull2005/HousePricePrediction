import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------- Page setup ----------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Random_Forest_House_Price.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------- Header ----------
st.title("🏠 House Price Prediction")
st.caption("Fill in the details below and get an estimated sale price.")

# ---------- Input form ----------
# A form means the page only reruns when you press the button.
with st.form("house_form"):

    # --- Size & layout ---
    with st.container(border=True):
        st.markdown("#### 🛏️ Size & Layout")
        c1, c2, c3 = st.columns(3)
        bedrooms = c1.number_input("Bedrooms", 0, 10, 3)
        bathrooms = c2.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.5)
        floors = c3.selectbox("Floors", [1, 2, 3])

        c4, c5 = st.columns(2)
        sqft_living = c4.number_input("Living area (sq.ft)", 0, 20000, 2000, step=50)
        sqft_lot = c5.number_input("Lot size (sq.ft)", 0, 1000000, 8000, step=100)

    # --- Quality & condition ---
    with st.container(border=True):
        st.markdown("#### ⭐ Quality & Condition")
        c1, c2, c3 = st.columns(3)
        grade = c1.slider("Build quality (1–13)", 1, 13, 8)
        view = c2.slider("View rating (0–4)", 0, 4, 0)
        age = c3.number_input("Age (years)", 0, 126, 30)

        c4, c5 = st.columns(2)
        waterfront = c4.toggle("Waterfront")
        renovated = c5.toggle("Renovated")

    # --- Neighborhood ---
    with st.container(border=True):
        st.markdown("#### 🏘️ Neighborhood")
        c1, c2 = st.columns(2)
        sqft_living15 = c1.number_input("Avg neighbor living area (sq.ft)", 0, 20000, 2000, step=50)
        sqft_lot15 = c2.number_input("Avg neighbor lot size (sq.ft)", 0, 1000000, 8000, step=100)

    submitted = st.form_submit_button("Predict Price", type="primary", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    if sqft_living < 200:
        st.error("Living area is too small. Please enter at least 200 sq.ft.")
        st.stop()

    data = pd.DataFrame([{
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Sqft_living": sqft_living,
        "Sqft_lot": sqft_lot,
        "Floors": floors,
        "Waterfront": int(waterfront),
        "View": view,
        "Grade": grade,
        "Sqft_living15": sqft_living15,
        "Sqft_lot15": sqft_lot15,
        "Is_renovated": int(renovated),
        "Age": age,
    }])

    price = model.predict(data)[0]

    if price < 200:
        st.error("Could not get a valid estimate. Try adjusting the details.")
    else:
        st.divider()
        with st.container(border=True):
            st.metric("Estimated Price", f"${price:,.0f}")
            st.caption(f"≈ ${price / sqft_living:,.0f} per sq.ft of living area")

        with st.expander("View submitted details"):
            st.dataframe(data.T.rename(columns={0: "Value"}), use_container_width=True)