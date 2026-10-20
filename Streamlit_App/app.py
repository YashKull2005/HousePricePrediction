import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="House Price Prediction", layout="wide")

model_path = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_model():
    return joblib.load(model_path / "Random_Forest_House_Price.pkl")

model = load_model()

st.title("House Price Prediction")
st.write("Enter the house details below to get an estimated price.")

st.subheader("Property Details")

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    age = st.number_input("Age (years)", min_value=0, max_value=126, value=30, step=1)
with col2:
    sqft_living = st.number_input("Living Area (sq.ft)", min_value=0, max_value=20000, value=2000, step=50)
    sqft_lot = st.number_input("Lot Size (sq.ft)", min_value=0, max_value=1000000, value=8000, step=100)
    floors = st.selectbox("Floors", [1, 2, 3], index=0)
with col3:
    waterfront = st.selectbox("Waterfront", ["No", "Yes"], index=0)
    view = st.selectbox("View Rating", [0, 1, 2, 3, 4], index=0)
    grade = st.selectbox("Build Quality Grade", list(range(1, 14)), index=7)

st.subheader("Neighborhood & Status")

col4, col5, col6 = st.columns(3)
with col4:
    sqft_living15 = st.number_input("Avg Neighbor Living Area (sq.ft)", min_value=0, max_value=20000, value=2000, step=50)
with col5:
    sqft_lot15 = st.number_input("Avg Neighbor Lot Size (sq.ft)", min_value=0, max_value=1000000, value=8000, step=100)
with col6:
    is_renovated = st.selectbox("Renovated", ["No", "Yes"], index=0)

if st.button("Predict Price"):
    if sqft_living < 200:
        st.error("Living area is too low. Please enter at least 200 sq.ft.")
    else:
        data = pd.DataFrame([{
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Sqft_living": sqft_living,
            "Sqft_lot": sqft_lot,
            "Floors": floors,
            "Waterfront": 1 if waterfront == "Yes" else 0,
            "View": view,
            "Grade": grade,
            "Sqft_living15": sqft_living15,
            "Sqft_lot15": sqft_lot15,
            "Is_renovated": 1 if is_renovated == "Yes" else 0,
            "Age": age,
        }])

        prediction = model.predict(data)[0]

        if prediction < 200:
            st.error("Could not get a valid estimate for these inputs. Try adjusting the details.")
        else:
            st.success(f"Estimated Price: ${prediction:,.0f}")

        with st.expander("View Submitted Details"):
            st.dataframe(data.T.rename(columns={0: "Value"}), hide_index=True)
