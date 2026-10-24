"""Sri Lanka used-car price predictor.   Run:  streamlit run app.py"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from catboost import Pool

from car_utils import CAT_FEATURES, FEATURES, fill_missing, make_input_row

st.set_page_config(page_title="SL Car Price Predictor", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


bundle = load_model()
model, stats = bundle["model"], bundle["fill_stats"]

st.title("🚗 Sri Lanka used-car price predictor")
st.caption(f"{bundle['model_name']} model trained on ikman.lk / Kaggle listings · test R² {bundle['test_r2']:.2f} · "
           f"typical error ±{bundle['test_mape']:.0f}% · prices in LKR lakhs (1 lakh = Rs. 100,000)")

with st.sidebar:
    st.header("Car details")
    brands = sorted(bundle["brand_models"])
    brand = st.selectbox("Brand", brands, index=brands.index("TOYOTA"))
    models = bundle["brand_models"][brand]
    car_model = st.selectbox("Model", models, index=models.index("AQUA") if "AQUA" in models else 0)
    year = st.number_input("Year of manufacture", 1990, 2026, 2014)
    engine = st.number_input("Engine (cc)", 600, 7000, 1500, step=50)
    gear = st.selectbox("Gearbox", ["AUTOMATIC", "MANUAL"])
    fuel = st.selectbox("Fuel", ["HYBRID", "PETROL", "DIESEL"])
    condition = st.selectbox("Condition", ["USED", "NEW"])
    mileage = st.number_input("Mileage (km), 0 if unknown", 0, 800_000, 124_000, step=1000)
    go = st.button("Predict price", type="primary", use_container_width=True)

if not go:
    st.info("Enter the car details on the left and press **Predict price**.")
    st.stop()

# 1. user input -> one row -> fill missing values exactly as in training
row = fill_missing(make_input_row(brand, car_model, year, engine, gear, fuel, condition, mileage or None), stats)
# 2. predict (the model predicts log(price), so convert back)
price = float(np.expm1(model.predict(row))[0])
low, high = price * (1 - bundle["test_mape"] / 100), price * (1 + bundle["test_mape"] / 100)

st.metric("Predicted asking price", f"Rs. {price:,.1f} lakhs")
st.write(f"≈ **Rs. {price * 1e5:,.0f}** · likely range Rs. {low:,.1f} – {high:,.1f} lakhs")

# 3. explain it with SHAP (CatBoost's built-in exact SHAP values)
shap_values = model.get_feature_importance(Pool(row, cat_features=CAT_FEATURES), type="ShapValues")[0]
effects = 100 * (np.exp(shap_values[:-1]) - 1)
order = np.argsort(np.abs(effects))
labels = [f"{FEATURES[i]} = {row.iloc[0, i]}" for i in order]

st.subheader("Why this price? (SHAP)")
st.caption(f"Starting from the average car (Rs. {np.expm1(shap_values[-1]):.1f} L), each feature pushes the price "
           "up (red) or down (blue).")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(labels, effects[order], color=np.where(effects[order] > 0, "#d6604d", "#4393c3"))
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Effect on price (%)")
fig.tight_layout()
st.pyplot(fig)
