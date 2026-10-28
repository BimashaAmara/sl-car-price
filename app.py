"""Sri Lanka used-car price predictor (NGBoost).   Run:  streamlit run app.py"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st
from scipy.stats import norm

from car_utils import FEATURES, encode, fill_missing, make_input_row

st.set_page_config(page_title="SL Car Price Predictor", page_icon="🚗", layout="wide")
Z80 = norm.ppf(0.9)  # an 80% range covers mean ± 1.28 standard deviations


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    explainer = shap.TreeExplainer(bundle["model"], model_output=0)  # SHAP for the expected log(price)
    return bundle, explainer


bundle, explainer = load_model()
model, stats = bundle["model"], bundle["fill_stats"]

st.title("🚗 Sri Lanka used-car price predictor")
st.caption(f"{bundle['model_name']} (Natural Gradient Boosting) trained on ikman.lk / Kaggle listings · "
           f"test R² {bundle['test_r2']:.2f} · average error {bundle['test_mape']:.0f}% · "
           f"prices in LKR lakhs (1 lakh = Rs. 100,000)")

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

# 1. user input -> one row -> fill missing values and encode exactly as in training
row = fill_missing(make_input_row(brand, car_model, year, engine, gear, fuel, condition, mileage or None), stats)
x = encode(row, stats)

# 2. NGBoost predicts a distribution for log(price): mean (loc) and spread (scale)
dist = model.pred_dist(x)
mu = float(dist.params["loc"][0])
sigma = float(dist.params["scale"][0]) * bundle["sigma_scale"]  # widened using validation data (see notebook)
price, low, high = np.expm1(mu), np.expm1(mu - Z80 * sigma), np.expm1(mu + Z80 * sigma)
width = 100 * (high - low) / price
confidence = "High" if width < 35 else "Medium" if width < 60 else "Low"

c1, c2, c3 = st.columns(3)
c1.metric("Predicted asking price", f"Rs. {price:,.1f} L")
c2.metric("80% price range", f"Rs. {low:,.1f} – {high:,.1f} L")
c3.metric("Confidence", confidence, help="How wide the range is. Rare brands and very old cars get wider ranges.")
st.write(f"≈ **Rs. {price * 1e5:,.0f}**. In testing, about {100 * bundle['coverage80']:.0f}% of real prices "
         "fell inside the model's 80% range.")

# 3. explain the price with SHAP (exact TreeSHAP on NGBoost's mean)
values = explainer.shap_values(x)[0]
effects = 100 * (np.exp(values) - 1)
order = np.argsort(np.abs(effects))
labels = [f"{FEATURES[i]} = {row.iloc[0, i]}" for i in order]

st.subheader("Why this price? (SHAP)")
st.caption(f"Starting from the average car (Rs. {np.expm1(float(np.ravel(explainer.expected_value)[0])):.1f} L), each feature "
           "pushes the price up (red) or down (blue).")
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(labels, effects[order], color=np.where(effects[order] > 0, "#d6604d", "#4393c3"))
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("Effect on price (%)")
fig.tight_layout()
st.pyplot(fig)
