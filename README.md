# Sri Lanka used-car price prediction with NGBoost

Predicts a fair asking price **and a realistic price range** for a used car in Sri Lanka from its listing
details (brand, model, year, engine, gearbox, fuel, condition, mileage), and explains each prediction with SHAP.

* **Data:** 11,750 real Sri Lankan listings (ikman.lk + Kaggle Sri Lanka car data, from
  [Kathuskan/AutoValue](https://github.com/Kathuskan/AutoValue), `data/cars.csv`) → 9,139 after cleaning.
  No personal data (no seller names, phone numbers or addresses).
* **New algorithm: NGBoost** (Natural Gradient Boosting, Duan et al. 2020), not covered in the lectures. It predicts
  a probability distribution for each car's price, so every car gets its own calibrated price range.
* **Compared with models taught in the module:** Linear Regression, k-NN, Decision Tree, Random Forest, XGBoost and
  CatBoost, all tuned with RandomizedSearchCV and 5-fold cross-validation.
* **Explainability:** SHAP (for the price and for the uncertainty), PDP/ICE plots and LIME.
* **Front-end:** Streamlit app, live at https://sl-car-price.streamlit.app

## Files
| File | What it is |
|---|---|
| `car_price_report.ipynb` | the full analysis: data → EDA → preprocessing → NGBoost vs taught models → evaluation → SHAP / PDP / LIME → discussion |
| `cars.csv` | the dataset |
| `car_utils.py` | shared helpers (cleaning, filling missing values, encoding), used by the notebook, app and training |
| `model.pkl` | the trained NGBoost model (pickle) |
| `app.py` | Streamlit app: enter a car, get its price, its 80% price range and a SHAP explanation |
| `train.py` + `params.json` | retrains the final NGBoost model from raw data with the tuned settings |
| `tests/test_model.py` | automatic checks (held-out car, older car cheaper, luxury brand costs more, range contains the real price) |
| `drift_check.py` | compares 2024–25 listings with 2026 listings |
| `.github/workflows/ci.yml`, `Dockerfile` | GitHub Actions: retrain, test, drift check and Docker build on every push |

## Run it
```
pip install -r requirements-dev.txt
jupyter notebook                  # open car_price_report.ipynb
streamlit run app.py              # the app
python train.py                   # retrain the final model
pytest                            # tests
```

Live app: https://sl-car-price.streamlit.app
