# Sri Lanka used-car price prediction

Predicts a fair asking price for a used car in Sri Lanka from its listing details (brand, model, year, engine,
gearbox, fuel, condition, mileage), and explains each prediction with SHAP.

* **Data:** 11,750 real Sri Lankan listings (ikman.lk + Kaggle Sri Lanka car data, from
  [Kathuskan/AutoValue](https://github.com/Kathuskan/AutoValue)) → 9,139 after cleaning.
* **Models compared:** Linear Regression (baseline), Random Forest (bagging), XGBoost, LightGBM and CatBoost
  (boosting), Voting and Stacking. All are tuned with RandomizedSearchCV and 5-fold cross-validation.
* **Final model: CatBoost** with test R² 0.88 and about 12% average error (Linear Regression: R² 0.78, 20% error).
* **Explainability:** SHAP (global and per car) and LIME.

## Files
| File | What it is |
|---|---|
| `car_price_report.ipynb` | the full analysis: EDA → cleaning → models → tuning → comparison → SHAP/LIME → inference |
| `app.py` | Streamlit app: enter a car, get its price and a SHAP explanation |
| `car_utils.py` | shared helpers (cleaning + filling missing values), used by the notebook, app and training |
| `model.pkl` | the trained CatBoost model (pickle) |
| `cars.csv` | the dataset |
| **MLOps** | |
| `train.py` + `params.json` | retrains the final model, logs it to MLflow (DagsHub), and fails if the error is too high (quality gate) |
| `tests/test_model.py` | automatic checks (held-out car within 15%, older car cheaper, luxury brand costs more) |
| `drift_check.py` | monitoring: compares 2024–25 listings with 2026 listings (Evidently) |
| `.github/workflows/ci.yml` | GitHub Actions: on every push it retrains, runs the gate, tests, checks drift and builds Docker |
| `Dockerfile` | packages the app as a container |

## Run it
```
pip install -r requirements-dev.txt
streamlit run app.py              # the app
jupyter notebook                  # open car_price_report.ipynb
python train.py                   # retrain + MLflow + quality gate
pytest                            # tests
python drift_check.py             # drift report → drift_report.html
```

## MLOps flow
```
push to GitHub → GitHub Actions: retrain (train.py) → quality gate → tests → drift check → Docker build
                                   └─ logs every run to MLflow on DagsHub
main branch → Streamlit Community Cloud redeploys the app automatically
```
Live app: https://sl-car-price.streamlit.app · MLflow runs: https://dagshub.com/BimashaAmara/sl-car-price/experiments
