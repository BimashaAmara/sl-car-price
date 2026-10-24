"""Retrain the final model from scratch, track it with MLflow, and check its quality.

    python train.py

* Uses the same cleaning (car_utils.load_and_clean) and the tuned CatBoost settings from the notebook (params.json).
* Logs parameters, metrics and the model to MLflow:
  DagsHub if DAGSHUB_OWNER / DAGSHUB_TOKEN / DAGSHUB_REPO are set, otherwise a local ./mlruns folder.
* Quality gate: exits with an error if the test MAPE is above the limit, so CI fails instead of shipping a bad model.
* Writes model.pkl (used by app.py) and metrics.json.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

from car_utils import CAT_FEATURES, fill_missing, fit_fill_stats, load_and_clean

SEED = 42
MAX_TEST_MAPE = 20.0  # quality gate


def main() -> int:
    params = json.load(open("params.json"))
    df = load_and_clean("cars.csv")
    hold = (df.model_full == "AQUA") & (df.YOM == 2014) & (df.fuel == "HYBRID") & (df.price == 79.5)
    df = df.drop(index=df.index[hold][0])  # the one car kept out for the demo (same as the notebook)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=pd.qcut(df["price"], 10, labels=False))
    stats = fit_fill_stats(train)
    X_train, X_test = fill_missing(train, stats), fill_missing(test, stats)

    model = CatBoostRegressor(**params, random_seed=SEED, verbose=0, allow_writing_files=False)
    model.fit(X_train, np.log1p(train["price"]), cat_features=CAT_FEATURES)
    pred = np.expm1(model.predict(X_test))
    metrics = {"test_r2": r2_score(test["price"], pred),
               "test_mape": 100 * mean_absolute_percentage_error(test["price"], pred),
               "train_rows": len(train), "test_rows": len(test)}
    print(json.dumps(metrics, indent=2))

    bundle = {"model": model, "fill_stats": stats, "features": list(X_train.columns), "model_name": "CatBoost",
              "test_mape": metrics["test_mape"], "test_r2": metrics["test_r2"],
              "brand_models": train.groupby("brand")["model"].apply(lambda s: sorted(s.unique())).to_dict()}
    pickle.dump(bundle, open("model.pkl", "wb"))
    json.dump(metrics, open("metrics.json", "w"), indent=2)

    # --- experiment tracking ---
    import mlflow

    if os.getenv("DAGSHUB_OWNER") and os.getenv("DAGSHUB_TOKEN"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_OWNER"]
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
        repo = os.getenv("DAGSHUB_REPO", "sl-car-price")
        mlflow.set_tracking_uri(f"https://dagshub.com/{os.environ['DAGSHUB_OWNER']}/{repo}.mlflow")
    mlflow.set_experiment("sl-car-price")
    with mlflow.start_run(run_name="catboost-train"):
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifact("model.pkl")
        mlflow.log_artifact("metrics.json")

    if metrics["test_mape"] > MAX_TEST_MAPE:
        print(f"QUALITY GATE FAILED: test MAPE {metrics['test_mape']:.1f}% > {MAX_TEST_MAPE}%")
        return 1
    print(f"Quality gate passed: test MAPE {metrics['test_mape']:.1f}% <= {MAX_TEST_MAPE}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
