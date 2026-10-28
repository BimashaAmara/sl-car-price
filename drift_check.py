"""Monitoring: has the market changed since the model was trained?

    python drift_check.py

Compares listings from Dec 2024 - Feb 2025 (reference) with Aug - Sep 2026 (current), after Sri Lanka
reopened vehicle imports in 2025. Uses Evidently to test every feature for drift, writes drift_report.html,
and also checks how the model's error differs between the two periods.
"""

import pickle

import numpy as np
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from car_utils import CAT_FEATURES, FEATURES, NUM_FEATURES, encode, fill_missing, load_and_clean

df = load_and_clean("cars.csv")
reference = df[df["date"] < "2026-01-01"]
current = df[df["date"] >= "2026-01-01"]
print(f"Reference: {len(reference):,} listings (Dec 2024 - Feb 2025) | Current: {len(current):,} listings (2026)")

definition = DataDefinition(numerical_columns=NUM_FEATURES, categorical_columns=CAT_FEATURES)
snapshot = Report([DataDriftPreset(columns=FEATURES)]).run(
    reference_data=Dataset.from_pandas(reference[FEATURES], data_definition=definition),
    current_data=Dataset.from_pandas(current[FEATURES], data_definition=definition),
)
snapshot.save_html("drift_report.html")
summary = next(m["value"] for m in snapshot.dict()["metrics"] if m["config"]["type"].endswith("DriftedColumnsCount"))
print(f"Drifted features: {int(summary['count'])} of {len(FEATURES)} (share {summary['share']:.2f})")

print("\nWhat changed:")
for label, part in [("2024-25", reference), ("2026", current)]:
    print(f"  {label}: median age {part['age'].median():.0f} years, "
          f"new cars {100 * (part['condition'] == 'NEW').mean():.0f}%, hybrids {100 * (part['fuel'] == 'HYBRID').mean():.0f}%, "
          f"median price Rs. {part['price'].median():.0f} L")

bundle = pickle.load(open("model.pkl", "rb"))
for label, part in [("2024-25", reference), ("2026", current)]:
    pred = np.expm1(bundle["model"].predict(encode(fill_missing(part, bundle["fill_stats"]), bundle["fill_stats"])))
    print(f"  model error (MAPE) on {label} listings: {100 * np.mean(np.abs(pred - part['price']) / part['price']):.1f}%")

if summary["share"] >= 0.3:
    print("\nDRIFT DETECTED -> retrain the model on recent listings (python train.py) and redeploy.")
else:
    print("\nNo significant drift.")
print("Full report: drift_report.html")
