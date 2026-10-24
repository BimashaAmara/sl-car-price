"""Checks run automatically by GitHub Actions on every push."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from car_utils import fill_missing, make_input_row  # noqa: E402

bundle = pickle.load(open(Path(__file__).resolve().parents[1] / "model.pkl", "rb"))


def price(**kw):
    car = dict(brand="TOYOTA", model="AQUA", year=2014, engine_cc=1500, gear="AUTOMATIC", fuel="HYBRID",
               condition="USED", mileage_km=124000, listing_year=2025)
    car.update(kw)
    return float(np.expm1(bundle["model"].predict(fill_missing(make_input_row(**car), bundle["fill_stats"])))[0])


def test_held_out_aqua_within_15_percent():
    assert abs(price() / 79.5 - 1) < 0.15  # real asking price of the car kept out of training


def test_older_car_is_cheaper():
    assert price(year=2010, mileage_km=150000) < price(year=2018, mileage_km=60000)


def test_luxury_brand_costs_more():
    assert price(brand="MERCEDES-BENZ", model="OTHER", fuel="PETROL", engine_cc=2000) > price()


def test_unknown_mileage_and_unseen_model_still_work():
    p = price(model="SOMETHING-NEW", mileage_km=None)
    assert 5 < p < 500
