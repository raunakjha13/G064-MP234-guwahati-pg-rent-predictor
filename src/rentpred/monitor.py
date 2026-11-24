import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from rentpred.clean import LEAK_COLUMN, clean_data

TRAIN_PATH = "data/raw/train.csv"
LATER_PATH = "data/raw/later.csv"
MODEL_PATH = "models/model.joblib"
TARGET = "rent_rs"
NUMERIC = ["distance_to_campus_km", "room_area_sqft", "sharing",
           "meals_per_day", "building_age_yr", "wifi_mbps"]
FEATURES = NUMERIC + ["furnishing", "locality"]


def psi(expected, actual, bins=10):
    """Population Stability Index of one column. Bins come from the expected (train) data."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_frames():
    train = clean_data(pd.read_csv(TRAIN_PATH))
    later = pd.read_csv(LATER_PATH).drop(columns=[LEAK_COLUMN])
    return train, later


def psi_table(train, later):
    return {col: round(psi(train[col], later[col]), 4) for col in NUMERIC}


def mae_by_locality(model, later):
    """MAE of the model on later.csv, for each locality."""
    result = {}
    for loc, rows in later.groupby("locality"):
        pred = model.predict(rows[FEATURES])
        result[loc] = round(mean_absolute_error(rows[TARGET], pred), 1)
    return result


if __name__ == "__main__":
    train, later = load_frames()
    table = psi_table(train, later)
    print("PSI:", table)
    print("Most drifted:", max(table, key=table.get))

    model = joblib.load(MODEL_PATH)
    by_loc = mae_by_locality(model, later)
    print("MAE by locality:", by_loc)
    print("Worst served:", max(by_loc, key=by_loc.get))