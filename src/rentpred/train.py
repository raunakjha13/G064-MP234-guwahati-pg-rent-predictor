import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rentpred.clean import LEAK_COLUMN, clean_data

with open("params.yaml") as f:
    PARAMS = yaml.safe_load(f)

TRAIN_PATH = PARAMS["data"]["train_path"]
LATER_PATH = PARAMS["data"]["later_path"]
TARGET = PARAMS["target"]
EXPERIMENT = PARAMS["mlflow"]["experiment"]
MODEL_NAME = PARAMS["mlflow"]["model_name"]
SETTINGS = PARAMS["model_settings"]

NUMERIC = ["distance_to_campus_km", "room_area_sqft", "sharing",
           "meals_per_day", "building_age_yr", "wifi_mbps"]
CATEGORICAL = ["furnishing", "locality"]
FEATURES = NUMERIC + CATEGORICAL


def build_model(settings):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    forest = RandomForestRegressor(random_state=42, **settings)
    return Pipeline([("prep", prep), ("model", forest)])


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    data_hash = hashlib.sha256(open(TRAIN_PATH, "rb").read()).hexdigest()
    train = clean_data(pd.read_csv(TRAIN_PATH))
    later = pd.read_csv(LATER_PATH).drop(columns=[LEAK_COLUMN])

    best = {"mae": float("inf")}
    for settings in SETTINGS:
        with mlflow.start_run() as run:
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params(settings)