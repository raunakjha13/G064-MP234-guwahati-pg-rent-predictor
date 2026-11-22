import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from rentpred.clean import LEAK_COLUMN, clean_data
TRAIN_PATH ="data/raw/train.csv"
LATER_PATH ="data/raw/later.csv"
TARGET ="rent_rs"
NUMERIC =["distance_to_campus_km","room_area_sqft","sharing","meals_per_day","building_age_yr","wifi_mbps"]
CATEGORICAL = ["furnishing","locality"]
FEATURES = NUMERIC + CATEGORICAL
EXPERIMENT ="mp234-pg-rent"
MODEL_NAME ="mp234-rent-model"
SETTINGS = [
    {"n_estimators": 100,"max_depth": 6, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
]

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
            model = build_model(settings)
            model.fit(train[FEATURES], train[TARGET])
            mae = mean_absolute_error(later[TARGET], model.predict(later[FEATURES]))
            mlflow.log_metric("mae_later", mae)
            info = mlflow.sklearn.log_model(
                model, name="model", serialization_format="cloudpickle"
            )
            print(f"run {run.info.run_id}  settings={settings}  mae_later={mae:.1f}")
            if mae < best["mae"]:
                best = {"mae": mae, "run_id": run.info.run_id,
                        "uri": info.model_uri, "model": model}
    version = mlflow.register_model(best["uri"], MODEL_NAME).version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    joblib.dump(best["model"], "models/model.joblib")
    print("\nBEST RUN ID:", best["run_id"])
    print("MODEL:", MODEL_NAME, "| champion version:", version, "| MAE:", round(best["mae"], 1))

if __name__ == "__main__":
    main()