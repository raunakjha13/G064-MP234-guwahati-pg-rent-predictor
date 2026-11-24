import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from rentpred.monitor import FEATURES, TARGET, TRAIN_PATH, load_frames
from rentpred.train import EXPERIMENT, MODEL_NAME, SETTINGS, build_model

REQUIRED_DROP = 0.15          # the client's rule: v2 must be at least 15% better
V2_SETTINGS = SETTINGS[2]     # same settings as the v1 champion


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT)

    train, later = load_frames()
    later = later.assign(date=pd.to_datetime(later["date"]))
    later = later.sort_values("date").reset_index(drop=True)
    cut = int(len(later) * 0.7)
    first, last = later.iloc[:cut], later.iloc[cut:]
    print("later rows:", len(later), "| first 70%:", len(first), "| last 30%:", len(last))

    v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
    mae_v1 = mean_absolute_error(last[TARGET], v1.predict(last[FEATURES]))

    cols = FEATURES + [TARGET]
    data = pd.concat([train[cols], first[cols]], ignore_index=True)

    with mlflow.start_run() as run:
        data_hash = hashlib.sha256(open(TRAIN_PATH, "rb").read()).hexdigest()
        mlflow.log_param("data_sha256", data_hash)
        mlflow.log_params(V2_SETTINGS)
        mlflow.log_param("trained_on", "train + first 70% of later")

        v2 = build_model(V2_SETTINGS)
        v2.fit(data[FEATURES], data[TARGET])
        mae_v2 = mean_absolute_error(last[TARGET], v2.predict(last[FEATURES]))
        mlflow.log_metric("mae_v1_check", mae_v1)
        mlflow.log_metric("mae_v2_check", mae_v2)
        info = mlflow.sklearn.log_model(v2, name="model", serialization_format="cloudpickle")
        run_id = run.info.run_id

    version = mlflow.register_model(info.model_uri, MODEL_NAME).version
    promote = mae_v2 <= (1 - REQUIRED_DROP) * mae_v1
    if promote:
        MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
        joblib.dump(v2, "models/model.joblib")

    improvement = (mae_v1 - mae_v2) / mae_v1 * 100
    print(f"v1 MAE on last 30%: {mae_v1:.1f}")
    print(f"v2 MAE on last 30%: {mae_v2:.1f}  ({improvement:.1f}% lower)")
    print(f"v2 run id: {run_id} | registered as version {version}")
    print("DECISION:", "promote" if promote else "keep")


if __name__ == "__main__":
    main()