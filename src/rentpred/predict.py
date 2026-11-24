import mlflow
import mlflow.sklearn
import pandas as pd

from rentpred.monitor import FEATURES
from rentpred.train import MODEL_NAME

HOLDOUT_PATH = "data/raw/holdout_features.csv"
OUT_PATH = "predictions/holdout_predictions.csv"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    holdout = pd.read_csv(HOLDOUT_PATH)
    pred = model.predict(holdout[FEATURES])
    out = pd.DataFrame({"record_id": holdout["record_id"], "prediction": pred.round(2)})
    out.to_csv(OUT_PATH, index=False)
    print(len(out), "rows saved to", OUT_PATH)
    print(out.head())


if __name__ == "__main__":
    main()