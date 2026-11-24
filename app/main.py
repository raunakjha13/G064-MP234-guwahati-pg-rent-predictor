import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rentpred.logging_config import get_logger

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
COLUMNS = ["distance_to_campus_km", "room_area_sqft", "sharing", "meals_per_day",
           "building_age_yr", "wifi_mbps", "furnishing", "locality"]

model = joblib.load(MODEL_PATH)
logger = get_logger()
app = FastAPI(title="PG Room Rent Predictor")


class RentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_to_campus_km: float
    room_area_sqft: float = Field(..., ge=40)      # client rule: below 40 is a typing mistake
    sharing: int
    meals_per_day: int
    building_age_yr: float = Field(..., le=70)     # client rule: above 70 is impossible
    wifi_mbps: float
    furnishing: Literal["full", "semi", "unfurnished"]
    locality: Literal["Sector East", "Sector West", "Market Side"]

    @field_validator("furnishing", mode="before")
    @classmethod
    def tidy_furnishing(cls, value):
        return value.strip().lower() if isinstance(value, str) else value


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: RentRequest):
    row = pd.DataFrame([request.model_dump()])[COLUMNS]
    prediction = round(float(model.predict(row)[0]), 2)
    logger.info("prediction made", extra={"extra_fields": {
        "prediction": prediction, "locality": request.locality}})
    return {"prediction": prediction}