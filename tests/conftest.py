import pandas as pd
import pytest


@pytest.fixture
def small_df():
    """Three fake rows that contain the problems we found in train.csv."""
    return pd.DataFrame({
        "record_id": ["R1", "R2", "R3"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "distance_to_campus_km": [1.0, 2.0, 3.0],
        "room_area_sqft": [200.0, 20.0, 220.0],   # row 2: Sector East, in square metres
        "sharing": [1, 2, 1],
        "meals_per_day": [2, 2, 2],
        "building_age_yr": [5, 10, 15],
        "wifi_mbps": [50.0, -1.0, 70.0],           # row 2: sentinel
        "furnishing": [" Full", "UNF.", "sem."],   # messy spellings
        "locality": ["Market Side", "Sector East", "Sector West"],
        "rent_in_signed_agreement": [20000, 21000, 22000],
        "rent_rs": [20100, 21100, 22100],
    })