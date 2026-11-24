from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID = {
    "distance_to_campus_km": 2.5, "room_area_sqft": 200, "sharing": 2,
    "meals_per_day": 2, "building_age_yr": 10, "wifi_mbps": 50,
    "furnishing": "semi", "locality": "Sector West",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_a_number():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_building_age_above_70_is_rejected():
    response = client.post("/predict", json={**VALID, "building_age_yr": 71})
    assert response.status_code == 422


def test_room_area_below_40_is_rejected():
    response = client.post("/predict", json={**VALID, "room_area_sqft": 39})
    assert response.status_code == 422


def test_extra_field_is_rejected():
    response = client.post("/predict", json={**VALID, "rent_in_signed_agreement": 1})
    assert response.status_code == 422