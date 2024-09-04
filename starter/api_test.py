
import json
from fastapi.testclient import TestClient
from main import app, DataIn, DataOut


client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Healthy"


def test_predict():
    data_in = DataIn(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States"
    )
    response = client.post("/predict", json=data_in.dict())
    assert response.status_code == 200
    assert response.json() == {"prediction": "<50k"}
    
    
