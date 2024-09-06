''' Validate api in production '''

import requests

URL = 'https://nd0821-c3-starter-code-1-9cz9.onrender.com'


def test_api():
    url_health = f'{URL}'
    response = requests.get(url_health)
    assert response.status_code == 200
    assert response.text == '"Healthy"'


def test_predict():
    url_predict = f'{URL}/predict'
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = requests.post(url_predict, json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<50k"}
