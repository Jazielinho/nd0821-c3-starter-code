
import pytest
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope='module')
def data():
    data = pd.read_csv('starter/data/census.csv')
    return data

@pytest.fixture(scope='module')
def X_y(data):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train


def test_train_model(X_y):
    X, y = X_y
    model = train_model(X, y)
    assert model is not None
    assert type(model) == type(RandomForestClassifier())


def test_compute_model_metrics(X_y):
    X, y = X_y
    model = train_model(X, y)
    preds = inference(model, X)
    metrics = compute_model_metrics(y, preds)
    assert metrics is not None
    assert len(metrics) == 3
    for metric in metrics:
        assert metric >= 0
        assert metric <= 1


def test_inference(X_y):
    X, y = X_y
    model = train_model(X, y)
    preds = inference(model, X)
    assert preds is not None
    assert len(preds) == len(y)
    for pred in preds:
        assert pred == 0 or pred == 1