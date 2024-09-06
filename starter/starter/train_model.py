# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import pandas as pd
import joblib
import os

# Add the necessary imports for the starter code.
current_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = os.path.join(current_directory, "../data")

# Add code to load in the data.
data = pd.read_csv(os.path.join(data_directory, "census.csv"))

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

precision, recall, fbeta = compute_model_metrics(y_test, inference(model, X_test))
train_precision, train_recall, train_fbeta = compute_model_metrics(y_train, inference(model, X_train))

# save metrics into screenshots folder
metrics_directory = os.path.join(current_directory, "../screenshots")
with open(os.path.join(metrics_directory, "test_metrics.txt"), "w") as outfile:
    outfile.write(f"Precision: {precision}\nRecall: {recall}\nFbeta: {fbeta}")

with open(os.path.join(metrics_directory, "train_metrics.txt"), "w") as outfile:
    outfile.write(f"Precision: {train_precision}\nRecall: {train_recall}\nFbeta: {train_fbeta}")

model_directory = os.path.join(current_directory, "../model")

# save model
joblib.dump(encoder, os.path.join(model_directory, "encoder.joblib"))
joblib.dump(lb, os.path.join(model_directory, "lb.joblib"))
joblib.dump(model, os.path.join(model_directory, "model.joblib"))


