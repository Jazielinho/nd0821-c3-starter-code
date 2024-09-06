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

# Optional enhancement
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
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

precision, recall, fbeta = compute_model_metrics(
    y_test, inference(model, X_test)
)
train_precision, train_recall, train_fbeta = compute_model_metrics(
    y_train, inference(model, X_train)
)

''' metrics by race and sex '''
metrics_by_race = {}

for race in data['race'].unique():
    _test = test[test['race'] == race]
    _X_test, _y_test, _, _ = process_data(
        _test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    metrics_by_race[race] = compute_model_metrics(
        _y_test, inference(model, _X_test)
    )

metrics_by_sex = {}

for sex in data['sex'].unique():
    _test = test[test['sex'] == sex]
    _X_test, _y_test, _, _ = process_data(
        _test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    metrics_by_sex[sex] = compute_model_metrics(
        _y_test, inference(model, _X_test)
    )


# save metrics into screenshots folder
metrics_directory = os.path.join(current_directory, "../screenshots")
with open(
        os.path.join(metrics_directory, "test_metrics.txt"), "w"
) as outfile:
    outfile.write(
        f"Precision: {precision}\nRecall: {recall}\nFbeta: {fbeta}"
    )

with open(
        os.path.join(metrics_directory, "train_metrics.txt"), "w"
) as outfile:
    outfile.write(
        f'''
        Precision: {train_precision}
        Recall: {train_recall}
        Fbeta: {train_fbeta}
        '''
    )

with open(
        os.path.join(metrics_directory, "metrics_by_race.txt"), "w"
) as outfile:
    for race, metrics in metrics_by_race.items():
        outfile.write(
            f'''
            Race: {race}
            Precision: {metrics[0]}
            Recall: {metrics[1]}
            Fbeta: {metrics[2]}
            '''
        )

with open(
        os.path.join(metrics_directory, "metrics_by_sex.txt"), "w"
) as outfile:
    for sex, metrics in metrics_by_sex.items():
        outfile.write(
            f'''
            Sex: {sex}
            Precision: {metrics[0]}
            Recall: {metrics[1]}
            Fbeta: {metrics[2]}
            '''
        )


model_directory = os.path.join(current_directory, "../model")

# save model
joblib.dump(encoder, os.path.join(model_directory, "encoder.joblib"))
joblib.dump(lb, os.path.join(model_directory, "lb.joblib"))
joblib.dump(model, os.path.join(model_directory, "model.joblib"))
