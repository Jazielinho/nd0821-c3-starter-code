
import pandas as pd
import joblib
import os
import logging
import boto3
import io

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if os.environ.get("RUN_RENDER", 'false').lower() == 'true':
    logger.info("In Render")
    bucket_name = 'udacity.dvc.fastapi'

    s3 = boto3.client('s3')

    logger.info("Downloading model from S3")
    model_s3_key = 'files/md5/bb/df67e5d72b986a167a92e448165136'
    model_in_memory = io.BytesIO()
    s3.download_fileobj(bucket_name, model_s3_key, model_in_memory)
    model_in_memory.seek(0)
    model = joblib.load(model_in_memory)
    logger.info("Model downloaded")

    logger.info("Downloading encoder from S3")
    encoder_s3_key = 'files/md5/67/1137fde4e14aa3798df1330ebf6533'
    encoder_in_memory = io.BytesIO()
    s3.download_fileobj(bucket_name, encoder_s3_key, encoder_in_memory)
    encoder_in_memory.seek(0)
    encoder = joblib.load(encoder_in_memory)
    logger.info("Encoder downloaded")

    logger.info("Downloading lb from S3")
    lb_s3_key = 'files/md5/cd/49d8bf0d8408a57a516b72247936c3'
    lb_in_memory = io.BytesIO()
    s3.download_fileobj(bucket_name, lb_s3_key, lb_in_memory)
    lb_in_memory.seek(0)
    lb = joblib.load(lb_in_memory)
    logger.info("lb downloaded")
else:
    logger.info("Not in Render")
    current_directory = os.path.dirname(os.path.realpath(__file__))
    model_directory = os.path.join(current_directory, "model")

    model = joblib.load(os.path.join(model_directory, "model.joblib"))
    encoder = joblib.load(os.path.join(model_directory, "encoder.joblib"))
    lb = joblib.load(os.path.join(model_directory, "lb.joblib"))


app = FastAPI()

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


class DataIn(BaseModel):
    age : int = 39
    workclass : str =  "State-gov"
    fnlgt : int = 77516
    education : str = "Bachelors"
    education_num : int = 13
    marital_status : str = Field("Never-married", alias="marital-status")
    occupation : str = "Adm-clerical"
    relationship : str = "Not-in-family"
    race : str = "White"
    sex : str = "Male"
    capital_gain : int = 2174
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = Field("United-States", alias="native-country")


class DataOut(BaseModel):
    prediction : str


@app.post("/predict", response_model=DataOut)
def predict(data: DataIn):
    try:
        data_dict = data.dict()
        data_dict['marital-status'] = data_dict.pop('marital_status')
        data_dict['native-country'] = data_dict.pop('native_country')
        data_df = pd.DataFrame([data_dict])
        data_df, _, _, _ = process_data(
            data_df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
        )
        prediction = inference(model, data_df)
        prediction = ">=50k" if prediction[0] == 1 else "<50k"
        return DataOut(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return "Healthy"
