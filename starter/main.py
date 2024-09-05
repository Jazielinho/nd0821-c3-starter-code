
import pandas as pd
import joblib
import os
import subprocess
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_dvc_commands():
    try:
        for dir_name in ['dvc-cache', 'dvc-tmp', 'dvc-state']:
            os.makedirs(f'/tmp/{dir_name}', exist_ok=True)
        
        subprocess.run(['dvc', 'cache', 'dir', '/tmp/dvc-cache'], check=True)
        
        env = os.environ.copy()
        env['DVC_TMP_DIR'] = '/tmp/dvc-tmp'
        env['DVC_STATE_DIR'] = '/tmp/dvc-state'
        subprocess.run(['dvc', 'pull'], env=env, check=True)
    except PermissionError:
        logger.warning("No se pudo escribir en /tmp. Intentando sin configurar directorios específicos.")
        try:
            subprocess.run(['dvc', 'pull'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al ejecutar dvc pull: {e}")
            raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al ejecutar comando DVC: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise

if os.environ.get('RUN_RENDER', 'false').lower() == 'true':
    logger.info("Ejecutando comandos DVC...")
    run_dvc_commands()
    logger.info("Comandos DVC completados.")


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
