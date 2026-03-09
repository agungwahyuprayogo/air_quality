# Import the required libraries.
from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn
import pandas as pd

import utils
import data_pipeline
import preprocessing


# Constant variables.
PATH_CONFIG = "../config/config.yaml"


# Load serialized estimators.
config = utils.load_config(PATH_CONFIG)
ohe_stasiun = utils.deserialize_data(config["path_fitted_encoder_stasiun"])
scaler = utils.deserialize_data(config["path_fitted_scaler"])
le_encoder = utils.deserialize_data(config["path_fitted_encoder_label"])
best_model = utils.deserialize_data(config["path_production_model"])


# Define input data structure.
class DataAPI(BaseModel):
    """Represents the user input data structure."""
    stasiun : str
    pm10 : int
    pm25 : int
    so2 : int
    co : int
    o3 : int
    no2 : int

# Create API object.
app = FastAPI()

# Define handlers.
@app.get("/")
def home():
    return {"message": "Hello, FastAPI up!"}

@app.post("/predict/")
def predict(data: DataAPI):
    # Convert DataAPI to Pandas DataFrame.
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop=True)

    # Convert dtype.
    data = pd.concat(
        [
            data[config["features"][0]],
            data[config["features"][1:]].astype(int)
        ],
        axis = 1
    )

    # Do data defense.
    try:
        data_pipeline.data_defense(data, config, api=True)
    except AssertionError as err:
        return {"res": [], "error_msg": str(err)}

    # Encoding stasiun.
    data = preprocessing.transform_ohe_encoder(data, ohe_stasiun)

    # Scale the data.
    data = preprocessing.transform_scaler(data, scaler)

    # Predict data.
    y_pred = best_model.predict(data)

    # Inverse transform.
    y_pred = list(le_encoder.inverse_transform(y_pred))[0]

    return {"res": y_pred, "error_msg": ""}


if __name__ == "__main__":
    # Run uvicorn server.
    # host="0.0.0.0" -> localhost
    # From api.py get FastAPI object (app).
    uvicorn.run("api:app", host="0.0.0.0", port=8080)