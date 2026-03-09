# Import the required libraries.
import json
import copy
import hashlib

import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import (
    BaggingClassifier as BGC,
    RandomForestClassifier as RFC,
    AdaBoostClassifier as ABC,
    GradientBoostingClassifier as GBC
)

from sklearn.metrics import precision_score
from sklearn.model_selection import RandomizedSearchCV

from utils import *

import warnings
warnings.filterwarnings("ignore")


# Constant variables.
PATH_CONFIG = "../config/config.yaml"


# Function to load preprocessed data.
def load_data(config, data_conf):
    """
    Load the preprocessed data.

    Parameters:
    ----------
    config : dict
        The loaded configuration file.

    data_conf : str
        The data configuration type.
        The value must one of these value: ['train', 'valid', 'test']
    """

    # Ensure the data_conf is valid.
    list_data_conf = ["train", "valid", "test"]

    if data_conf not in list_data_conf:
        raise RuntimeError(f"The data configuration {data_conf} is invalid.")
    else:
        data_conf = str(data_conf)
        path = f"path_clean_{data_conf}"

        X = joblib.load(config[path][0])
        y = joblib.load(config[path][1])

        return X, y

def create_training_log():
    """Return a dictionary representing the training log structure."""
    logger = {
        "model_name": [],
        "model_id": [],
        "training_time": [],
        "training_date": [],
        "train_prec": [],
        "cv_prec": [],
        "data_configuration": []
    }

    return logger

def update_training_log(current_log, path_log):
    """
    Update the training log.

    Parameters:
    ----------
    current_log : dict
        The training log current state.

    path_log : str
        The directory of training log.

    Returns:
    -------
    last_log : dict
        The updated training log.
    """

    # Ensure the current log immutable.
    current_log = copy.deepcopy(current_log)

    # Open the training log file.
    try:
        with open(path_log, 'r') as file:
            last_log = json.load(file)
        file.close()
        
    # If the training log does not exists.
    except FileNotFoundError as err:
        # Create the new training log.
        with open(path_log, 'w') as file:
            file.write("[]")
        file.close()

        # Reload the new training log.
        with open(path_log, 'r') as file:
            last_log = json.load(file)
        file.close()

    # Add the current log to previous log.
    last_log.append(current_log)

    # Rewrite the training log with the updated one.
    with open(path_log, 'w') as file:
        json.dump(last_log, file)
        file.close()

    return last_log

# Function to create model object.
def create_model_object():
    """Return a list of model to be fitted."""

    # Create model object.
    knn = KNN()
    lgr = LGR()
    dtc = DTC()
    bgc = BGC()
    rfc = RFC()
    abc = ABC()
    gbc = GBC()

    # Create list of model.
    list_of_model = [
        {"model_name": knn.__class__.__name__, "model_object": knn, "model_id": ""},
        {"model_name": lgr.__class__.__name__, "model_object": lgr, "model_id": ""},
        {"model_name": dtc.__class__.__name__, "model_object": dtc, "model_id": ""},
        {"model_name": bgc.__class__.__name__, "model_object": bgc, "model_id": ""},
        {"model_name": rfc.__class__.__name__, "model_object": rfc, "model_id": ""},
        {"model_name": abc.__class__.__name__, "model_object": abc, "model_id": ""},
        {"model_name": gbc.__class__.__name__, "model_object": gbc, "model_id": ""}
    ]

    return list_of_model

# Function to create hyperparameter space.
def create_param_space():
    """Return a dict of model hyperparameter."""

    # Define each model hyprerparameter space.
    knn_params = {
        "n_neighbors": [2, 3, 4, 5, 6, 10, 15, 20, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    lgr_params = {
        "C": [0.01, 0.1, 1.0, 10.0]
    }

    # Hyperparameter for DTC, RFC, and GBC.
    DEPTH = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Hyperparameter for BGC, RFC, ABC, and GBC.
    B = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Hyperparameter for ABC and GBC.
    LR = [0.001, 0.01, 0.05, 0.1, 1]

    dist_params = {
        "KNeighborsClassifier": knn_params,
        "LogisticRegression": lgr_params,
        "DecisionTreeClassifier": {
            "max_depth": DEPTH
        },
        "BaggingClassifier": {
            "n_estimators": B
        },
        "RandomForestClassifier": {
            "n_estimators": B,
            "max_depth": DEPTH
        },
        "AdaBoostClassifier": {
            "n_estimators": B,
            "learning_rate": LR
        },
        "GradientBoostingClassifier": {
            "n_estimators": B,
            "learning_rate": LR,
            "max_depth": DEPTH
        }
    }

    return dist_params

# Function o fit & tune model (do CV + HT).
def evaluate_model(models, hyperparameters, config, path_log):
    """Cross Validation & Hyperparameter Tuning."""

    # Load data train.
    X_train, y_train = load_data(config, "train")

    # Create training log.
    logger = create_training_log()

    # Define a dictionary to store the trained models.
    trained_models = {}

    # For each data configuration.
    for data_conf in X_train:
        X_train_conf = X_train[data_conf]
        y_train_conf = y_train[data_conf]
        print(f"Data : {str(data_conf).upper()}")

        # Fit & tune each model.
        for m, h in zip(models, hyperparameters):
            print(f"Fit & Tune Model : {m['model_name']}...")
            # Create tuner object.
            tuner = RandomizedSearchCV(
                estimator = m["model_object"],
                param_distributions = hyperparameters[h],
                n_iter = 100,
                scoring = "precision",
                cv = 5,
                return_train_score = True,
                n_jobs = -1,
                verbose = 1
            )

            # Compute the training time.
            start_time = time_stamp()
            tuner.fit(X_train_conf, y_train_conf)
            finished_time = time_stamp()

            training_time = finished_time - start_time
            training_time = training_time.total_seconds()

            # Get the model with best hyperparameters.
            best_model = tuner.best_estimator_

            # Get the scores of best model.
            best_index = tuner.best_index_
            train_prec = tuner.cv_results_["mean_train_score"][best_index]
            cv_prec = tuner.cv_results_["mean_test_score"][best_index]

            # Store the training information.
            model_name = f"{data_conf} - {m['model_name']}"
            logger["model_name"].append(model_name)

            plain_id = str(training_time)
            cipher_id = hashlib.md5(plain_id.encode()).hexdigest()
            logger["model_id"].append(cipher_id)

            logger["training_time"].append(training_time)
            logger["training_date"].append(str(start_time))
            logger["train_prec"].append(train_prec)
            logger["cv_prec"].append(cv_prec)
            logger["data_configuration"].append(data_conf)

            # Store the best model.
            trained_models[model_name] = [best_model, train_prec, cv_prec]
        print()

    # Update the current training log.
    training_log = update_training_log(logger, path_log)

    return trained_models, training_log

# Function to show the performance summary.
def training_log_to_df(training_log):
    """Return dataframe of performance summary."""
    performances = pd.DataFrame()

    for log in training_log:
        performances = pd.concat([performances, pd.DataFrame(log)])

    performances = performances.sort_values(
        ["cv_prec", "training_time"],
        ascending = [False, True]
    )

    performances = performances.reset_index(drop=True)

    selected_cols = ["model_name", "train_prec", "cv_prec", "training_time"]
    return performances[selected_cols]


# Main function.
def main():
    # 1. Load configuration file.
    config = load_config(PATH_CONFIG)

    # 2. Model Fit & Tune (CV + HT).
    models = create_model_object()
    hyperparameters = create_param_space()

    PATH_TRAINING_LOG = config["path_training_log"]
    trained_models, training_log = evaluate_model(models, hyperparameters, config, PATH_TRAINING_LOG)

    # 3. Get the best model.
    performances = training_log_to_df(training_log)

    best_name = performances["model_name"][0]
    best_model = trained_models[best_name][0]

    print(f"Best Model : {best_name}")

    # 4. Model Serialization.
    PATH_PRODUCTION_MODEL = config["path_production_model"]
    serialize_data(best_model, PATH_PRODUCTION_MODEL)


if __name__ == "__main__":
    main()