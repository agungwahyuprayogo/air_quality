import pandas as pd
import numpy as np
import copy
import hashlib
import json
import time
from datetime import datetime
from tqdm import tqdm

# Sklearn Models
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.ensemble import (BaggingClassifier as BGC, RandomForestClassifier as RFC, 
                              AdaBoostClassifier as ABC, GradientBoostingClassifier as GBC)
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV

# Custom Utils
import utils

# --- CONFIGURATION ---
PATH_CONFIG = "../config/config.yaml"
PATH_LOG = "../logs/training_log.json"

def load_train_data(config, suffix):
    """
    Load data train (X, y) berdasarkan suffix (rus, ros, smote) dari config.
    Contoh: config['path_train_rus']
    """
    key = f"path_train_{suffix}"
    if key not in config:
        raise ValueError(f"Path {key} tidak ditemukan di config.yaml")
        
    X = utils.load_joblib(config[key][0])
    y = utils.load_joblib(config[key][1])
    return X, y

def create_log_template():
    return {
        'model_name': [], 'model_id': [], 'training_time': [],
        'training_date': [], 'performance': [], 'f1_score_avg': [],
        'data_configuration': []
    }

def update_training_log(current_log, path_log):
    """Menyimpan log training ke file JSON."""
    try:
        with open(path_log, 'r') as file:
            last_log = json.load(file)
    except FileNotFoundError:
        last_log = []

    last_log.append(current_log)

    with open(path_log, 'w') as file:
        json.dump(last_log, file, indent=4)
    
    return last_log

def train_eval_model(models, prefix_name, X_train, y_train, X_valid, y_valid, data_config):
    """Melakukan training dasar (baseline) untuk berbagai model."""
    logger = create_log_template()
    trained_models = copy.deepcopy(models)
    
    print(f"Training {prefix_name} with {data_config}...")
    
    for model_dict in tqdm(trained_models):
        model_obj = model_dict['model_object']
        model_name = f"{prefix_name} - {model_dict['model_name']}"
        
        # Training
        start_time = datetime.now()
        model_obj.fit(X_train, y_train)
        end_time = datetime.now()
        
        elapsed = (end_time - start_time).total_seconds()
        
        # Evaluation
        y_pred = model_obj.predict(X_valid)
        report = classification_report(y_valid, y_pred, output_dict=True)
        
        # ID Creation
        plain_id = str(start_time) + str(end_time)
        cipher_id = hashlib.md5(plain_id.encode()).hexdigest()
        model_dict['model_id'] = cipher_id
        
        # Logging
        logger['model_name'].append(model_name)
        logger['model_id'].append(cipher_id)
        logger['training_time'].append(elapsed)
        logger['training_date'].append(str(start_time))
        logger['performance'].append(report)
        logger['f1_score_avg'].append(report['macro avg']['f1-score'])
        logger['data_configuration'].append(data_config)
        
    update_training_log(logger, PATH_LOG)
    return trained_models

def hyperparameter_tuning(estimator, param_space, X_train, y_train):
    """Melakukan Random Search untuk mencari hyperparameter terbaik."""
    print(f"Tuning {estimator.__class__.__name__}...")
    
    tuner = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=50, # Bisa dinaikin jadi 100 kalau PC kuat :v
        scoring="f1", # Kita fokus ke F1 Score untuk kelas positif (TIDAK BAIK)
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start_time = time.time()
    tuner.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"Best Param: {tuner.best_params_}")
    print(f"Best CV F1: {tuner.best_score_:.4f}")
    print(f"Time: {int(end_time - start_time)}s\n")
    
    return tuner.best_estimator_, tuner.best_score_

if __name__ == "__main__":
    # 1. Load Configuration
    config = utils.load_config(PATH_CONFIG)
    
    # 2. Load Data (Menggunakan path dari config, bukan Hardcode)
    print("Loading Data...")
    X_rus, y_rus = load_train_data(config, "rus")
    X_ros, y_ros = load_train_data(config, "ros")
    X_sm, y_sm = load_train_data(config, "smote")
    
    X_valid = utils.load_joblib(config["path_valid_feng"][0])
    y_valid = utils.load_joblib(config["path_valid_feng"][1])
    
    # 3. Define Models for Initial Screening
    # (Kita inisialisasi satu set model dasar)
    base_models = [
        {"model_name": "KNN", "model_object": KNN()},
        {"model_name": "LGR", "model_object": LGR(random_state=42)},
        {"model_name": "DTC", "model_object": DTC(random_state=42)},
        {"model_name": "RFC", "model_object": RFC(random_state=42)},
        {"model_name": "GBC", "model_object": GBC(random_state=42)},
        {"model_name": "ABC", "model_object": ABC(random_state=42)}
        # Baseline & Bagging bisa ditambahkan jika perlu
    ]

    # 4. Training Loop (Experimentation)
    # RUS
    train_eval_model(base_models, "Baseline", X_rus, y_rus, X_valid, y_valid, "undersampling")
    # ROS
    train_eval_model(base_models, "Baseline", X_ros, y_ros, X_valid, y_valid, "oversampling")
    # SMOTE
    train_eval_model(base_models, "Baseline", X_sm, y_sm, X_valid, y_valid, "smote")
    
    print("\n--- Initial Experiment Finished. Logs saved. ---\n")
    
    # 5. Hyperparameter Tuning 
    
    print("Starting Hyperparameter Tuning on SMOTE Data...")
    
    # Define Parameter Grids
    params_grid = {
        "KNN": {
            "model": KNN(),
            "params": {"n_neighbors": np.arange(1, 13), "weights": ["uniform", "distance"], "p": [1, 2]}
        },
        "LGR": {
            "model": LGR(random_state=42),
            "params": {"C": [0.01, 0.1, 1.0, 10.0]}
        },
        "DTC": {
            "model": DTC(random_state=42),
            "params": {"max_depth": np.arange(1, 13)}
        },
        "RFC": {
            "model": RFC(random_state=42),
            "params": {"n_estimators": [50, 100, 200], "max_depth": np.arange(1, 13)}
        },
        "GBC": {
            "model": GBC(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": np.arange(1, 13)}
        }
    }
    
    best_models_tuned = {}
    best_score_overall = -1
    best_model_overall = None
    best_model_name = ""

    # Loop Tuning Otomatis
    for name, config_dict in params_grid.items():
        estimator = config_dict["model"]
        space = config_dict["params"]
        
        best_model, score = hyperparameter_tuning(estimator, space, X_sm, y_sm)
        
        # Simpan hasil
        best_models_tuned[name] = best_model
        
        # Cek apakah ini model terbaik sejauh ini?
        if score > best_score_overall:
            best_score_overall = score
            best_model_overall = best_model
            best_model_name = name

    # 6. Saving The Champion Model
    print("=============================================")
    print(f"CHAMPION MODEL: {best_model_name}")
    print(f"Best CV F1 Score: {best_score_overall:.4f}")
    print("=============================================")
    
    prod_model_path = "../models/best_model.pkl"
    utils.dump_joblib(best_model_overall, prod_model_path)
    
    # Update Config
    utils.update_config("path_production_model", prod_model_path, config, PATH_CONFIG)
    
    print("Pipeline Modelling Selesai! Model terbaik tersimpan.")