import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import utils  # Kita panggil file utils.py yang baru dibuat

# --- 1. CONFIGURATION ---
PATH_CONFIG = "../config/config.yaml" 


def read_data(path_folder):
    """Membaca dan menggabungkan raw data dari folder."""
    print("Loading Data...")
    raw_dataset = pd.DataFrame()
    for i in tqdm(os.listdir(path_folder)):
        raw_dataset = pd.concat([pd.read_csv(path_folder + i), raw_dataset])
    
    # Reset Index
    raw_dataset = raw_dataset.reset_index(drop=True)
    return raw_dataset

def clean_data(df):
    """Membersihkan data kotor (Cleaning Process)."""
    print("Cleaning Data...")
    
    # 1. Handling Tanggal
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    
    # 2. Handling Polutan (PM10, PM25, SO2, CO, O3, NO2)
    # List kolom polutan yang punya masalah "---"
    pollutants = ["pm10", "pm25", "so2", "co", "o3", "no2"]
    
    for col in pollutants:
        # Ganti NaN dengan -1 dulu (khusus pm25)
        if col == "pm25":
            df[col] = df[col].fillna(-1)
            
        # Replace "---" jadi -1 dan convert ke int
        df[col] = df[col].replace("---", -1).astype(int)
    
    # 3. Handling Kolom Max (Kasus baris 297)
    if df["max"].dtype == 'O': # Cek kalau masih object
        error_idx = 297
        if error_idx in df.index:
            df.loc[error_idx, "max"] = df.loc[error_idx, "pm10"]
            df.loc[error_idx, "critical"] = "PM10"
            df.loc[error_idx, "categori"] = "BAIK"
        
        df["max"] = df["max"].astype(int)

    # 4. Handling Category
    # Drop "TIDAK ADA DATA"
    df = df[df["categori"] != "TIDAK ADA DATA"].copy()
    
    # Rename column
    df = df.rename(columns={"categori": "category"})
    
    return df

def check_data(input_data, params):
    """Data Defense mechanism."""
    print("Checking Data...")
    # Cek tipe data (Simplified version of your assert)
    assert input_data.select_dtypes(include=["datetime"]).columns.to_list() == params["datetime_columns"], "Error datetime cols"
    assert set(input_data.select_dtypes(include=["number"]).columns.to_list()) == set(params["int32_columns"]), "Error int cols"
    
    # Cek Range 
    cols = ["pm10", "pm25", "so2", "co", "o3", "no2"]
    param_keys = ["range_pm10", "range_pm25", "range_so2", "range_co", "range_o3", "range_no2"]

    for col, key in zip(cols, param_keys):
        assert input_data[col].between(params[key][0], params[key][1]).sum() == len(input_data), f"Error range {col}"
    
    print("Data Defense Passed!")

def split_data(df, params):
    """Split data jadi Train, Valid, Test."""
    print("Splitting Data...")
    X = df[params["features"]].copy()
    y = df[params["label"]].copy()
    
    # Split 1: Train vs Not Train (80:20)
    X_train, X_not_train, y_train, y_not_train = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )
    
    # Split 2: Valid vs Test (50:50 dari sisa 20%) -> 10% Valid, 10% Test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_not_train, y_not_train, test_size=0.5, random_state=123, stratify=y_not_train
    )
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

if __name__ == "__main__":
    # 1. Load Config
    config = utils.load_config(PATH_CONFIG)
    
    # 2. Read Data
    raw_data = read_data(config["path_raw_data"])
    
    # (Optional) Dump Joined Data 
    utils.dump_joblib(raw_data, "../data/interim/joined_dataset.pkl")
    config = utils.update_config("path_joined_data", "../data/interim/joined_dataset.pkl", config, PATH_CONFIG)
    
    # 3. Clean Data
    clean_df = clean_data(raw_data)
    
    # Update Config for Labels & Ranges 
    config = utils.update_config("label", "category", config, PATH_CONFIG)
    
    # Update Range Values in Config
    cols = ["pm10", "pm25", "so2", "co", "o3", "no2"]
    param_keys = ["range_pm10", "range_pm25", "range_so2", "range_co", "range_o3", "range_no2"]
    for col, key in zip(cols, param_keys):
        min_val, max_val = int(np.min(clean_df[col])), int(np.max(clean_df[col]))
        config = utils.update_config(key, [min_val, max_val], config, PATH_CONFIG)
        
    # Dump Validated Data
    utils.dump_joblib(clean_df, "../data/interim/validated_data.pkl")
    config = utils.update_config("path_validated_data", "../data/interim/validated_data.pkl", config, PATH_CONFIG)

    # 4. Check Data (Defense)
    check_data(clean_df, config)
    
    # 5. Split Data
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(clean_df, config)
    
    # 6. Dump Splitted Data
    path_split = "../data/interim/"
    utils.dump_joblib(X_train, f"{path_split}X_train.pkl")
    utils.dump_joblib(y_train, f"{path_split}y_train.pkl")
    utils.dump_joblib(X_valid, f"{path_split}X_valid.pkl")
    utils.dump_joblib(y_valid, f"{path_split}y_valid.pkl")
    utils.dump_joblib(X_test, f"{path_split}X_test.pkl")
    utils.dump_joblib(y_test, f"{path_split}y_test.pkl")
    
    print("Pipeline Selesai! Data siap dipakai.")