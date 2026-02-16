import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import utils

# --- CONFIGURATION ---
PATH_CONFIG = "../config/config.yaml"

def load_dataset(config):
    """Memuat data dari folder interim berdasarkan config."""
    # Load Train
    X_train = utils.load_joblib(config["path_train_set"][0])
    y_train = utils.load_joblib(config["path_train_set"][1])
    
    # Load Valid
    X_valid = utils.load_joblib(config["path_valid_set"][0])
    y_valid = utils.load_joblib(config["path_valid_set"][1])
    
    # Load Test
    X_test = utils.load_joblib(config["path_test_set"][0])
    y_test = utils.load_joblib(config["path_test_set"][1])
    
    # Gabungkan X dan y sementara untuk kemudahan preprocessing
    train_set = pd.concat([X_train, y_train], axis=1)
    valid_set = pd.concat([X_valid, y_valid], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    
    return train_set, valid_set, test_set

def join_label_categories(df, config):
    """Menggabungkan kategori label menjadi BAIK dan TIDAK BAIK."""
    if config["label"] in df.columns:
        df["category"] = df["category"].replace("SEDANG", "TIDAK SEHAT")
        df["category"] = df["category"].replace("TIDAK SEHAT", "TIDAK BAIK")
        return df
    else:
        raise RuntimeError("Kolom Label tidak ditemukan!")

def impute_missing_values(df, config, is_train=False):
    """
    Melakukan imputasi data.
    - PM10 & PM25: Berdasarkan rata-rata kelas (BAIK/TIDAK BAIK).
    - SO2: Mean.
    - CO, O3, NO2: Median.
    """
    # 1. Revert -1 to NaN
    df = df.replace(-1, np.nan)
    
    # 2. Imputasi PM10 & PM25 (Logic Spesifik User)
    for col in ["pm10", "pm25"]:
        # Ambil nilai imputasi dari config
        val_baik = config[f"impute_{col}"]["BAIK"]
        val_tidak_baik = config[f"impute_{col}"]["TIDAK BAIK"]
        
        # Fill NA berdasarkan kondisi kelas
        df.loc[(df["category"] == "BAIK") & (df[col].isna()), col] = val_baik
        df.loc[(df["category"] == "TIDAK BAIK") & (df[col].isna()), col] = val_tidak_baik
        
    # 3. Imputasi SO2, CO, O3, NO2
    impute_values = {
        "so2": config["impute_so2"],
        "co": config["impute_co"],
        "o3": config["impute_o3"],
        "no2": config["impute_no2"]
    }
    df = df.fillna(value=impute_values)
    
    return df

def process_stasiun_ohe(df, config, type="transform"):
    """Handling kolom Stasiun dengan One Hot Encoding."""
    ohe_path = "../models/ohe_stasiun.pkl"
    
    # Load atau Init OHE
    if type == "fit":
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(np.array(df["stasiun"]).reshape(-1, 1))
        utils.dump_joblib(ohe, ohe_path)
    else:
        ohe = utils.load_joblib(ohe_path)
        
    # Transform
    stasiun_feat = ohe.transform(np.array(df["stasiun"]).reshape(-1, 1))
    stasiun_df = pd.DataFrame(stasiun_feat, columns=list(ohe.categories_[0]), index=df.index)
    
    # Concat & Drop
    df = pd.concat([stasiun_df, df], axis=1)
    df.drop(columns=["stasiun"], inplace=True)
    
    # Pastikan nama kolom string semua
    df.columns = df.columns.astype(str)
    
    return df

def process_scaling(df, config, type="transform"):
    """Scaling data numerik menggunakan StandardScaler."""
    scaler_path = "../models/scaler.pkl"
    target_col = "category"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if type == "fit":
        scaler = StandardScaler()
        scaler.fit(X)
        utils.dump_joblib(scaler, scaler_path)
    else:
        scaler = utils.load_joblib(scaler_path)
    
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    # Gabung lagi dengan y
    return pd.concat([X_scaled, y], axis=1)

def encode_label(y, type="transform"):
    """Encoding label target (BAIK/TIDAK BAIK -> 0/1)."""
    le_path = "../models/le_category.pkl"
    
    if type == "fit":
        le = LabelEncoder()
        le.fit(y)
        utils.dump_joblib(le, le_path)
    else:
        le = utils.load_joblib(le_path)
    
    return pd.Series(le.transform(y), index=y.index)

def balancing_data(df):
    """Melakukan RUS, ROS, dan SMOTE."""
    X = df.drop(columns=["category"])
    y = df["category"]
    
    # 1. RUS
    rus = RandomUnderSampler(random_state=123)
    X_rus, y_rus = rus.fit_resample(X, y)
    
    # 2. ROS
    ros = RandomOverSampler(random_state=123)
    X_ros, y_ros = ros.fit_resample(X, y)
    
    # 3. SMOTE
    smote = SMOTE(random_state=123)
    X_sm, y_sm = smote.fit_resample(X, y)
    
    return (X_rus, y_rus), (X_ros, y_ros), (X_sm, y_sm)

if __name__ == "__main__":
    # 1. Load Config & Data
    config = utils.load_config(PATH_CONFIG)
    train_data, valid_data, test_data = load_dataset(config)
    
    # 2. Join Categories (Update Config Label baru)
    print("Joining Categories...")
    config = utils.update_config("label_categories_new", ["BAIK", "TIDAK BAIK"], config, PATH_CONFIG)
    
    train_data = join_label_categories(train_data, config)
    valid_data = join_label_categories(valid_data, config)
    test_data = join_label_categories(test_data, config)
    
    # 3. Calculate Imputation Values (Fit on Train Only)
    print("Calculating Imputation Values...")
    # PM10 & PM25 Logic
    for col in ["pm10", "pm25"]:
        val_baik = float(train_data[train_data['category'] == 'BAIK'][col].mean())
        val_tidak_baik = float(train_data[train_data['category'] == 'TIDAK BAIK'][col].mean())
        
        config = utils.update_config(f"impute_{col}", 
                                     {"BAIK": val_baik, "TIDAK BAIK": val_tidak_baik}, 
                                     config, PATH_CONFIG)
    
    # Other Columns Logic
    impute_vals = {
        "impute_so2": float(train_data["so2"].mean()),
        "impute_co": float(train_data["co"].median()),
        "impute_o3": float(train_data["o3"].median()),
        "impute_no2": float(train_data["no2"].median())
    }
    for key, val in impute_vals.items():
        config = utils.update_config(key, val, config, PATH_CONFIG)

    # Apply Imputation
    train_data = impute_missing_values(train_data, config)
    valid_data = impute_missing_values(valid_data, config)
    test_data = impute_missing_values(test_data, config)
    
    # 4. Encoding Stasiun (Fit on Train, Transform on all)
    print("Encoding Stasiun...")
    train_data = process_stasiun_ohe(train_data, config, type="fit")
    config = utils.update_config("path_ohe_stasiun", "../models/ohe_stasiun.pkl", config, PATH_CONFIG)
    
    valid_data = process_stasiun_ohe(valid_data, config, type="transform")
    test_data = process_stasiun_ohe(test_data, config, type="transform")
    
    # 5. Scaling (Fit on Train, Transform on all)
    print("Scaling Features...")
    train_data = process_scaling(train_data, config, type="fit")
    config = utils.update_config("path_scaler", "../models/scaler.pkl", config, PATH_CONFIG)
    
    valid_data = process_scaling(valid_data, config, type="transform")
    test_data = process_scaling(test_data, config, type="transform")
    
    # 6. Balancing Data (Only for Train) & Label Encoding
    print("Balancing & Label Encoding...")
    
    # Fit Label Encoder first
    encode_label(train_data["category"], type="fit")
    
    # Balancing
    data_rus, data_ros, data_sm = balancing_data(train_data)
    
    # Helper to process & save balanced data
    def process_and_save(X, y, suffix):
        # Encode y
        y_enc = encode_label(y, type="transform")
        
        # Save
        utils.dump_joblib(X, f"../data/processed/X_{suffix}.pkl")
        utils.dump_joblib(y_enc, f"../data/processed/y_{suffix}.pkl")
        
        # Update Config
        utils.update_config(f"path_train_{suffix}", 
                            [f"../data/processed/X_{suffix}.pkl", f"../data/processed/y_{suffix}.pkl"], 
                            config, PATH_CONFIG)

    process_and_save(data_rus[0], data_rus[1], "rus")
    process_and_save(data_ros[0], data_ros[1], "ros")
    process_and_save(data_sm[0], data_sm[1], "smote")
    
    # 7. Process Valid & Test (Label Encoding Only)
    print("Processing Valid & Test Sets...")
    y_valid_enc = encode_label(valid_data["category"], type="transform")
    y_test_enc = encode_label(test_data["category"], type="transform")
    
    # Save Valid
    utils.dump_joblib(valid_data.drop(columns=["category"]), "../data/processed/X_valid_feng.pkl")
    utils.dump_joblib(y_valid_enc, "../data/processed/y_valid_feng.pkl")
    config = utils.update_config("path_valid_feng", 
                                 ["../data/processed/X_valid_feng.pkl", "../data/processed/y_valid_feng.pkl"], 
                                 config, PATH_CONFIG)

    # Save Test
    utils.dump_joblib(test_data.drop(columns=["category"]), "../data/processed/X_test_feng.pkl")
    utils.dump_joblib(y_test_enc, "../data/processed/y_test_feng.pkl")
    config = utils.update_config("path_test_feng", 
                                 ["../data/processed/X_test_feng.pkl", "../data/processed/y_test_feng.pkl"], 
                                 config, PATH_CONFIG)
    
    print("Preprocessing Selesai! Model OHE & Scaler tersimpan. Data balancing siap.")