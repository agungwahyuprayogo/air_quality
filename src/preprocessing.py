# Import the required libraries.
import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import (
    RandomOverSampler as ROS,
    SMOTE
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler
)
from utils import *
from data_pipeline import *


# Constant variables.
PATH_CONFIG = "../config/config.yaml"


# Function for load data.
def load_data(config):
    """
    Load every set of data.

    Parameters:
    ----------
    config : dict
        The loaded configuration file.

    Returns:
    -------
    data_train, data_valid, data_test : pd.DataFrame
        The loaded data.
    """

    # Load the train set.
    X_train = deserialize_data(config["path_data_train"][0])
    y_train = deserialize_data(config["path_data_train"][1])

    # Load the valid set.
    X_valid = deserialize_data(config["path_data_valid"][0])
    y_valid = deserialize_data(config["path_data_valid"][1])

    # Load the test set.
    X_test = deserialize_data(config["path_data_test"][0])
    y_test = deserialize_data(config["path_data_test"][1])

    # Concat X and y of each set.
    data_train = pd.concat(
        [X_train, y_train],
        axis = 1
    )
    # print(f"Train set shape : {data_train.shape}")

    data_valid = pd.concat(
        [X_valid, y_valid],
        axis = 1
    )
    # print(f"Valid set shape : {data_valid.shape}")

    data_test = pd.concat(
        [X_test, y_test],
        axis = 1
    )
    # print(f"Test set shape  : {data_test.shape}\n")

    return data_train, data_valid, data_test

# Function for join categories.
def join_categories(data, config):
    """
    Join categories SEDANG & TIDAK SEHAT -> TIDAK BAIK.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    config : dict
        The loaded configuration file.

    Returns:
    -------
    data : pd.DataFrame
        The loaded data with categories joined.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Check if label found in data.
    if config["label"] in data.columns.tolist():        

        # Rename SEDANG to TIDAK SEHAT.
        data["category"] = data["category"].replace("SEDANG", "TIDAK SEHAT")

        # Rename TIDAK SEHAT to TIDAK BAIK.
        data["category"] = data["category"].replace("TIDAK SEHAT", "TIDAK BAIK")

        return data
    else:
        raise RuntimeError("Label is not detected in the dataset.")

# Function to replace -1 with NaN.
def nan_replace(data):
    """
    Replace any -1 with NaN (Not a Number).

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    Returns:
    -------
    data : pd.DataFrame
        The processed data.
    """

    # Ensure the raw data immutable.
    data = data.copy()

    # Replace all -1 to NaN.
    data = data.replace(-1, np.nan)

    return data

# Function to calculate class mean for pm10 and pm25.
def calculate_class_mean(data, column):
    """
    Calculate the class mean for column pm10 and pm25.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    column : str
        The column name.

    Returns:
    -------
    impute_baik, impute_tidak_baik : float
        The mean for each class.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Boolean condition for each class.
    data_baik = data["category"] == "BAIK"
    data_tidak_baik = data["category"] == "TIDAK BAIK"

    # Calculate the class mean.
    impute_baik = float(data[data_baik][column].mean())
    impute_tidak_baik = float(data[data_tidak_baik][column].mean())

    # print(f"Mean {column} class BAIK       : {impute_baik}")
    # print(f"Mean {column} class TIDAK BAIK : {impute_tidak_baik}\n")

    return impute_baik, impute_tidak_baik

# Function to impute missing values in column pm10 and pm25 using class mean.
def impute_class_mean(data, column, impute_baik, impute_tidak_baik):
    """
    Impute the missing value for column pm10 and pm25.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    column : str
        The column name.

    impute_baik : float
        The mean for class BAIK.

    impute_tidak_baik : float
        The mean for class TIDAK BAIK.
    
    Returns:
    -------
    data : pd.DataFrame
        The imputed data.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Boolean condition for each class.
    data_baik = data["category"] == "BAIK"
    data_tidak_baik = data["category"] == "TIDAK BAIK"

    # Boolean condition for missing values.
    missing_values = data[column].isnull() == True

    # Slice the missing values for each class.
    missing_baik = data[data_baik & missing_values]
    missing_tidak_baik = data[data_tidak_baik & missing_values]

    # print(f"Num of missing value in {column} class BAIK before imputation       : {len(missing_baik)}")
    # print(f"Num of missing value in {column} class TIDAK BAIK before imputation : {len(missing_tidak_baik)}\n")

    # Impute the missing values.
    data.loc[data[data_baik & missing_values].index, column] = impute_baik
    data.loc[data[data_tidak_baik & missing_values].index, column] = impute_tidak_baik

    # print(f"Num of missing value in {column} class BAIK after imputation        : {data[data_baik][column].isnull().sum()}")
    # print(f"Num of missing value in {column} class TIDAK BAIK after imputation  : {data[data_tidak_baik][column].isnull().sum()}\n")

    return data

# Function to calculate impute values for the other columns.
def calculate_impute_values(data):
    """
    Calculate the impute values for column so2, co, o3, and no2.
        - so2 imputed using the mean.
        - co, o3, and no2 imputed using the median.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    Returns:
    -------
    impute_values : dict
        The calculated impute values.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Calculate the impute values.
    impute_so2 = float(data["so2"].mean())
    impute_co = float(data["co"].median())
    impute_o3 = float(data["o3"].median())
    impute_no2 = float(data["no2"].median())

    impute_values = {
        "so2": impute_so2,
        "co": impute_co,
        "o3": impute_o3,
        "no2": impute_no2
    }

    return impute_values

# Function to impute missing values for the other columns.
def impute_missing_values(data, impute_values):
    """
    Impute the missing values for column so2, co, o3, and no2.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded data.

    impute_values : dict
        The calculated impute values.

    Returns:
    -------
    data : pd.DataFrame
        The imputed data.
    """

    # Ensure raw data immutable.
    data = data.copy()
    # print(f"Num of missing values before imputation :\n{data.isnull().sum()}\n")
    
    # Impute the missing values.
    data = data.fillna(value = impute_values)
    # print(f"Num of missing values after imputation  :\n{data.isnull().sum()}")

    return data

# Function to fit the encoder.
def fit_ohe_encoder(data, path_ohe):
    """
    Fit the OHE encoder.
    
    Parameters:
    ----------
    data : pd.Series
        Categorical input data.

    path_ohe : str
        The OHE encoder location.
    
    Returns:
    -------
    ohe_encoder : sklearn.preprocessing.OneHotEncoder
        Fitted OHE encoder object.
    """

    # Sklearn only accepts 2D matrix, thus we need to reshape the data.
    column = "stasiun"
    X_stasiun = np.array(data[column]).reshape(-1, 1)

    # Create the encoder object.
    ohe_encoder = OneHotEncoder(sparse_output=False)

    # Fit the encoder.
    ohe_encoder.fit(X_stasiun)
    
    # Serialize the ohe_encoder.    
    joblib.dump(ohe_encoder, path_ohe)
    
    return ohe_encoder

# Function to encode the data.
def transform_ohe_encoder(data, encoder):
    """
    Transform the categorical column using OHE encoder.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Data to be transformed.
        
    encoder : sklearn.preprocessing.OneHotEncoder
        The fitted encoder.
        
    Returns:
    -------
    data : pd.DataFrame
        The concatenated data with OHE columns.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Sklearn only accepts 2D matrix, thus we need to reshape the data.
    column = "stasiun"
    X_stasiun = np.array(data[column]).reshape(-1, 1)

    # Encode the data.
    stasiun_features = encoder.transform(X_stasiun)

    # Convert to dataframe.
    stasiun_features = pd.DataFrame(
        stasiun_features.tolist(),
        columns = list(encoder.categories_[0]),
        index = data.index
    )

    # Concat the OHE features with the original data.
    data = pd.concat(
        [stasiun_features, data],
        axis = 1
    )
    
    # Drop the original column.
    data = data.drop(columns = column)

    # Convert columns type to string.
    new_col = [str(col_name) for col_name in data.columns.tolist()]
    data.columns = new_col
    
    return data

# Function to fit the scaler.
def fit_scaler(data, path_scaler):
    """
    Fit the scaler.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Input data (all features must be in numeric form)

    path_scaler : str
        The scaler location.
        
    Returns:
    -------
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object (storing the mean & std of all features)
    """

    # Create scaler object.
    scaler = StandardScaler()

    # Fit the scaler.
    scaler.fit(data)

    # Serialize the scaler.    
    joblib.dump(scaler, path_scaler)
    
    return scaler

# Function to scale the data.
def transform_scaler(data, scaler):
    """
    Transform the data using scaler.
    
    Parameters:
    ----------
    data : pd.DataFrame
        Input data (all features must be in numeric form)    
        
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object (storing the mean & std of all features)
        
    Returns:
    -------
    data : pd.DataFrame
        The scaled data
    """

    # Ensure raw data immutable.
    data = data.copy()

    # Scale the data.
    scaled_data = scaler.transform(data)

    # Convert to dataframe.
    X_scaled = pd.DataFrame(
        scaled_data,
        columns = data.columns,
        index = data.index
    )
    
    return X_scaled

# Function to fit label encoder.
def fit_label_encoder(label, path_le):
    """
    Fit the label encoder.

    Parameters:
    ----------
    label : pd.Series
        Categorical label.

    path_le : str
        The label encoder location.

    Returns:
    -------
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder object.
    """

    # Create the label encoder object.
    label_encoder = LabelEncoder()

    # Fit the label encoder.
    label_encoder.fit(label)

    # Serialize the label encoder.
    joblib.dump(label_encoder, path_le)

    return label_encoder

# Function to encode the label.
def transform_label_encoder(label, encoder):
    """
    Transform the categorical label using label encoder.

    Parameters:
    ----------
    label : pd.Series
        Categorical label.

    encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder object.

    Returns:
    encoded_label : pd.Series
        The encoded label.
    """

    # Ensure raw label immutable.
    label = label.copy()

    # Encode the label.
    encoded_label = pd.Series(
        encoder.transform(label),
        index = label.index,
        name = "category"
    )

    return encoded_label

# Function to balancing the label.
def label_balancer(X, y, balancer_type, config, random_state=123):
    """
    Balancing the category label.

    Parameters:
    ----------
    X : pd.DataFrame
        The scaled data.

    y : pd.DataFrame
        The label to be balanced.

    balancer_type : str
        The balancer type.

    config : dict
        The loaded configuration file.

    random_state : int, default = 123
        For reproducibility.

    Returns:
    -------
    X_balanced : pd.DataFrame
        The features with balanced label.

    y_balanced : pd.Series
        The label with balanced label.
    """

    # Ensure the raw data immutable.
    X = X.copy()
    y = y.copy()

    # Set the balancer.
    list_balancer = ["rus", "ros", "sm"]

    if str(balancer_type).lower() not in list_balancer:
        raise RuntimeError("The balancer type is invalid.")
    else:
        if str(balancer_type).lower() == "rus":
            balancer = RUS(random_state = random_state)            
        elif str(balancer_type).lower() == "ros":
            balancer = ROS(random_state = random_state)
        else:
            balancer = SMOTE(random_state = random_state)

        # Fit resample the balancer.
        X_balanced, y_balanced = balancer.fit_resample(X, y)

        # print(f"The label are balanced using {balancer.__class__.__name__}")

        # Check the label distribution.
        # print(y_balanced.value_counts())

        return X_balanced, y_balanced

# Main function.
def main():
    # 1. Load configuration file.
    config = load_config(PATH_CONFIG)

    # 2. Load each set of data.
    data_train, data_valid, data_test = load_data(config)

    print("PREPROCESSING - START")

    # 3. Join label categories.
    data_train = join_categories(data_train, config)
    data_valid = join_categories(data_valid, config)
    data_test = join_categories(data_test, config)
    
    # 4. Handling missing values.
    data_train = nan_replace(data_train)
    data_valid = nan_replace(data_valid)
    data_test = nan_replace(data_test)

    # 5. Missing Values Imputation.
    # 5.1. Column pm10.
    impute_baik, impute_tidak_baik = calculate_class_mean(data_train, "pm10")
    data_train = impute_class_mean(data_train, "pm10", impute_baik, impute_tidak_baik)
    data_valid = impute_class_mean(data_valid, "pm10", impute_baik, impute_tidak_baik)
    data_test = impute_class_mean(data_test, "pm10", impute_baik, impute_tidak_baik)

    # 5.2. Column pm25.
    impute_baik, impute_tidak_baik = calculate_class_mean(data_train, "pm25")
    data_train = impute_class_mean(data_train, "pm25", impute_baik, impute_tidak_baik)
    data_valid = impute_class_mean(data_valid, "pm25", impute_baik, impute_tidak_baik)
    data_test = impute_class_mean(data_test, "pm25", impute_baik, impute_tidak_baik)

    # 5.3. Column so2, co, o3, and no2.
    impute_values = calculate_impute_values(data_train)
    data_train = impute_missing_values(data_train, impute_values)
    data_valid = impute_missing_values(data_valid, impute_values)
    data_test = impute_missing_values(data_test, impute_values)

    # 6. Split Input-Output.
    X_train, y_train = split_input_output(data_train, config)
    X_valid, y_valid = split_input_output(data_valid, config)
    X_test, y_test = split_input_output(data_test, config)

    # 7. Encode column stasiun.
    PATH_ENCODER_STASIUN = config["path_fitted_encoder_stasiun"]
    encoder = fit_ohe_encoder(X_train, PATH_ENCODER_STASIUN)

    X_train_encoded = transform_ohe_encoder(X_train, encoder)
    X_valid_encoded = transform_ohe_encoder(X_valid, encoder)
    X_test_encoded = transform_ohe_encoder(X_test, encoder)

    # 8. Scale the data.
    PATH_SCALER = config["path_fitted_scaler"]

    scaler = fit_scaler(X_train_encoded, PATH_SCALER)

    X_train_scaled = transform_scaler(X_train_encoded, scaler)
    X_valid_scaled = transform_scaler(X_valid_encoded, scaler)
    X_test_scaled = transform_scaler(X_test_encoded, scaler)

    # 9. Encode the label.
    PATH_LABEL_ENCODER = config["path_fitted_encoder_label"]

    label_encoder = fit_label_encoder(y_train, PATH_LABEL_ENCODER)

    y_train_encoded = transform_label_encoder(y_train, label_encoder)
    y_valid_encoded = transform_label_encoder(y_valid, label_encoder)
    y_test_encoded = transform_label_encoder(y_test, label_encoder)

    # 10. Balancing the label.
    X_rus, y_rus = label_balancer(X_train_scaled, y_train_encoded, "rus", config)
    X_ros, y_ros = label_balancer(X_train_scaled, y_train_encoded, "ros", config)
    X_sm, y_sm = label_balancer(X_train_scaled, y_train_encoded, "sm", config)

    print("PREPROCESSING - END")

    # 11. Serialize the preprocessed data.
    X_train = {
        "Undersampling": X_rus,
        "Oversampling": X_ros,
        "SMOTE": X_sm
    }

    y_train = {
        "Undersampling": y_rus,
        "Oversampling": y_ros,
        "SMOTE": y_sm
    }

    label = config["label"]
    y_valid = data_valid[label]
    X_valid = data_valid.drop(columns = label)

    y_test = data_test[label]
    X_test = data_test.drop(columns = label)

    data_configuration = {
        "train": {
            "X_train": X_train,
            "y_train": y_train
        },
        "valid": {
            "X_valid": X_valid,
            "y_valid": y_valid
        },
        "test": {
            "X_test": X_test,
            "y_test": y_test
        }
    }

    serialize_data(X_train, config["path_clean_train"][0])
    serialize_data(y_train, config["path_clean_train"][1])

    serialize_data(X_valid, config["path_clean_valid"][0])
    serialize_data(y_valid, config["path_clean_valid"][1])

    serialize_data(X_test, config["path_clean_test"][0])
    serialize_data(y_test, config["path_clean_test"][1])


if __name__ == "__main__":
    main()