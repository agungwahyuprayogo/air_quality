# Import the required libraries.
import os
import pandas as pd
import copy

from sklearn.model_selection import train_test_split
from utils import *


# Constant variables.
PATH_CONFIG = "../config/config.yaml"
RANDOM_STATE = 123


# Function to load raw data.
def load_raw_data(path_data):
    """
    Load csv files and join into one dataframe.

    Parameters:
    ----------
    path_data : str
        Raw dataset location.

    Returns:
    -------
    raw_dataset : pd.DataFrame
        Loaded and joined data.
    """

    # Create variable to store raw dataset.
    raw_dataset = pd.DataFrame()

    # Load and join the csv files.
    for i in os.listdir(path_data):
        raw_dataset = pd.concat([pd.read_csv(path_data + i), raw_dataset])

    # Reset index.
    raw_dataset = raw_dataset.reset_index(drop=True)
    
    return raw_dataset

# Function for data validation.
def data_validation(data):
    """
    Do data validation for removing bad data.

    Parameters:
    ----------
    data : pd.DataFrame
        The loaded raw dataset.

    Returns:
    -------
    validated_data : pd.DataFrame
        The validated data.
    """

    # Ensure raw data immutable.
    data = data.copy()

    # 1. Handling column tanggal.
    data["tanggal"] = pd.to_datetime(data["tanggal"])

    # 2. Handling column pm10.
    data["pm10"] = data["pm10"].replace(
        "---",
        -1
    ).astype(int)

    # 3. Handling column pm25.
    data["pm25"] = data["pm25"].fillna(-1)
    data["pm25"] = data["pm25"].replace(
        "---",
        -1
    ).astype(int)

    # 4. Handling column so2.
    data["so2"] = data["so2"].replace(
        "---",
        -1
    ).astype(int)

    # 5. Handling column co.
    data["co"] = data["co"].replace(
        "---",
        -1
    ).astype(int)

    # 6. Handling column o3.
    data["o3"] = data["o3"].replace(
        "---",
        -1
    ).astype(int)

    # 7. Handling column no2.
    data["no2"] = data["no2"].replace(
        "---",
        -1
    ).astype(int)

    # 8. Handling column max.
    idx_trouble = data[data["max"] == "PM25"].index[0]
    data.loc[idx_trouble, "max"] = data.loc[idx_trouble, "pm10"]
    data.loc[idx_trouble, "critical"] = "PM10"
    data.loc[idx_trouble, "categori"] = "BAIK"
    data["max"] = data["max"].astype(int)

    # 9. Handling column categori.
    missing_labels = data[data["categori"] == "TIDAK ADA DATA"]
    data = data.drop(index = missing_labels.index)
    data = data.rename(columns = {"categori": "category"})

    return data

# Function for data defense.
def data_defense(data, config, api=False):
    """
    Do data defense for checking the data types and range of data.

    Parameters:
    ----------
    data : pd.DataFrame
        The data to be checked.

    config : dict
        Loaded configuration parameters.

    api : bool
        To check whether the input data from API or not.

    Returns:
    -------
    None, it's a void function.
    """

    # Ensure raw data and raw config immutable.
    data = copy.deepcopy(data)
    config = copy.deepcopy(config)

    # If the input is not from API.
    if not api:
        # Check data types.
        assert data.select_dtypes("datetime").columns.to_list() == \
            config["columns_datetime"], "an error occurs in datetime column(s)."
        assert data.select_dtypes("object").columns.to_list() == \
            config["columns_object"], "an error occurs in object column(s)."
        assert data.select_dtypes("int").columns.to_list() == \
            config["columns_int"], "an error occurs in int column(s)."
    else:
        # In case data defense from API.
        col_objects = config["columns_object"]
        # Feature used only stasiun, the rest are discarded.
        del col_objects[1:]

        # Column max not used as features.
        col_ints = config["columns_int"]
        del col_ints[-1]
        
        # Check data types.
        assert data.select_dtypes("object").columns.to_list() == \
            col_objects, "an error occurs in object column(s)."
        assert data.select_dtypes("int").columns.to_list() == \
            col_ints, "an error occurs in int column(s)."

    # Check range of data.
    assert set(data['stasiun']).issubset(set(config['range_stasiun'])), \
        "an error occurs in stasiun range."
    assert data['pm10'].between(config['range_pm10'][0], config['range_pm10'][1]).sum() == \
        len(data), "an error occurs in pm10 range."
    assert data['pm25'].between(config['range_pm25'][0], config['range_pm25'][1]).sum() == \
        len(data), "an error occurs in pm25 range."
    assert data['so2'].between(config['range_so2'][0], config['range_so2'][1]).sum() == \
        len(data), "an error occurs in so2 range."
    assert data['co'].between(config['range_co'][0], config['range_co'][1]).sum() == \
        len(data), "an error occurs in co range."
    assert data['o3'].between(config['range_o3'][0], config['range_o3'][1]).sum() == \
        len(data), "an error occurs in o3 range."
    assert data['no2'].between(config['range_no2'][0], config['range_no2'][1]).sum() == \
        len(data), "an error occurs in no2 range."

# Function for Input-Output Split.
def split_input_output(data, config):
    """
    Split the input(X) and output (y).

    Parameters:
    ----------
    data : pd.DataFrame
        The processed dataset.

    config : dict
        Loaded configuration parameters.

    Returns:
    -------
    X : pd.DataFrame
        The input data.

    y : pd.Series
        The output data.
    """

    X = data[config["features"]].copy()
    y = data[config["label"]].copy()

    # print(f"Original data shape : {data.shape}")
    # print(f"Selected Features   : {config["features"]}")
    # print(f"X data shape        : {X.shape}")
    # print(f"y data shape        : {y.shape}\n")

    return X, y

# Function for Train-Test Split.
def split_train_test(X, y, test_size, random_state):
    """
    Split the train and test set.

    Parameters:
    ----------
    X : pd.DataFrame
        The input data.

    y : pd.Series
        The output data.

    test_size : float
        The proportion of test set.

    random_state : int
        For reproducibility

    Returns:
    -------
    X_train, X_test : pd.DataFrame
        The train and test input.

    y_train, y_test : pd.Series
        The train and test output.
    """

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, y,
                                            test_size = test_size,
                                            random_state = random_state,
                                            stratify = y
                                       )

    # print(f"X_train shape : {X_train.shape}")
    # print(f"y_train shape : {y_train.shape}")
    # print(f"X_test shape  : {X_test.shape}")
    # print(f"y_test shape  : {y_test.shape}\n")

    return X_train, X_test, y_train, y_test

# Main function.
def main():
    # 1. Load configuration file.
    config = load_config(PATH_CONFIG)

    # 2. Load the raw dataset.
    PATH_DATA_RAW = config["path_data_raw"]
    raw_dataset = load_raw_data(PATH_DATA_RAW)

    # 3. Serialize the joined dataset.
    serialize_data(
        data = raw_dataset,
        path = config["path_data_joined"]
    )

    # 4. Data validation.
    validated_data = data_validation(raw_dataset)

    # 5. Serialize the validated dataset.
    serialize_data(
        data = validated_data,
        path = config["path_data_validated"]
    )

    # 6. Data defense.
    data_defense(
        data = validated_data,
        config = config
    )

    # 7. Split input-output.
    X, y = split_input_output(
        data = validated_data,
        config = config
    )

    # 8. Split train-test.
    X_train, X_not_train, y_train, y_not_train = split_train_test(
        X = X,
        y = y,
        test_size = 0.2,
        random_state = RANDOM_STATE
    )

    X_valid, X_test, y_valid, y_test = split_train_test(
        X = X_not_train,
        y = y_not_train,
        test_size = 0.5,
        random_state = RANDOM_STATE
    )

    # 9. Serialize the train, valid, and test set.
    serialize_data(X_train, config["path_data_train"][0])
    serialize_data(y_train, config["path_data_train"][1])

    serialize_data(X_valid, config["path_data_valid"][0])
    serialize_data(y_valid, config["path_data_valid"][1])

    serialize_data(X_test, config["path_data_test"][0])
    serialize_data(y_test, config["path_data_test"][1])


if __name__ == "__main__":
    main()