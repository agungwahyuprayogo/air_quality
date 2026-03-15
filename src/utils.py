# Import the required libraries.
import yaml
import joblib

from datetime import datetime


# Common functions.
# Function to load configuration parameter.

PATH_CONFIG = "/home/config/config.yaml"

def load_config():
    try:
        # Gunakan PATH_CONFIG (huruf besar semua)
        with open(PATH_CONFIG, 'r') as file: 
            params = yaml.safe_load(file)
    except FileNotFoundError:
        raise RuntimeError(f"Configuration file not found in {PATH_CONFIG}")
    return params



# Function to update configuration parameter.
def update_config(key, value, params, path_config):
    """
    Update the configuration parameter values.

    Parameters:
    ----------
    key : str
        The key to be updated.

    value : any type supported in Python
        The updated value.

    params : dict
        Loaded configuration parameters.

    path_config : str
        Configuration file location.

    Returns:
    -------
    config : dict
        Updated configuration parameters.
    """

    # To maintain the raw config immutable.
    params = params.copy()

    # Update the configuration parameters.
    params[key] = value

    with open(path_config, 'w') as file:
        yaml.dump(params, file)

    print(f"Params Updated! \nKey: {key} \nValue: {value}\n")

    # Reload the updated configuration parameters.
    config = load_config(path_config)

    return config

# Function to serialize data.
def serialize_data(data, path):
    """
    Dump data into pickle file.

    Parameters:
    ----------
    data : any Python instance
        The data to be serialize.

    path : str
        The serialized data location.

    Returns:
    -------
    None, its a void function.
    """
    print(f"Data serialized to {path}")
    return joblib.dump(data, path)

# Function to deserialize data.
def deserialize_data(path):
    """
    Load and return pickle file.

    Parameters:
    ----------
    path : str
        The serialized data location.

    Returns:
    -------
    None, its a void function.
    """
    print(f"Data deserialized from {path}")
    return joblib.load(path)

# Function to show the current datetime.
def time_stamp():
    """Return the current datetime."""
    return datetime.now()