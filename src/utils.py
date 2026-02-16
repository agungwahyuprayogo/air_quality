import yaml
import joblib

def load_config(path_config):
    """Memuat file konfigurasi (config.yaml)."""
    try:
        with open(path_config, 'r') as file:
            params = yaml.safe_load(file)
    except FileNotFoundError as err:
        raise RuntimeError(f"Configuration file not found in {path_config}")
    return params

def update_config(key, value, params, path_config):
    """Update nilai konfigurasi dan simpan kembali ke file YAML."""
    params = params.copy()
    params[key] = value
    
    with open(path_config, 'w') as file:
        yaml.dump(params, file)
    
    print(f"[CONFIG] Updated: {key} -> {value}")
    # Reload config agar variable di memori selalu update
    return load_config(path_config)

def dump_joblib(data, path):
    """Wrapper untuk menyimpan data menggunakan joblib."""
    joblib.dump(data, path)
    print(f"[DUMP] Data saved to {path}")

def load_joblib(path):
    """Wrapper untuk memuat data menggunakan joblib."""
    return joblib.load(path)