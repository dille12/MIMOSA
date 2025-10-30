import json
import os

def ensure_file_exists(file_path):
    """Ensure the JSON file exists, if not, create an empty JSON file."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({}, file)

def get_value(file_path, key, default=None):
    """Retrieve a value by key from the JSON file, returning a default value if the key is missing."""
    ensure_file_exists(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get(key, default)

def set_value(file_path, key, value):
    """Set a value for a key in the JSON file."""
    ensure_file_exists(file_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    data[key] = value
    print("JSON", data)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)