import os
import yaml

DEFAULT_CONFIG = {
    "field_of_view": {
        "mode": "textfile",
        "textfile": {
            "key": "FOV",
            "line_separator": "$"
        },
        "exif": {
            "tag": "XResolution"
        },
        "constant": {
            "value": 500.0
        },
        "custom": {
            "script": "./fov_hook.py"
        }
    }
}


def load_config(path="data/config.yaml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.isfile(path):
        with open(path, "w") as f:
            yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg
