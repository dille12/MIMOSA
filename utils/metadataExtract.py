import traceback
def extract_value(d, key, fmt="raw", default=None):
    """
    Extract a value from a dict by key or a list of keys.
    Supports formatting and defaults.
    """
    def _extract_single(k):
        if k not in d:
            print(f"KEY '{k}' NOT FOUND, returning default:", default)
            print("AVAILABLE KEYS:", list(d.keys()))
            return default

        val = d[k]

        try:

            if fmt == "auto":
                if isinstance(val, (tuple, list)):
                    return float(val[0])
                return float(val)

            if fmt == "raw":
                return val

            if fmt == "float":
                return float(val)

            if fmt == "int":
                return int(val)

            if fmt == "str":
                return str(val)

            if fmt == "list":
                return list(val)

            if fmt == "list_float":
                return [float(x) for x in val]

            if fmt == "scalar_first":
                return float(val[0])

            if fmt == "list_first_two_float":
                return [float(val[0]), float(val[1])]

            if callable(fmt):
                return fmt(val)

        except Exception:
            traceback.print_exc()
            return default

        return default

    # if key is iterable (but not string/bytes), treat as multiple keys
    if isinstance(key, (list, tuple)):
        return [_extract_single(k) for k in key]
    else:
        return _extract_single(key)



def flatten_dict(d, parent_key="", sep="."):
    flat = {}

    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, dict):
            flat.update(flatten_dict(v, new_key, sep))

        elif isinstance(v, list):
            for i, item in enumerate(v):
                idx_key = f"{new_key}{sep}{i}"
                if isinstance(item, dict):
                    flat.update(flatten_dict(item, idx_key, sep))
                else:
                    flat[idx_key] = item

        else:
            flat[new_key] = v

    return flat

def normalize_value(val, default=""):
    """
    Normalize values into strings:
    - Tuple (name, value, unit) -> "valueunit"
    - Single number -> string
    - String -> unchanged
    - List of above -> list of strings
    - None -> default
    """
    try:
        if val is None:
            return default

        # handle lists
        if isinstance(val, list):
            return [normalize_value(v, default) for v in val]

        # handle tuples
        if isinstance(val, tuple):
            # search for first numeric and first unit-like string
            num = None
            unit = ""
            for x in val:
                if num is None and isinstance(x, (int, float)):
                    num = x
                elif isinstance(x, str) and x.strip() != "":
                    unit = x
            if num is not None:
                return f"{num}{unit}"
            # fallback: join all as strings
            return "".join(map(str, val))

        # single number
        if isinstance(val, (int, float)):
            return str(val)

        # string
        if isinstance(val, str):
            return val

        # fallback
        return str(val)

    except Exception:
        return default



# assuming normalize_value is already imported
# from your_module import normalize_value

class TestNormalizeValue():

    print(normalize_value([('Width', 57.16, 'µm'), ('Height', 42.87, 'µm')]))
    # ["114.3µm", "85.75µm"]

    print(normalize_value([42, "hello", ('X', 5, 'cm')]))
    # ["42", "hello", "5cm"]

    print(normalize_value([[('A', 1, 'm'), ('B', 2, 'm')], 3]))
    # [["1m", "2m"], "3"]

    print(normalize_value(None, default="NA"))
    # "NA"

    print(normalize_value([]))
    # []


if __name__ == "__main__":
    TestNormalizeValue()
