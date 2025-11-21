def extract_value(d, key, fmt="raw", default=None):
    if key not in d:
        print(f"KEY '{key}' NOT FOUND, returning default:", default)
        print("AVAILABLE KEYS:", list(d.keys()))
        return default

    val = d[key]

    try:
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
            # first element but float
            return float(val[0])

        if fmt == "list_first_two_float":
            return [float(val[0]), float(val[1])]

        # allow custom callable for advanced cases
        if callable(fmt):
            return fmt(val)

    except Exception:
        return default

    return default


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
