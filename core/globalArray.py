import pandas as pd
import os

DISPERSION_XLSX_PATH = "data/compiled_dispersion_data.xlsx"

def update_global_dispersion_data(sample_name: str, data: dict, app = None):
    data_row = {"Sample": sample_name}
    data_row.update(data)

    if os.path.exists(DISPERSION_XLSX_PATH):
        df = pd.read_excel(DISPERSION_XLSX_PATH)
    else:
        df = pd.DataFrame(columns=data_row.keys())

    # Ensure all columns from data_row are present
    for col in data_row:
        if col not in df.columns:
            df[col] = pd.NA

    # Reorder data_row to match df.columns exactly
    aligned_row = pd.Series(data_row)[df.columns].values

    if "Sample" in df.columns and sample_name in df["Sample"].values:
        df.loc[df["Sample"] == sample_name, df.columns] = aligned_row
    else:
        df.loc[len(df)] = aligned_row

    df = df.sort_values(by="Sample", key=lambda x: x.astype(str))

    df = df.applymap(
        lambda x: str(x).replace("µ", "u").replace("μ", "u") if isinstance(x, str) else x
            )
    df = df.replace(r"\s*u?m(\^2)?", "", regex=True)

    df.to_excel(DISPERSION_XLSX_PATH, index=False)

    if app:
        app.notify(f"Data saved in sheet {sample_name} to: {DISPERSION_XLSX_PATH}")

    return df
