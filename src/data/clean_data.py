import pandas as pd

DROP_COLS = ["UID", "Product ID"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop identifiers
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Ensure correct dtypes
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    # Basic sanity checks
    if "Torque [Nm]" in df.columns:
        df["Torque [Nm]"] = df["Torque [Nm]"].clip(lower=0)

    return df
