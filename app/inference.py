import pandas as pd
import numpy as np

FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

def build_input_df(request, expected_columns):
    df = pd.DataFrame([{col: np.nan for col in expected_columns}])

    mapping = {
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear,
        "Type": request.product_type,
    }

    for col, val in mapping.items():
        if col in df.columns:
            df[col] = val

    # Derived feature
    if (
        "Process temperature [K]" in df.columns
        and "Air temperature [K]" in df.columns
    ):
        df["Temp Delta [K]"] = (
            df["Process temperature [K]"]
            - df["Air temperature [K]"]
        )

    # Default failure modes (no leakage)
    for col in FAILURE_MODE_COLS:
        if col in df.columns:
            df[col] = 0

    return df
