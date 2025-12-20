import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Example derived feature
    if (
        "Process Temperature [K]" in df.columns
        and "Air Temperature [K]" in df.columns
    ):
        df["Temp Delta [K]"] = (
            df["Process Temperature [K]"] - df["Air Temperature [K]"]
        )

    return df
