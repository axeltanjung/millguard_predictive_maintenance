from sklearn.model_selection import train_test_split
from src.utils.config import TEST_SIZE, RANDOM_STATE

def split_data(df, target_col="Machine failure"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if y.nunique() > 1 else None

    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )
