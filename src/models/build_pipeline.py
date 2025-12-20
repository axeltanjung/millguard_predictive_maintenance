from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline(preprocessor):
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs"
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf)
    ])
