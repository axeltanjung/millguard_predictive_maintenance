import pandas as pd
import numpy as np

def get_global_feature_importance(model, top_n=5):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    features = pre.get_feature_names_out()
    coefs = clf.coef_[0]

    df = pd.DataFrame({
        "feature": features,
        "impact": np.abs(coefs)
    }).sort_values("impact", ascending=False)

    return df["feature"].head(top_n).tolist()


def get_local_contribution(model, input_df, top_n=5):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X = pre.transform(input_df)
    contrib = X[0] * clf.coef_[0]

    df = pd.DataFrame({
        "feature": pre.get_feature_names_out(),
        "contribution": contrib
    })

    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).head(top_n)

    return df[["feature", "contribution"]].to_dict(orient="records")
