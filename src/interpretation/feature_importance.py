import pandas as pd
import numpy as np

def extract_feature_importance(model):
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    })

    return df.sort_values("abs_coefficient", ascending=False)
