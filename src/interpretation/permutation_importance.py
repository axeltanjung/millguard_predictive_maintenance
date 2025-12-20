from sklearn.inspection import permutation_importance

def compute_permutation_importance(model, X, y, n_repeats=10):
    return permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        scoring="roc_auc",
        n_jobs=-1
    )
    