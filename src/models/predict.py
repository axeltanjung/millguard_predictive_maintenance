def predict(model, X):
    return model.predict_proba(X)[:, 1]
