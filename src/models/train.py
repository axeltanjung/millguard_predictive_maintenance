import joblib
from sklearn.model_selection import GridSearchCV
from src.utils.config import SCORING, CV_FOLDS, MODEL_PATH

def train_model(pipeline, X_train, y_train):
    param_grid = {
        "clf__C": [0.1, 1.0, 10.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        scoring=SCORING,
        cv=CV_FOLDS,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, MODEL_PATH)

    return best_model, grid.best_params_
