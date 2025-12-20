from src.data.load_data import load_raw_data
from src.data.clean_data import clean_data
from src.data.split_data import split_data
from src.features.build_features import build_features
from src.features.encoders import build_preprocessor
from src.models.build_pipeline import build_pipeline
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.interpretation.feature_importance import extract_feature_importance

df = load_raw_data("data/raw/millguard_raw.csv")
df = clean_data(df)
df = build_features(df)

X_train, X_test, y_train, y_test = split_data(df)

preprocessor = build_preprocessor(X_train)
pipeline = build_pipeline(preprocessor)

model, best_params = train_model(pipeline, X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

importance = extract_feature_importance(model)

print(best_params)
print(metrics["classification_report"])
print(importance.head(10))
