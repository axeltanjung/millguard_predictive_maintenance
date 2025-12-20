from fastapi import FastAPI
import joblib

from app.schemas import PredictionRequest, PredictionResponse
from app.inference import build_input_df
from app.explanation import (
    get_global_feature_importance,
    get_local_contribution
)

app = FastAPI(
    title="MillGuard Predictive Maintenance API",
    version="1.0"
)

# -------------------------
# Load model once
# -------------------------
model = joblib.load("models/baseline_logreg.pkl")

# Extract expected schema
pre = model.named_steps["preprocess"]
EXPECTED_COLS = []
for _, _, cols in pre.transformers_:
    EXPECTED_COLS.extend(cols)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    input_df = build_input_df(req, EXPECTED_COLS)

    prob = float(model.predict_proba(input_df)[0, 1])

    # Risk bucket
    if prob >= 0.7:
        risk = "HIGH"
        action = "Immediate maintenance recommended"
    elif prob >= 0.4:
        risk = "MEDIUM"
        action = "Monitor closely and prepare maintenance"
    else:
        risk = "LOW"
        action = "Normal operation"

    return PredictionResponse(
        failure_probability=prob,
        risk_level=risk,
        global_top_features=get_global_feature_importance(model),
        local_contribution=get_local_contribution(model, input_df),
        operating_condition={
            "air_temperature": req.air_temperature,
            "process_temperature": req.process_temperature,
            "temp_delta": req.process_temperature - req.air_temperature,
            "rotational_speed": req.rotational_speed,
            "torque": req.torque,
            "tool_wear": req.tool_wear,
        },
        recommended_action=action
    )
