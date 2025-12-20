from pydantic import BaseModel
from typing import Dict, List

class PredictionRequest(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    product_type: str  # L / M / H


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    failure_probability: float
    risk_level: str
    global_top_features: List[str]
    local_contribution: List[FeatureContribution]
    operating_condition: Dict[str, float]
    recommended_action: str
