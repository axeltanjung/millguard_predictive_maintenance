import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MillGuard – Predictive Maintenance",
    layout="centered"
)

st.title("🛠️ MillGuard – Predictive Maintenance")
st.caption("Failure risk prediction using machine condition signals")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/baseline_logreg.pkl")

model = load_model()

# -----------------------------
# Sidebar – Input Features
# -----------------------------
st.sidebar.header("Machine Input Parameters")

air_temp = st.sidebar.slider(
    "Air Temperature [K]",
    min_value=290.0,
    max_value=320.0,
    value=300.0
)

process_temp = st.sidebar.slider(
    "Process Temperature [K]",
    min_value=300.0,
    max_value=340.0,
    value=310.0
)

rot_speed = st.sidebar.slider(
    "Rotational Speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.sidebar.slider(
    "Torque [Nm]",
    min_value=0.0,
    max_value=80.0,
    value=40.0
)

tool_wear = st.sidebar.slider(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=100
)

product_type = st.sidebar.selectbox(
    "Product Quality Type",
    options=["L", "M", "H"]
)

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_df = pd.DataFrame([{
    "Air Temperature [K]": air_temp,
    "Process Temperature [K]": process_temp,
    "Rotational Speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool Wear [min]": tool_wear,
    "Type": product_type
}])

# Derived feature (must match training)
input_df["Temp Delta [K]"] = (
    input_df["Process Temperature [K]"]
    - input_df["Air Temperature [K]"]
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Failure Risk"):
    prob = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")

    st.metric(
        label="Failure Probability",
        value=f"{prob:.2%}"
    )

    # Risk interpretation
    if prob >= 0.7:
        st.error("🚨 High Risk of Machine Failure")
    elif prob >= 0.4:
        st.warning("⚠️ Moderate Risk – Monitor Closely")
    else:
        st.success("✅ Low Risk – Normal Operation")

    st.markdown("---")

    st.markdown("### Interpretation Notes")
    st.markdown(
        """
        - Prediction is **probabilistic**, not deterministic  
        - High probability suggests **preventive maintenance** should be considered  
        - Model is trained on **synthetic data** and intended for **decision support**
        """
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("MillGuard © Predictive Maintenance Demo | Data Science Project")
