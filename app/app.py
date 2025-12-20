import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def get_expected_columns(model):
    preprocessor = model.named_steps["preprocess"]
    cols = []
    for _, _, features in preprocessor.transformers_:
        cols.extend(features)
    return cols

def get_global_feature_importance(model, top_n=10):
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs,
            "Impact": np.abs(coefs)
        })
        .sort_values("Impact", ascending=False)
        .head(top_n)
    )
    return df

def get_local_contribution(model, input_df, top_n=8):
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]

    X_transformed = preprocessor.transform(input_df)
    coefs = clf.coef_[0]

    contrib = X_transformed[0] * coefs

    feature_names = preprocessor.get_feature_names_out()

    df = (
        pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contrib
        })
        .assign(Abs=lambda x: np.abs(x["Contribution"]))
        .sort_values("Abs", ascending=False)
        .head(top_n)
    )

    return df

def plot_local_contribution(local_exp):
    fig, ax = plt.subplots(figsize=(7, 4))

    local_exp = local_exp.sort_values("Contribution")

    ax.barh(
        local_exp["Feature"],
        local_exp["Contribution"]
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Contribution to Failure Risk")
    ax.set_title("Local Feature Contribution")

    plt.tight_layout()
    return fig


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
expected_cols = get_expected_columns(model)

# Base input from UI
input_data = {
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rot_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Type": product_type,
}

# Create full feature frame
input_df = pd.DataFrame([{col: np.nan for col in expected_cols}])

# Fill known values
for col, val in input_data.items():
    if col in input_df.columns:
        input_df[col] = val

# Derived feature
if (
    "Process temperature [K]" in input_df.columns
    and "Air temperature [K]" in input_df.columns
):
    input_df["Temp Delta [K]"] = (
        input_df["Process temperature [K]"]
        - input_df["Air temperature [K]"]
    )

# Default values for failure mode flags
for col in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    if col in input_df.columns:
        input_df[col] = 0

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

    st.markdown("### 🔍 Key Drivers of Failure (Global Model View)")

    global_imp = get_global_feature_importance(model, top_n=10)

    chart_df = (
        global_imp
        .assign(Sign=lambda d: d["Coefficient"].apply(
            lambda x: "Increase Risk" if x > 0 else "Reduce Risk"
        ))
        .set_index("Feature")[["Impact"]]
    )

    st.bar_chart(chart_df, height=400)

    st.markdown("### 🧠 Why This Machine Is At Risk")

    local_exp = get_local_contribution(model, input_df)

    fig = plot_local_contribution(local_exp)
    st.pyplot(fig)

    st.markdown("### 🏭 Current Operating Condition")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Air Temp (K)", f"{air_temp:.1f}")
        st.metric("Process Temp (K)", f"{process_temp:.1f}")
        st.metric("Temp Delta (K)", f"{process_temp - air_temp:.1f}")

    with col2:
        st.metric("Rotational Speed (rpm)", f"{rot_speed}")
        st.metric("Torque (Nm)", f"{torque:.1f}")
        st.metric("Tool Wear (min)", f"{tool_wear}")
    
    if (process_temp - air_temp) > 15:
        st.warning("⚠️ High temperature delta detected")

    st.markdown("### 🛠️ Recommended Action")

    if prob >= 0.7:
        st.error("""
        **Immediate Action Required**
        - Schedule preventive maintenance
        - Inspect cooling / heat dissipation system
        - Reduce operational load if possible
        """)
    elif prob >= 0.4:
        st.warning("""
        **Monitor Closely**
        - Increase inspection frequency
        - Track temperature and torque trends
        - Prepare maintenance resources
        """)
    else:
        st.success("""
        **Normal Operation**
        - Continue standard monitoring
        - No immediate maintenance required
        """)


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("MillGuard © Predictive Maintenance Demo | Data Science Project")
