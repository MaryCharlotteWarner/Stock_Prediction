import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import joblib
import boto3
import shap

warnings.simplefilter("ignore")

# ===============================
# AWS CONFIG
# ===============================

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
BUCKET_NAME = st.secrets["aws_credentials"]["AWS_BUCKET"]
ENDPOINT_NAME = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()

# ===============================
# MODEL FEATURES
# ===============================
# Changed from placeholders to relevant regression feature names
FEATURE_KEYS = [
    "IBM_CR_Cum",
    "NVDA_CR_Cum",
    "GOOGL_CR_Cum",
    "AMD_CR_Cum"
]

# Added user-friendly labels
FEATURE_LABELS = {
    "IBM_CR_Cum": "IBM cumulative return",
    "NVDA_CR_Cum": "NVIDIA cumulative return",
    "GOOGL_CR_Cum": "Alphabet cumulative return",
    "AMD_CR_Cum": "AMD cumulative return"
}

MODEL_INFO = {
    "explainer": "explainer.shap",
    "keys": FEATURE_KEYS,
    "inputs": [
        {"name": k, "label": FEATURE_LABELS[k], "min": -10.0, "max": 10.0, "default": 0.0, "step": 0.01}
        for k in FEATURE_KEYS
    ]
}

# ===============================
# CALL ENDPOINT
# ===============================

def call_model_api(input_df):
    runtime = session.client("sagemaker-runtime")

    try:
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=input_df.to_json(orient="values")
        )

        result = response["Body"].read().decode().strip()
        prediction = float(result.strip("[]"))

        return round(prediction, 6), 200

    except Exception as e:
        return f"Error: {str(e)}", 500

# ===============================
# LOAD SHAP EXPLAINER
# ===============================

@st.cache_resource
def load_shap_explainer():
    s3 = session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])

    if not os.path.exists(local_path):
        s3.download_file(
            BUCKET_NAME,
            f"explainer/{MODEL_INFO['explainer']}",
            local_path
        )

    return joblib.load(local_path)

def display_explanation(input_df):
    try:
        explainer = load_shap_explainer()
        shap_values = explainer(input_df)

        st.subheader("SHAP Explanation")
        fig = plt.figure(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"SHAP explanation could not load: {e}")

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="Regression Predictor", layout="wide")
st.title("Regression Prediction App")
st.write("Enter the four market feature values below and click Run Prediction.")

with st.form("prediction_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = pd.DataFrame([user_inputs])

    prediction, status = call_model_api(input_df)

    if status == 200:
        st.metric("Predicted Value", prediction)
        display_explanation(input_df)
    else:
        st.error(prediction)
