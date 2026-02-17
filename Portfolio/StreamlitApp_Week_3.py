import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import tempfile
import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
import shap

warnings.simplefilter("ignore")

# ===============================
# AWS CONFIG
# ===============================

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]

BUCKET_NAME = "marycharlottewarnerbucket"
ENDPOINT_NAME = "HW2-pipeline-endpoint-auto"

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)

# ===============================
# MODEL CONFIG
# ===============================

FEATURE_KEYS = [
    "MSFT_Return",
    "USD_Index_Return",
    "SP500_Return",
    "VIX_Return"
]

MODEL_INFO = {
    "endpoint": ENDPOINT_NAME,
    "explainer": "explainer.shap",
    "keys": FEATURE_KEYS,
    "inputs": [
        {"name": k, "min": -0.2, "max": 0.2, "default": 0.0, "step": 0.001}
        for k in FEATURE_KEYS
    ]
}

# ===============================
# PREDICTION FUNCTION
# ===============================

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values)
        pred_val = float(raw_pred[0])
        return round(pred_val, 6), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ===============================
# SHAP LOADER
# ===============================

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
    explainer = load_shap_explainer()
    shap_values = explainer(input_df.values)

    st.subheader("🔍 SHAP Explanation")

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="AAPL Return Predictor", layout="wide")
st.title("📈 AAPL 5-Day Return Predictor")

with st.form("prediction_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_inputs])

    prediction, status = call_model_api(input_df)

    if status == 200:
        st.metric("Predicted AAPL 5-Day Return", prediction)
        display_explanation(input_df)
    else:
        st.error(prediction)
