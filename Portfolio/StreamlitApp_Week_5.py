import os
import sys
import warnings
import tempfile
import tarfile
import posixpath

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import boto3
import sagemaker

from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline
import shap

# ===============================
# Setup & PATH FIX (IMPORTANT)
# ===============================
warnings.simplefilter("ignore")

import pathlib
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.feature_utils import get_bitcoin_historical_prices

# ===============================
# Secrets
# ===============================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint_bitcoin = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ===============================
# AWS Session
# ===============================
@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)

# ===============================
# Data
# ===============================
df_prices = get_bitcoin_historical_prices()

MIN_VAL = 0.5 * df_prices.iloc[:, 0].min()
MAX_VAL = 2.0 * df_prices.iloc[:, 0].max()
DEFAULT_VAL = df_prices.iloc[:, 0].mean()

MODEL_INFO = {
    "endpoint": aws_endpoint_bitcoin,
    "explainer": "explainer_bitcoin.shap",
    "pipeline": "finalized_bitcoin_model.tar.gz",
    "keys": ["Close Price"],
    "inputs": [{
        "name": "Close Price",
        "type": "number",
        "min": float(MIN_VAL),
        "max": float(MAX_VAL),
        "default": float(DEFAULT_VAL),
        "step": 100.0,
    }],
}

# ===============================
# Load Pipeline
# ===============================
@st.cache_resource
def load_pipeline(_session, bucket, key_prefix):

    s3 = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    local_tar = os.path.join(tempfile.gettempdir(), filename)

    s3.download_file(
        Bucket=bucket,
        Key=f"{key_prefix}/{filename}",
        Filename=local_tar,
    )

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(os.path.join(tempfile.gettempdir(), joblib_file))

# ===============================
# Load SHAP Explainer
# ===============================
@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):

    s3 = _session.client("s3")

    if not os.path.exists(local_path):
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path,
        )

    return joblib.load(local_path)

# ===============================
# Prediction
# ===============================
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(pred_val, pred_val), 200

    except Exception as e:
        return f"Error: {e}", 500

# ===============================
# SHAP Explanation
# ===============================
def display_explanation(input_df):

    explainer_name = MODEL_INFO["explainer"]

    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    full_pipeline = load_pipeline(
        session,
        aws_bucket,
        "sklearn-pipeline-deployment",
    )

    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    transformed = preprocessing_pipeline.transform(input_df)

    shap_values = explainer(transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# ===============================
# UI
# ===============================
st.set_page_config(page_title="ML Deployment Compiler", layout="wide")
st.title("👨‍💻 ML Deployment Compiler")

with st.form("pred_form"):

    st.subheader("Inputs")
    user_inputs = {}

    for inp in MODEL_INFO["inputs"]:
        user_inputs[inp["name"]] = st.number_input(
            inp["name"],
            min_value=inp["min"],
            max_value=inp["max"],
            value=inp["default"],
            step=inp["step"],
        )

    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.concat(
        [df_prices, pd.DataFrame([data_row], columns=df_prices.columns)]
    )

    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df)
    else:
        st.error(res)
