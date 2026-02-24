import os
import sys
import warnings
import json
import tarfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import boto3
import shap

from imblearn.pipeline import Pipeline

# ===============================
# Setup
# ===============================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import get_bitcoin_historical_prices

# ===============================
# Streamlit Secrets
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
runtime_client = session.client("sagemaker-runtime")

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
    "inputs": [
        {
            "name": "Close Price",
            "type": "number",
            "min": MIN_VAL,
            "max": MAX_VAL,
            "default": DEFAULT_VAL,
            "step": 100.0,
        }
    ],
}

# ===============================
# Load Pipeline
# ===============================
@st.cache_resource
def load_pipeline(_session, bucket, key_prefix):

    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    local_tar = os.path.join(tempfile.gettempdir(), filename)

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{key_prefix}/{filename}",
        Filename=local_tar,
    )

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=tempfile.gettempdir())
        joblib_files = [
            f for f in tar.getnames() if f.endswith(".joblib")
        ]

    if not joblib_files:
        raise FileNotFoundError("No .joblib file found in model tar.gz")

    joblib_path = os.path.join(tempfile.gettempdir(), joblib_files[0])

    return joblib.load(joblib_path)

# ===============================
# Load SHAP Explainer
# ===============================
@st.cache_resource
def load_shap_explainer(_session, bucket, key_prefix, local_path):

    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=f"{key_prefix}/{MODEL_INFO['explainer']}",
            Filename=local_path,
        )

    return joblib.load(local_path)

# ===============================
# SageMaker Prediction
# ===============================
def predict_sagemaker(input_df):

    payload = json.dumps(input_df.values.tolist())

    response = runtime_client.invoke_endpoint(
        EndpointName=MODEL_INFO["endpoint"],
        ContentType="application/json",
        Body=payload,
    )

    result = json.loads(response["Body"].read().decode())
    return result

# ===============================
# Example Streamlit UI
# ===============================
st.title("Bitcoin Prediction App")

close_price = st.number_input(
    "Close Price",
    min_value=float(MIN_VAL),
    max_value=float(MAX_VAL),
    value=float(DEFAULT_VAL),
    step=100.0,
)

if st.button("Predict"):
    input_df = pd.DataFrame([[close_price]], columns=["Close Price"])
    prediction = predict_sagemaker(input_df)
    st.write("Prediction:", prediction)
