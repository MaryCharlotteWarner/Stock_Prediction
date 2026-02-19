import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json

import joblib
import tarfile
import tempfile

import boto3
from imblearn.pipeline import Pipeline
import shap


# ===============================
# Setup
# ===============================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
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
        region_name="us-east-1"
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
    "inputs": [{
        "name": "Close Price",
        "type": "number",
        "min": MIN_VAL,
        "max": MAX_VAL,
        "default": DEFAULT_VAL,
        "step": 100.0
    }]
}


# ===============================
# Load Pipeline
# ===============================
def load_pipeline(_session, bucket, key):

    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}",
        Filename=filename
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith(".joblib")][0]

    return joblib.load(joblib_file)


# ===============================
# Load SHAP Explainer
# ===============================
def load_shap_explainer(_session, bucket, key, local_path):

    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
