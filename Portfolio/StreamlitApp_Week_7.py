import os
import json
import tarfile
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer
from sklearn.pipeline import Pipeline
import shap

from feature_utils import convert_input_pca_regression

warnings.simplefilter("ignore")

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Stock Return Regression App", layout="wide")
st.title("📈 Stock Return Regression App")
st.caption(
    "Enter four market feature values, click Run Prediction, and the app will predict the target stock's future return."
)

# ---------------------------
# AWS SECRETS
# ---------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ---------------------------
# AWS SESSION
# ---------------------------
@st.cache_resource
def get_boto_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )

boto_session = get_boto_session()
sm_session = sagemaker.Session(boto_session=boto_session)

# ---------------------------
# FEATURE CONFIG
# These should match the 4 inputs your project is using
# ---------------------------
FEATURE_CONFIG = [
    {
        "name": "IBM_CR_Cum",
        "label": "IBM cumulative return",
        "help": "Recent cumulative return signal for IBM.",
        "min": -5.0,
        "max": 5.0,
        "default": 0.0,
        "step": 0.05,
    },
    {
        "name": "NVDA_CR_Cum",
        "label": "NVIDIA cumulative return",
        "help": "Recent cumulative return signal for NVIDIA.",
        "min": -5.0,
        "max": 5.0,
        "default": 0.0,
        "step": 0.05,
    },
    {
        "name": "GOOGL_CR_Cum",
        "label": "Alphabet cumulative return",
        "help": "Recent cumulative return signal for Alphabet / Google.",
        "min": -5.0,
        "max": 5.0,
        "default": 0.0,
        "step": 0.05,
    },
    {
        "name": "AMD_CR_Cum",
        "label": "AMD cumulative return",
        "help": "Recent cumulative return signal for AMD.",
        "min": -5.0,
        "max": 5.0,
        "default": 0.0,
        "step": 0.05,
    },
]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer_filename": "explainer_pca.shap",
    "pipeline_filename": "finalized_pca_model.tar.gz",
    "pipeline_s3_prefix": "sklearn-pipeline-deployment",
    "explainer_s3_prefix": "explainer",
}

# ---------------------------
# LOADERS
# ---------------------------
@st.cache_resource
def load_pipeline_from_s3():
    s3_client = boto_session.client("s3")

    local_tar_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["pipeline_filename"])

    if not os.path.exists(local_tar_path):
        s3_client.download_file(
            Bucket=aws_bucket,
            Key=f"{MODEL_INFO['pipeline_s3_prefix']}/{MODEL_INFO['pipeline_filename']}",
            Filename=local_tar_path,
        )

    extract_dir = os.path.join(tempfile.gettempdir(), "hw5_pipeline_extract")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [name for name in tar.getnames() if name.endswith(".joblib")]

    if not joblib_files:
        raise FileNotFoundError("No .joblib model file found inside the tar.gz file.")

    model_path = os.path.join(extract_dir, joblib_files[0])
    return joblib.load(model_path)


@st.cache_resource
def load_explainer_from_s3():
    s3_client = boto_session.client("s3")

    local_explainer_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer_filename"])

    if not os.path.exists(local_explainer_path):
        s3_client.download_file(
            Bucket=aws_bucket,
            Key=f"{MODEL_INFO['explainer_s3_prefix']}/{MODEL_INFO['explainer_filename']}",
            Filename=local_explainer_path,
        )

    with open(local_explainer_path, "rb") as f:
        return shap.Explainer.load(f)

# ---------------------------
# ENDPOINT CALL
# ---------------------------
def call_model_api(user_inputs: dict):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        prediction = predictor.predict(user_inputs)
        pred_value = np.array(prediction).reshape(-1)[0]
        return float(pred_value), 200
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# ---------------------------
# SHAP DISPLAY
# ---------------------------
def display_explanation(user_inputs: dict):
    try:
        explainer = load_explainer_from_s3()
        best_pipeline = load_pipeline_from_s3()

        model_ready_df = convert_input_pca_regression(
            json.dumps(user_inputs),
            "application/json",
        )

        preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
        transformed = preprocessing_pipeline.transform(model_ready_df)

        try:
            transformed_feature_names = preprocessing_pipeline.get_feature_names_out()
        except:
            transformed_feature_names = [f"Feature_{i}" for i in range(transformed.shape[1])]

        transformed_df = pd.DataFrame(transformed, columns=transformed_feature_names)
        shap_values = explainer(transformed_df)

        st.subheader("🔍 SHAP Explanation")

        fig = plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

        impact_series = pd.Series(
            np.abs(shap_values[0].values),
            index=shap_values[0].feature_names
        ).sort_values(ascending=False)

        st.write("### Top feature impacts")
        st.dataframe(impact_series.head(10).reset_index().rename(
            columns={"index": "Feature", 0: "Absolute SHAP Impact"}
        ))

    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {str(e)}")

# ---------------------------
# USER INPUT UI
# ---------------------------
st.markdown("## Enter Four Feature Values")

with st.form("prediction_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, feature in enumerate(FEATURE_CONFIG):
        with cols[i % 2]:
            user_inputs[feature["name"]] = st.number_input(
                label=feature["label"],
                min_value=float(feature["min"]),
                max_value=float(feature["max"]),
                value=float(feature["default"]),
                step=float(feature["step"]),
                help=feature["help"],
            )

    submitted = st.form_submit_button("Run Prediction")

# ---------------------------
# RUN RESULT
# ---------------------------
if submitted:
    st.markdown("## Prediction Result")
    result, status = call_model_api(user_inputs)

    if status == 200:
        st.success("Prediction completed successfully.")
        st.metric("Predicted Future Return", round(result, 4))
        display_explanation(user_inputs)
    else:
        st.error(result)
