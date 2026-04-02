import streamlit as st
import numpy as np
import boto3

# =========================
# AWS CONFIG (from secrets)
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# =========================
# CREATE CLIENT
# =========================
runtime = boto3.client(
    "sagemaker-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_id,
    aws_secret_access_key=aws_secret,
    aws_session_token=aws_token
)

# =========================
# STREAMLIT UI
# =========================
st.title("📈 Regression Model Prediction")

st.write("Enter values for your model:")

# ---- 496 INPUTS ----
num_inputs = 496
inputs = []

cols = st.columns(4)

for i in range(num_inputs):
    with cols[i % 4]:
        val = st.number_input(f"Feature {i+1}", value=0.0)
        inputs.append(val)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(data):
    arr = np.array(data)

    # ✅ FIX: convert 496 → 296
    if len(arr) > 296:
        arr = arr[:296]
    elif len(arr) < 296:
        arr = np.pad(arr, (0, 296 - len(arr)))

    payload = ",".join(map(str, arr))

    response = runtime.invoke_endpoint(
        EndpointName=aws_endpoint,
        ContentType="text/csv",
        Body=payload
    )

    result = response["Body"].read().decode("utf-8")
    return result

# =========================
# BUTTON
# =========================
if st.button("Run Prediction"):
    try:
        result = predict(inputs)
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
