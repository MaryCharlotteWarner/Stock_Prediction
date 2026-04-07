import os
import json
import numpy as np
import pandas as pd


def convert_input_pca_regression(request_body, request_content_type):
    """
    Convert user-entered feature values into a single-row dataframe
    that matches the regression training feature structure.

    Steps:
    1. Load SP500 dataset
    2. Recreate transformed feature matrix used for training
    3. Find the closest historical row
    4. Replace the 4 selected feature values with user inputs
    5. Return the final dataframe for prediction / SHAP
    """

    if request_content_type != "application/json":
        raise ValueError("This app only supports application/json input.")

    user_vals = json.loads(request_body)

    possible_paths = [
        "SP500Data.csv",
        "/mnt/data/SP500Data (5).csv",
        os.path.join(os.getcwd(), "SP500Data.csv"),
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        raise FileNotFoundError("Could not find SP500Data.csv.")

    dataset = pd.read_csv(data_path, index_col=0)

    # Keep this aligned with your notebook
    target = "MSFT"
    return_period = 5

    selected_features = ["IBM_CR_Cum", "NVDA_CR_Cum", "GOOGL_CR_Cum", "AMD_CR_Cum"]

    # Recreate transformed X structure
    X = np.log(dataset.drop([target], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [f"{col}_CR_Cum" for col in X.columns]
    X = X.dropna().copy()

    missing_features = [f for f in selected_features if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing selected features in transformed dataset: {missing_features}")

    # Find closest row to keep other unseen features realistic
    distances = np.sqrt(
        sum((X[feature] - float(user_vals[feature])) ** 2 for feature in selected_features)
    )

    closest_index = distances.idxmin()
    closest_row = X.loc[[closest_index]].copy()

    # Override selected features with user values
    for feature in selected_features:
        closest_row.loc[:, feature] = float(user_vals[feature])

    return closest_row
