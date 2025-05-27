import numpy as np
import pandas as pd
import time
import os


def construct_file_path(data_name):

    # Get the directory where this script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Construct the absolute path to the file
    return os.path.join(BASE_DIR, "data", f"{data_name}.csv")


def get_scores(data_name, scores=None):
    """
    Loads and processes scores for a specified dataset. The scores are calculated as
    |Y_t - hat Y_t| for various base forecasters hat Y_t. The datasets are all described
    in detail in the paper Online Conformal Prediction via
    Online Optimization. The datasets of the form name* should be input as f"{name}_{base_forecaster}_absolute-residual_scores"
    for base_forecaster as `ar`, `prophet`, `theta`, or `transformer`.


    Args:
        data_name (str): The name of the dataset. Supported options:

            - `elec`: Elec2 data with base forecaster being a one-day delayed moving average. \n
            - `daily-climate*`: Daily climate data.
            - `AMZN*`, `GOOGL*`, `MSFT*`: Stock data.
            - `synthetic_AR_2_1M`: 1_000_000 synthetic AR(2) data generated with [0.3, -0.3] AR parameters and standard normal noise.
            - `gaussian`: 10_000 i.i.d. Gaussian-distributed synthetic scores.
            - `ercot_preregistered`: ERCOT load and forecast data. This is the preregistered dataset used in the paper.

    Returns:
        np.ndarray: A processed score array.
    """
    if data_name == "elec":  # length 45264
        # Score is |Y_t - \hat Y_t| where \hat Y_t is a one-day delayed moving average
        data = pd.read_csv(construct_file_path("electricity-normalized.csv"))
        Y = data["nswdemand"].to_numpy()
        # Bug in PID paper code: actually predicting one-day delayed moving average now, as paper claims.
        Yhat = [np.mean(Y[i : i + 24]) for i in range(len(Y[48:]))]
        Y = Y[48:]
        scores = np.abs(Y - Yhat)
    elif data_name.startswith("synthetic_AR_2_1M"):

        np.random.seed(int(data_name.split("_")[-1]))
        # Parameters
        n = 1_000_000  # Length of the time series
        phi = [0.3, -0.3]  # AR(2) parameters
        sigma = 1.0  # Standard deviation of the noise

        # Generate white noise
        noise = np.random.normal(0, sigma, n)

        # Initialize the time series
        y = np.zeros(n)

        # Generate the AR(2) time series
        for t in range(2, n):
            y[t] = phi[0] * y[t - 1] + phi[1] * y[t - 2] + noise[t]

        scores = y
    elif data_name == "gaussian":
        scores = []
        for i in range(10_000):
            scores.append(np.random.normal(scale=i // 10))
        scores = np.array(scores)
    elif data_name == "ercot_preregistered":
        import gridstatusio as gs

        API_KEY = "1a692d6abfa547bfb58911dd29a3f088"
        START_TIME = "2024-12-18"
        END_TIME = "2025-01-04"

        # Collect scores via API.
        client = gs.GridStatusClient(API_KEY)
        df_data = client.get_dataset(dataset="ercot_load", start=START_TIME, end=END_TIME)
        time.sleep(1)  # To avoid API rate limit hit.
        df_forecasts = client.get_dataset(dataset="ercot_load_forecast", start=START_TIME, end=END_TIME)
        df_merged = pd.merge(df_data, df_forecasts, on="interval_start_utc", how="inner")
        scores = np.abs(df_merged["load"] - df_merged["load_forecast"]).values[200:]

    else:
        # Stocks (daily-climate, AMZN, GOOGL, MSFT) routed here.
        filename = construct_file_path(data_name)
        scores = np.loadtxt(filename)

    # Sometimes the first few scores in these datasets are nonsense.
    return scores[30:]
