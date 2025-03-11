import numpy as np
import matplotlib.pyplot as plt
import os
import json
import statsmodels.api as sm

METRICS = [
    "coverage",
    "quantile_loss",
    "set_size",
    "absolute_loss",
    "square_loss",
    "pos_excess",
    "neg_excess",
]


def convert_ndarray(obj):
    """Recursively convert numpy.ndarrays to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    return obj


def construct_ar_features(training_scores, p_order, max_offset=200):
    """
    Constructs autoregressive (AR) feature vectors from training scores.

    For each observation in the index range
        [max(p_order, len(training_scores) - max_offset), len(training_scores)),
    a feature vector is constructed from the previous `p_order` values (via a sliding window)
    with a constant 1 appended as the bias term.

    Parameters:
        training_scores (np.ndarray): 1D array of training scores.
        p_order (int): Number of lagged features (window length).
        max_offset (int, optional): Maximum number of recent observations to use (default: 200).

    Returns:
        np.ndarray: 2D array of shape (n, p_order + 1), where
                    n = len(training_scores) - max(p_order, len(training_scores) - max_offset).
    """
    n = len(training_scores)
    start = max(p_order, n - max_offset)
    num_points = n - start

    # Construct windows
    if hasattr(np.lib.stride_tricks, "sliding_window_view"):
        windows = np.lib.stride_tricks.sliding_window_view(training_scores, window_shape=p_order)
        windows_subset = windows[start - p_order : n - p_order]
    else:
        windows_subset = np.array([training_scores[t - p_order : t] for t in range(start, n)])

    # Add bias
    ones = np.ones((num_points, 1), dtype=training_scores.dtype)
    ar_features = np.hstack((windows_subset, ones))
    return ar_features


def fit_ar_quantile_loss(training_scores, p_order, alpha, max_offset=200):
    """
    Fits a quantile regression model using autoregressive features derived from training scores.

    This function constructs AR feature vectors for the most recent observations (up to
    `max_offset`), then fits a quantile regression model using statsmodels. The regression
    is performed on the same subset of the training data used for feature construction.
    The function returns the estimated regression coefficients.

    Parameters:
        training_scores (np.ndarray): Array of training scores.
        p_order (int): Order of AR model.
        alpha (float): Significance level for the quantile regression; the quantile is set to 1 - alpha.
        max_offset (int, optional): Maximum number of recent observations to use. Default is 200.

    Returns:
        np.ndarray: Array containing the estimated regression coefficients.
    """
    n = len(training_scores)
    start = max(p_order, n - max_offset)
    # Construct AR features using the most recent observations.
    X = construct_ar_features(training_scores, p_order, max_offset)
    # Use the same subset of training_scores as the dependent variable.
    y = training_scores[start:]

    model = sm.QuantReg(y, X)
    result = model.fit(q=1 - alpha, max_iter=5000, cov_type="none")
    return np.array(result.params)


def smooth_array(arr, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(arr, window, mode="valid")


def quantile_loss(y_true, y_pred, quantile):
    assert 0 <= quantile <= 1, "Quantile should be between 0 and 1"
    error = np.array(y_true) - np.array(y_pred)
    loss = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def absolute_loss(y_true, y_pred):
    error = np.abs(np.array(y_true) - np.array(y_pred))
    return np.mean(error)


def square_loss(y_true, y_pred):
    error = np.array(y_true) - np.array(y_pred)
    return np.mean(error**2)


def load_results(file_path):
    try:
        with open(file_path) as file:
            return json.load(file) or {}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def calculate_metrics(predictions, test_scores, alpha):
    """Calculate and update metrics for a method."""
    test_scores, predictions = np.array(test_scores), np.array(predictions)
    quantile = quantile_loss(test_scores, predictions, 1 - alpha)
    absolute = absolute_loss(test_scores, predictions)
    squared = square_loss(test_scores, predictions)
    coverage = np.array(predictions) >= np.array(test_scores)
    set_size = np.mean(predictions)

    return {
        "quantile_loss": quantile,
        "absolute_loss": absolute,
        "square_loss": squared,
        "set_size": set_size,
        "coverage": np.mean(coverage),
        "pos_excess": np.mean(predictions[coverage] - test_scores[coverage]),
        "neg_excess": np.mean(predictions[~coverage] - test_scores[~coverage]),
    }


def plot_rolling_coverage(rolling_coverage_dir, predictions, scores, conformal_predictor_name, alpha):

    os.makedirs(rolling_coverage_dir, exist_ok=True)
    plt.clf()
    plt.plot(smooth_array((np.array(predictions) >= np.array(scores)).astype(int), 500))
    plt.title(f"{conformal_predictor_name} rolling coverage")
    plt.xlabel("Time")
    plt.ylabel("Coverage")
    plt.axhline(y=1 - alpha, color="r", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{rolling_coverage_dir}/{conformal_predictor_name}.pdf", dpi=300)


def plot_metric_bars(values, categories, title, ylabel, path, hline=None):
    """Plot and save bar charts."""
    plt.clf()
    plt.bar(categories, values)
    for i, val in enumerate(values):
        plt.text(i, val + 0.1, f"{val:.3f}", ha="center", fontsize=8, rotation=33)
    plt.xticks(rotation=36, ha="right")
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    if hline:
        plt.axhline(y=hline, color="r", linestyle="--")
    plt.tight_layout()
    plt.savefig(path, dpi=300)


def eval(predictions, scores, alpha, expt_name, checkpoint_name, conformal_predictor_name, hypers):
    assert len(predictions) > 1
    """Main entry function to perform evaluations and generate plots."""
    checkpoint_dir = f"out/{expt_name}/{checkpoint_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    saved_results = load_results(f"{checkpoint_dir}/saved_results.json")
    saved_results[conformal_predictor_name] = calculate_metrics(predictions, scores, alpha)

    for metric in METRICS:  # Update this.
        values = [saved_results[conformal_predictor][metric] for conformal_predictor in saved_results]
        categories = list(saved_results.keys())
        plot_metric_bars(
            values,
            categories,
            metric,
            metric,
            f"{checkpoint_dir}/{metric}.pdf",
            hline=(1 - alpha if metric == "coverage" else None),
        )

    saved_results[conformal_predictor_name]["hypers"] = hypers

    # Add rolling coverage
    if len(predictions) > 500:
        rolling_coverage_dir = f"{checkpoint_dir}/rolling_coverage/"
        plot_rolling_coverage(rolling_coverage_dir, predictions, scores, conformal_predictor_name, alpha)

    # Convert all ndarrays in saved_results
    saved_results = convert_ndarray(saved_results)

    # Save the updated results
    with open(f"{checkpoint_dir}/saved_results.json", "w") as file:
        json.dump(saved_results, file, indent=4)
