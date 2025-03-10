import unittest
import numpy as np
import time
import cvxpy as cp
import statsmodels.api as sm
from onlineconformal.utils import fit_ar_quantile_loss, construct_ar_features


## Legacy quantile regression implementation
def fit_ar_quantile_loss_legacy(training_scores, p_order, alpha, max_offset=200):
    T = len(training_scores)
    X = []
    y = np.empty(0)

    phi = np.zeros((T, p_order + 1))
    for t in range(max(p_order, T - max_offset), T):
        phi = np.concatenate((training_scores[t - p_order : t], [1]))
        X.append(phi)
        y = np.append(y, training_scores[t])

    X_b = np.array(X)
    quantile = 1 - alpha
    beta = cp.Variable(X_b.shape[1])
    residuals = y - X_b @ beta
    quantile_loss = cp.sum(cp.maximum(quantile * residuals, (quantile - 1) * residuals))

    problem = cp.Problem(cp.Minimize(quantile_loss))
    problem.solve()

    return np.array(beta.value)


## Quantile loss helper function
def compute_quantile_loss(training_scores, p_order, alpha, max_offset, beta):
    """
    Computes the quantile loss for the given regression coefficients.

    The loss is computed over the subset of training data used to construct AR features.

    Parameters:
        training_scores (np.ndarray): Array of training scores.
        p_order (int): Order of AR model.
        alpha (float): Significance level for the quantile regression; the quantile is set to 1 - alpha.
        max_offset (int): Maximum number of recent observations used.
        beta (np.ndarray): Regression coefficients.

    Returns:
        float: The computed quantile loss.
    """
    n = len(training_scores)
    start = max(p_order, n - max_offset)
    X = construct_ar_features(training_scores, p_order, max_offset)
    y = training_scores[start:]
    residuals = y - X.dot(beta)
    return np.mean(np.maximum((1 - alpha) * residuals, alpha * residuals))


## Tests
class TestARQuantileRegression(unittest.TestCase):
    def test_quantile_loss_varied(self):
        """
        For a range of dataset sizes, AR orders, max_offsets, and alphas, compare the
        quantile loss from the statsmodels-based method to that from the legacy CVXPY-based method.
        Assert that the loss from the statsmodels method is less than or equal to the legacy loss
        (within a small tolerance).
        """
        np.random.seed(1)
        sizes = [100, 500, 1000]
        orders = [1, 3, 5]
        max_offsets = [50, 200, 500]
        alphas = [0.01, 0.05, 0.1]

        for size in sizes:
            for p_order in orders:
                if size < p_order:
                    continue  # Skip invalid configurations.
                for max_offset in max_offsets:
                    for alpha in alphas:
                        with self.subTest(size=size, p_order=p_order, max_offset=max_offset, alpha=alpha):
                            training_scores = np.random.randn(size)
                            beta_optim = fit_ar_quantile_loss(training_scores, p_order, alpha, max_offset)
                            beta_legacy = fit_ar_quantile_loss_legacy(training_scores, p_order, alpha, max_offset)
                            loss_optim = compute_quantile_loss(training_scores, p_order, alpha, max_offset, beta_optim)
                            loss_legacy = compute_quantile_loss(
                                training_scores, p_order, alpha, max_offset, beta_legacy
                            )

                            tol = 0.005  # Allowed tolerance in quantile loss.
                            self.assertLessEqual(
                                loss_optim,
                                loss_legacy + tol,
                                f"Statsmodels loss ({loss_optim:.4f}) exceeds legacy loss ({loss_legacy:.4f}) beyond tolerance {tol} "
                                f"for size={size}, p_order={p_order}, max_offset={max_offset}, alpha={alpha}",
                            )

    def test_execution_time_varied(self):
        """
        For a range of dataset sizes, AR orders, and max_offsets, compare the average execution times
        of the statsmodels-based method and the legacy CVXPY-based method. Assert that the statsmodels
        method runs faster within an allowed runtime tolerance.
        """
        np.random.seed(2)
        sizes = [20000, 30000, 40000]
        orders = [3, 5]
        max_offsets = [10000, 15000]
        alphas = [0.01]
        n_runs = 5
        runtime_tol = 0.05  # Allowable difference in average runtime (seconds)

        for size in sizes:
            for p_order in orders:
                if size < p_order:
                    continue
                for max_offset in max_offsets:
                    for alpha in alphas:
                        with self.subTest(size=size, p_order=p_order, max_offset=max_offset, alpha=alpha):
                            training_scores = np.random.randn(size)

                            start_time = time.perf_counter()
                            for _ in range(n_runs):
                                _ = fit_ar_quantile_loss(training_scores, p_order, alpha, max_offset)
                            time_optim = (time.perf_counter() - start_time) / n_runs

                            start_time = time.perf_counter()
                            for _ in range(n_runs):
                                _ = fit_ar_quantile_loss_legacy(training_scores, p_order, alpha, max_offset)
                            time_legacy = (time.perf_counter() - start_time) / n_runs

                            print(f"Size: {size}, p_order: {p_order}, max_offset: {max_offset}, alpha: {alpha}")
                            print(f"Statsmodels time: {time_optim:.4f}s, Legacy time: {time_legacy:.4f}s")

                            self.assertLessEqual(
                                time_optim,
                                time_legacy + runtime_tol,
                                f"Statsmodels version ({time_optim:.4f}s) is not faster than legacy version ({time_legacy:.4f}s) within tolerance {runtime_tol}s "
                                f"for size={size}, p_order={p_order}, max_offset={max_offset}, alpha={alpha}",
                            )


if __name__ == "__main__":
    unittest.main()
