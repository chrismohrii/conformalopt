import numpy as np
from tqdm import tqdm
from conformalopt.data import get_scores
import itertools
import warnings
from conformalopt.utils import *
from statsmodels.tsa.forecasting.theta import ThetaModel

METHOD_HPARAMS = {
    "quantile_tracker": ["lr", "p_order_qt", "bias"],
    "ar_quantile_loss_scorecaster": ["lr", "p_order_ar_scorecaster"],
    "theta_scorecaster": ["lr"],
}

DEFAULT_GRIDS = {
    "lr": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5],
    "p_order_qt": [0, 1, 2],
    "p_order_ar_scorecaster": [1],
    "bias": [1e-1, 1e0, 5, 1e1, 200, 1e2, 1e3],
}


class ConformalPredictor:
    """
    A class for implementing an online conformal predictor as in CITE.

    Attributes:
        alpha (float): Target miscoverage level. If set to 0.1, for example, eventually 90% coverage will be achieved.
        lr_type (str): Type of learning rate, can be `fixed`, `decaying`, or `proportional`.
            The `decaying` learning rate is of the form \\Theta(t**{-0.6}). The `proportional` option
            multiplies the learning rate by the range of scores over the last 20 scores.
        quantile_track (str or None): Type of quantile tracker, can be `scalar` or `linear`.
        scorecaster (str or None): Scorecasting method, either `theta_scorecaster` or `ar_quantile_loss_scorecaster`, or None.
        hypers (dict): Specified hyperparameters for the conformal predictor. If none are passed in, all appropriate
            hyperparameters will be tuned.
        conformal_predictor_name (str): Name of the conformal predictor to be used in out/ folder during evaluation.
            If the empty string is passed in, a name will be constructed based on the conformal predictor's attributes.
        expt_name (str): Experiment name for evaluation.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        lr_type: str = "fixed",
        quantile_tracker="linear",
        scorecaster=None,
        hypers={},
        conformal_predictor_name="",
        expt_name="expt",
    ):
        """
        Initializes the ConformalPredictor with given hyperparameters.

        Parameters:
            alpha (float): Target miscoverage level.
            lr_type (str): Type of learning rate (`fixed`, `decaying`, `proportional`).
            quantile_tracker (str): Type of quantile tracker (`scalar`, `linear`).
            scorecaster (str): Scorecasting method (e.g., `theta_scorecaster`).
            hypers (dict): Hyperparameters to initialize the model.
            conformal_predictor_name (str): Name of the conformal predictor used in eval.
            expt_name (str): Experiment name for evaluation.
        """
        assert quantile_tracker in [None, "scalar", "linear"]
        assert scorecaster in [None] + [key for key in METHOD_HPARAMS.keys() if "_scorecaster" in key]
        assert lr_type in ["fixed", "decaying", "proportional"]

        self.alpha = alpha
        self.quantile_track = quantile_tracker
        self.scorecaster = scorecaster
        self.lr_type = lr_type
        self.conformal_predictor_name = conformal_predictor_name
        self.expt_name = expt_name
        self.past_predictions = []
        self.past_test_scores = []
        self.quantile_tracker_parameter = None

        # Set up hyperparameters
        if self.quantile_track == "scalar":
            hypers.update({"p_order_qt": 0, "bias": 1})
        self.hypers = hypers.copy()
        self.original_hypers = hypers.copy()

        for method, condition in [
            ("quantile_tracker", self.quantile_track),
            (self.scorecaster, self.scorecaster is not None),
        ]:
            if condition:
                for hparam in METHOD_HPARAMS[method]:
                    self.hypers.setdefault(hparam, None)

    def init_active_fields(self):
        if self.quantile_track is not None:
            p_order = self.hypers["p_order_qt"]
            self.quantile_tracker_parameter = np.ones(p_order + 1)
            if p_order > 0:
                self.quantile_tracker_parameter *= 1 / p_order
        self.past_predictions = []
        self.past_test_scores = []

    def get_predictions(self):
        """
        Retrieves the past predictions made by the conformal predictor.

        Returns:
            list: The list of past predictions.
        """
        return self.past_predictions

    def set_hypers(self, hyper_dict):
        """
        Sets the hyperparameters for the model.

        Parameters:
            hyper_dict (dict): A dictionary of hyperparameters to set.
        """

        for hyper in hyper_dict.keys():
            self.hypers[hyper] = hyper_dict[hyper]

    def reset_hypers(self):
        """
        Resets the hyperparameters to their original values.
        """
        self.hypers = self.original_hypers.copy()

    def fit(self, val_scores, tune_all_hparams=False, specific_hypers_to_tune: list = [], hyper_grid_overrides={}):
        """
        Tunes the hyperparameters for the conformal predictor using a grid search on validation
        data. The hyperparameters leading to the best quantile loss, provided 1-alpha-0.01
        coverage was achieved, are selected. If no selection achieves this coverage constraint,
        the coverage constraint is disregarded.

        Parameters:
            val_scores (list): The validation scores to use for tuning hyperparameters.
            tune_all_hparams (bool): Whether to tune all hyperparameters or only those currently unspecified.
            specific_hypers_to_tune (list): Specific hyperparameters to tune, if provided.
            hyper_grid_overrides (dict): New grids to tune hyperparameters on, beyond default grid.
        """
        T_val = len(val_scores)
        hyper_grid = DEFAULT_GRIDS | hyper_grid_overrides

        # Determine which hypers to tune.
        hypers_to_tune = [hyper for hyper in self.hypers.keys() if self.hypers[hyper] == None] + specific_hypers_to_tune
        if tune_all_hparams:
            hypers_to_tune = self.hypers.keys()
        grid = [[(hyper_name, hyper_value) for hyper_value in hyper_grid[hyper_name]] for hyper_name in hypers_to_tune]
        hyper_choices = [{hyper: value for hyper, value in combination} for combination in itertools.product(*grid)]

        # Perform grid search.
        best_quantile_loss, best_hypers = np.inf, None

        cov_gap, coverage_achieved = 0.01, False
        best_quantile_loss_coverage_constrained, best_hypers_coverage_constrained = (
            np.inf,
            None,
        )

        for hypers in tqdm(hyper_choices):
            self.hypers = self.original_hypers | hypers
            self.init_active_fields()
            for t in range(T_val):
                prediction = self.predict()
                self.step(prediction, val_scores[t])

            curr_quantile_loss = quantile_loss(val_scores, self.past_predictions, 1 - self.alpha)
            curr_run_coverage_achieved = (
                np.abs(np.mean(np.array(self.past_predictions) >= val_scores) - (1 - self.alpha)) <= cov_gap
            )

            coverage_achieved |= curr_run_coverage_achieved

            if curr_quantile_loss < best_quantile_loss:
                best_quantile_loss = curr_quantile_loss
                best_hypers = self.hypers
            if curr_run_coverage_achieved and curr_quantile_loss < best_quantile_loss_coverage_constrained:
                best_quantile_loss_coverage_constrained = curr_quantile_loss
                best_hypers_coverage_constrained = self.hypers

        self.hypers = best_hypers_coverage_constrained if curr_run_coverage_achieved else best_hypers
        print(f"Tuned hyperparameters: {self.hypers}")
        self.init_active_fields()

        for key, value in self.hypers.items():
            if key in ["lr", "bias"] and (value == hyper_grid[key][0] or value == hyper_grid[key][-1]):
                warnings.warn(f"{key}:{value} is at the edge of hyperparameter grid.")

    def quantile_tracker_predict(self):
        return self.quantile_tracker_parameter @ self.get_covariate()

    def scorecaster_predict(self):
        training_scores = np.array(self.past_test_scores[-200:])
        if len(training_scores) >= 10:
            if self.scorecaster == "theta_scorecaster":
                model = ThetaModel(
                    training_scores.astype(float),
                    period=1,
                ).fit()
                return model.forecast(1).iloc[0]
            elif self.scorecaster == "ar_quantile_loss_scorecaster":
                theta = fit_ar_quantile_loss(training_scores, self.hypers["p_order_ar_scorecaster"], self.alpha)
                return theta @ self.get_covariate(self.hypers["p_order_ar_scorecaster"])
        return 0

    def predict(self):
        """
        Makes a prediction by combining the quantile_tracker_predict and scorecaster_predict.

        Returns:
            float: The final prediction.
        """
        prediction = sum(
            fn()
            for fn, condition in [
                (self.quantile_tracker_predict, self.quantile_track),
                (self.scorecaster_predict, self.scorecaster is not None),
            ]
            if condition
        )

        self.past_predictions.append(prediction)
        return prediction

    def get_covariate(self, p_order=None):
        """Returns the covariate or feature vector based on past test scores.
        This is the last p_order scores along with a bias term (1 appended),
        and could be updated for richer feature vectors.

        Args:
            p_order (int, optional): The number of past test scores to include.
                If None, defaults to `self.hypers["p_order_qt"]`.

        Returns:
            np.ndarray: A numpy array of shape `(p_order + 1,)` containing the past
            test scores (if available) and the bias term.
        """
        if self.quantile_track is not None:
            bias = self.hypers["bias"]
            if p_order == None:
                p_order = self.hypers["p_order_qt"]
            if len(self.past_test_scores) < p_order:
                return np.zeros(p_order + 1)
            return np.concatenate((self.past_test_scores[-p_order:] if p_order > 0 else [], [bias]))

    # Updates quantile tracker.
    def step(self, prediction, realized_score):
        """
        Updates the quantile tracker based on the realized score and prediction. The update
        follows one step of gradient descent with the quantile loss on the quantile tracker's
        parameter.

        Parameters:
            prediction (float): The predicted score.
            realized_score (float): The actual observed score.
        """
        if self.quantile_track is not None:
            time = len(self.past_test_scores) + 1
            lr_functions = {
                "fixed": lambda: self.hypers["lr"],
                "decaying": lambda: self.hypers["lr"] * (time**-0.6),
                "proportional": lambda: self.hypers["lr"]
                * (
                    (
                        np.array(self.past_test_scores[max(time - 20, 0) :]).max()
                        - np.array(self.past_test_scores[max(time - 20, 0) :]).min()
                    )
                    if len(self.past_test_scores) > 0
                    else 1
                ),
            }
            lr = lr_functions[self.lr_type]()
            error = realized_score > prediction
            self.quantile_tracker_parameter += lr * (error - self.alpha) * self.get_covariate()

        # Update history of scores
        self.past_test_scores.append(realized_score)

    def eval(self, checkpoint_name=""):
        """
        Evaluates the model and outputs the results in out/expt_name/checkpoint_name.
        If another ConformalPredictor has already produced evaluation results in
        this location, the results are combined. If `checkpoint_name` is not provided,
        then it is set to the current time step.

        Parameters:
            checkpoint_name (str): The checkpoint name for this evaluation.
        """
        if checkpoint_name == "":
            checkpoint_name = str(len(self.past_test_scores))
        if self.conformal_predictor_name == "":
            if self.quantile_track == "scalar":
                self.conformal_predictor_name = "SQT"
            elif self.quantile_track == "linear":
                self.conformal_predictor_name = f'LQT({self.hypers["p_order_qt"]})'
            if self.scorecaster is not None:
                self.conformal_predictor_name += f" + {self.scorecaster}"
            self.conformal_predictor_name += f", {self.lr_type}"
        eval(
            self.past_predictions,
            self.past_test_scores,
            self.alpha,
            self.expt_name,
            checkpoint_name,
            self.conformal_predictor_name,
            self.hypers,
        )


if __name__ == "__main__":
    data_abbr = "GOOGL"
    model_type = "theta"
    data_name = f"{data_abbr}_{model_type}_absolute-residual_scores"
    scores = get_scores(data_name)
    split = int(0.33 * len(scores))
    val_scores = scores[:split]
    test_scores = scores[split:]

    cp = ConformalPredictor(
        lr_type="decaying",
        quantile_tracker="linear",
        expt_name=f"{data_abbr}_{model_type}",
    )

    # Fit the method's hyperparameters (such as the dimension of the quantile tracker)
    # using a grid search on validation scores.
    cp.fit(val_scores=list(val_scores))

    for t in range(len(test_scores)):
        prediction = cp.predict()  # Make a prediction for the score.
        cp.step(prediction, test_scores[t])  # Update the quantile tracker parameter.

    cp.eval()  # Provides an evaluation in out/expt_name folder, alongside other ConformalPredictor objects if they exist.
