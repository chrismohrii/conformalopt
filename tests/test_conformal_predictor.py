import unittest
import numpy as np
import copy
import warnings
from unittest.mock import patch, MagicMock

from onlinecp import (
    ConformalPredictor,
    METHOD_HPARAMS,
    DEFAULT_GRIDS,
)


# Dummy functions to replace external dependencies.
def dummy_quantile_loss(val_scores, predictions, q):
    """Return the mean absolute error between true and predicted values."""
    return np.mean(np.abs(np.array(val_scores) - np.array(predictions)))


def dummy_eval(predictions, test_scores, alpha, expt_name, checkpoint_name, cp_name, hypers):
    """A dummy evaluation function that does nothing."""
    pass


class TestConformalPredictor(unittest.TestCase):

    def test_init_validations(self):
        # Check that invalid parameter values raise assertions.
        with self.assertRaises(AssertionError):
            ConformalPredictor(quantile_tracker="invalid_tracker")
        with self.assertRaises(AssertionError):
            ConformalPredictor(scorecaster="non_existent_scorecaster")
        with self.assertRaises(AssertionError):
            ConformalPredictor(lr_type="bad_lr")
        # Valid initialization should succeed.
        cp = ConformalPredictor(quantile_tracker="linear", scorecaster="theta_scorecaster", lr_type="fixed")
        self.assertEqual(cp.quantile_track, "linear")
        self.assertEqual(cp.scorecaster, "theta_scorecaster")
        self.assertEqual(cp.lr_type, "fixed")

    def test_init_active_fields(self):
        # Verify that active fields are initialized correctly.
        hypers = {"p_order_qt": 2, "bias": 1, "lr": 0.01}
        cp = ConformalPredictor(quantile_tracker="linear", hypers=hypers)
        cp.init_active_fields()
        expected = np.ones(3) * 0.5  # p_order 2 -> vector length 3 with each element 1/2
        np.testing.assert_allclose(cp.quantile_tracker_parameter, expected)
        self.assertEqual(cp.past_predictions, [])
        self.assertEqual(cp.past_test_scores, [])

    def test_get_predictions(self):
        # Check that get_predictions returns the past predictions.
        cp = ConformalPredictor(quantile_tracker="scalar")
        cp.past_predictions = [3.5, 4.2, 5.0]
        self.assertEqual(cp.get_predictions(), [3.5, 4.2, 5.0])

    def test_set_and_reset_hypers(self):
        # Test that setting and resetting hyperparameters works as expected.
        initial = {"lr": 0.1, "p_order_qt": 1, "bias": 1}
        cp = ConformalPredictor(quantile_tracker="linear", hypers=initial)
        cp.set_hypers({"lr": 0.05})
        self.assertEqual(cp.hypers["lr"], 0.05)
        cp.reset_hypers()
        self.assertEqual(cp.hypers["lr"], initial["lr"])

    def test_quantile_tracker_predict(self):
        # Ensure the quantile tracker computes the correct prediction.
        hypers = {"p_order_qt": 2, "bias": 1, "lr": 0.01}
        cp = ConformalPredictor(quantile_tracker="linear", hypers=hypers)
        cp.init_active_fields()
        cp.past_test_scores = [2, 4, 6]  # get_covariate() -> [4, 6, 1]
        np.testing.assert_array_equal(cp.get_covariate(), np.array([4, 6, 1]))
        expected_param = np.ones(3) * 0.5
        np.testing.assert_allclose(cp.quantile_tracker_parameter, expected_param)
        # Dot product: 0.5*4 + 0.5*6 + 0.5*1 = 5.5
        self.assertAlmostEqual(cp.quantile_tracker_predict(), 5.5)

    def test_scorecaster_predict_insufficient_scores(self):
        # When there are fewer than 10 test scores, scorecaster_predict should return 0.
        cp = ConformalPredictor(scorecaster="theta_scorecaster")
        cp.past_test_scores = [1, 2, 3]
        self.assertEqual(cp.scorecaster_predict(), 0)

    @patch("onlinecp.main.ThetaModel")
    def test_scorecaster_predict_theta(self, mock_theta_model):
        # Test theta_scorecaster branch.
        cp = ConformalPredictor(scorecaster="theta_scorecaster", hypers={"p_order_qt": 1, "bias": 1, "lr": 0.01})
        cp.past_test_scores = list(range(10))
        fake_forecast = MagicMock()
        fake_forecast.iloc = [5.0]
        instance = MagicMock()
        instance.fit.return_value.forecast.return_value = fake_forecast
        mock_theta_model.return_value = instance

        self.assertEqual(cp.scorecaster_predict(), 5.0)
        call_args, call_kwargs = mock_theta_model.call_args
        np.testing.assert_array_equal(call_args[0], np.array(cp.past_test_scores[-200:]).astype(float))
        self.assertEqual(call_kwargs["period"], 1)

    @patch("onlinecp.main.fit_ar_quantile_loss", return_value=np.array([2, 3]))
    def test_scorecaster_predict_ar(self, mock_fit_ar):
        # Test the AR quantile loss scorecaster branch.
        cp = ConformalPredictor(
            scorecaster="ar_quantile_loss_scorecaster", hypers={"p_order_ar_scorecaster": 1, "lr": 0.01}
        )
        # Provide at least 10 scores; covariate will be [last_score, bias] -> [10, 1]
        cp.past_test_scores = list(range(1, 10)) + [10]
        cp.hypers["bias"] = 1
        self.assertEqual(cp.scorecaster_predict(), 23)

    def test_predict(self):
        # Verify that predict returns the sum of tracker and scorecaster predictions.
        hypers = {"p_order_qt": 0, "bias": 1, "lr": 0.01}
        cp = ConformalPredictor(quantile_tracker="scalar", scorecaster="theta_scorecaster", hypers=hypers)
        cp.quantile_tracker_predict = lambda: 3.0
        cp.scorecaster_predict = lambda: 2.0
        cp.past_predictions = []
        pred = cp.predict()
        self.assertEqual(pred, 5.0)
        self.assertEqual(cp.past_predictions, [5.0])

    def test_get_covariate(self):
        # Test get_covariate when there are insufficient and sufficient test scores.
        hypers = {"p_order_qt": 3, "bias": 10, "lr": 0.01}
        cp = ConformalPredictor(quantile_tracker="linear", hypers=hypers)
        cp.past_test_scores = [1, 2]  # Not enough scores: should return zeros.
        np.testing.assert_array_equal(cp.get_covariate(), np.zeros(4))
        cp.past_test_scores = [1, 2, 3, 4]
        np.testing.assert_array_equal(cp.get_covariate(), np.array([2, 3, 4, 10]))

    def test_step(self):
        # Test that step updates parameters and appends the new score.
        hypers = {"p_order_qt": 1, "bias": 1, "lr": 0.01}
        cp = ConformalPredictor(quantile_tracker="linear", hypers=hypers, alpha=0.1, lr_type="fixed")
        cp.init_active_fields()
        cp.past_test_scores = [5]  # Covariate becomes [5, 1]
        cp.quantile_tracker_parameter = np.array([1.0, 1.0])
        prediction, realized_score = 7.0, 9.0  # Error: realized_score > prediction
        cp.step(prediction, realized_score)
        np.testing.assert_allclose(cp.quantile_tracker_parameter, np.array([1.045, 1.009]))
        self.assertEqual(cp.past_test_scores[-1], realized_score)

    def test_fit(self):
        # Use small grids to simplify hyperparameter tuning during testing.
        small_grids = {"lr": [0.01], "p_order_qt": [1], "bias": [1]}
        orig_grids = DEFAULT_GRIDS.copy()
        DEFAULT_GRIDS.update(small_grids)

        cp = ConformalPredictor(quantile_tracker="linear", hypers={}, alpha=0.1, lr_type="fixed")
        T_val = 15

        # Fake predict function to simulate different outcomes.
        def fake_predict():
            idx = getattr(cp, "predict_counter", 0)
            setattr(cp, "predict_counter", idx + 1)
            p = cp._val_scores[idx] - 1 if idx == 0 else cp._val_scores[idx] + 1
            cp.past_predictions.append(p)
            return p

        cp.predict = fake_predict
        cp.step = lambda pred, real: cp.past_test_scores.append(real)
        cp._val_scores = np.linspace(10, 19, T_val)

        with patch("onlinecp.utils.quantile_loss", side_effect=dummy_quantile_loss):
            with self.assertWarns(UserWarning):
                cp.fit(val_scores=list(cp._val_scores), tune_all_hparams=True)
        self.assertEqual(cp.hypers, {"lr": 0.01, "p_order_qt": 1, "bias": 1})
        DEFAULT_GRIDS.clear()
        DEFAULT_GRIDS.update(orig_grids)

    def test_eval(self):
        # Verify that the eval method constructs the proper checkpoint and predictor names.
        hypers = {"p_order_qt": 0, "bias": 1, "lr": 0.01}
        cp = ConformalPredictor(
            quantile_tracker="scalar", scorecaster="theta_scorecaster", hypers=hypers, expt_name="test_expt"
        )
        cp.past_predictions = [1, 2, 3]
        cp.past_test_scores = [0.5, 1.5, 2.5]
        cp.conformal_predictor_name = ""
        with patch("onlinecp.main.eval", side_effect=dummy_eval) as mock_eval:
            cp.eval(checkpoint_name="")
            expected_checkpoint = str(len(cp.past_test_scores))
            expected_name = "SQT" if cp.quantile_track == "scalar" else f'LQT({cp.hypers.get("p_order_qt", "NA")})'
            if cp.scorecaster is not None:
                expected_name += f" + {cp.scorecaster}"
            expected_name += f", {cp.lr_type}"
            mock_eval.assert_called_with(
                cp.past_predictions,
                cp.past_test_scores,
                cp.alpha,
                cp.expt_name,
                expected_checkpoint,
                expected_name,
                cp.hypers,
            )

    def test_choose_config_with_lower_quantile_loss_among_coverage(self):
        # Create a predictor with 'lr' to be tuned.
        cp = ConformalPredictor(
            lr_type="fixed", quantile_tracker="linear", hypers={"p_order_qt": 1, "bias": 1, "lr": None}
        )
        # Validation scores: all 10's.
        val_scores = [10] * 10

        # Override init_active_fields and step.
        cp.init_active_fields = lambda: (setattr(cp, "past_predictions", []), setattr(cp, "_predict_index", 0))
        cp.step = lambda pred, real: None

        # Fake predict function using a predefined mapping for each lr value.
        def fake_predict():
            fake_preds = {
                0.1: [10] * 9 + [0],  # 9/10 predictions >= 10 → coverage 0.9; quantile loss ~0.9
                0.2: [10] * 9 + [5],  # 9/10 predictions >= 10 → coverage 0.9; quantile loss ~0.45
            }
            current_lr = cp.hypers["lr"]
            idx = cp._predict_index
            cp._predict_index += 1
            pred_val = fake_preds[current_lr][idx]
            cp.past_predictions.append(pred_val)
            return pred_val

        cp.predict = fake_predict

        # Temporarily restrict the lr grid to the two test values.
        original_lr_grid = copy.deepcopy(DEFAULT_GRIDS["lr"])
        DEFAULT_GRIDS["lr"] = [0.1, 0.2]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lr:0.2 is at the edge of hyperparameter grid.")
            warnings.filterwarnings("ignore", message="bias:0.1 is at the edge of hyperparameter grid.")
            cp.fit(val_scores, tune_all_hparams=True)
        # Expect that the configuration with lr=0.2 (lower quantile loss) is chosen.
        self.assertEqual(cp.hypers["lr"], 0.2)

        DEFAULT_GRIDS["lr"] = original_lr_grid

    def test_ignore_config_with_lower_quantile_loss_if_no_coverage(self):
        # Create a predictor with 'lr' to be tuned.
        cp = ConformalPredictor(
            lr_type="fixed", quantile_tracker="linear", hypers={"p_order_qt": 1, "bias": 1, "lr": None}
        )
        # Validation scores: all 10's.
        val_scores = np.array([10] * 10)

        cp.init_active_fields = lambda: (setattr(cp, "past_predictions", []), setattr(cp, "_predict_index", 0))
        cp.step = lambda pred, real: None

        def fake_predict():
            fake_preds = {
                0.3: [10] * 10,  # All predictions equal 10 → coverage 1.0 (unacceptable, diff=0.1)
                0.4: [10] * 9 + [0],  # 9/10 predictions equal 10 → coverage 0.9 (acceptable)
            }
            current_lr = cp.hypers["lr"]
            idx = cp._predict_index
            cp._predict_index += 1
            pred_val = fake_preds[current_lr][idx]
            cp.past_predictions.append(pred_val)
            return pred_val

        cp.predict = fake_predict

        original_lr_grid = copy.deepcopy(DEFAULT_GRIDS["lr"])
        DEFAULT_GRIDS["lr"] = [0.3, 0.4]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lr:0.4 is at the edge of hyperparameter grid.")
            warnings.filterwarnings("ignore", message="bias:0.1 is at the edge of hyperparameter grid.")
            cp.fit(val_scores, tune_all_hparams=True)
        # Although lr=0.3 would yield lower quantile loss, its coverage (1.0) misses the target.
        # Therefore, expect lr=0.4 to be chosen.
        self.assertEqual(cp.hypers["lr"], 0.4)

        DEFAULT_GRIDS["lr"] = original_lr_grid


if __name__ == "__main__":
    unittest.main()
