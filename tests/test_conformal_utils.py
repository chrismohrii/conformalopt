import unittest
import numpy as np
import os
import json
import tempfile

from conformalopt import utils


class TestUtils(unittest.TestCase):

    def test_convert_ndarray_with_nested_structures(self):
        data = {"a": np.array([1, 2]), "b": [np.array([3, 4]), {"c": np.array([5])}], "d": 10}
        expected = {"a": [1, 2], "b": [[3, 4], {"c": [5]}], "d": 10}
        result = utils.convert_ndarray(data)
        self.assertEqual(result, expected)

    def test_construct_ar_features(self):
        training_scores = np.arange(10)
        p_order = 3
        features = utils.construct_ar_features(training_scores, p_order)
        expected_rows = 10 - 3
        self.assertEqual(features.shape, (expected_rows, p_order + 1))
        self.assertTrue(np.all(features[:, -1] == 1))
        np.testing.assert_array_equal(features[0, :3], np.array([0, 1, 2]))

    def test_fit_ar_quantile_loss(self):
        training_scores = np.linspace(1, 10, 10)
        p_order = 2
        alpha = 0.1
        coeffs = utils.fit_ar_quantile_loss(training_scores, p_order, alpha)
        self.assertEqual(coeffs.shape, (p_order + 1,))
        self.assertIsInstance(coeffs, np.ndarray)
        self.assertTrue(np.issubdtype(coeffs.dtype, np.floating))

    def test_smooth_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        window_size = 3
        smoothed = utils.smooth_array(arr, window_size)
        expected = np.convolve(arr, np.ones(window_size) / window_size, mode="valid")
        np.testing.assert_array_almost_equal(smoothed, expected)

    def test_load_results_file_not_found(self):
        result = utils.load_results("non_existent_file.json")
        self.assertEqual(result, {})

    def test_load_results_invalid_json(self):
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
            tmp.write("invalid json")
            tmp_path = tmp.name
        try:
            result = utils.load_results(tmp_path)
            self.assertEqual(result, {})
        finally:
            os.remove(tmp_path)

    def test_load_results_valid(self):
        data = {"key": "value"}
        with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        try:
            result = utils.load_results(tmp_path)
            self.assertEqual(result, data)
        finally:
            os.remove(tmp_path)

    def test_calculate_metrics(self):
        test_scores = np.array([10, 20, 30, 40])
        predictions = np.array([12, 18, 35, 38])
        alpha = 0.1
        metrics = utils.calculate_metrics(predictions, test_scores, alpha)
        expected_quantile_loss = utils.quantile_loss(test_scores, predictions, 1 - alpha)
        expected_abs_loss = utils.absolute_loss(test_scores, predictions)
        expected_sq_loss = utils.square_loss(test_scores, predictions)
        expected_set_size = np.mean(predictions)
        expected_coverage = np.mean(predictions >= test_scores)
        pos_excess = predictions[predictions >= test_scores] - test_scores[predictions >= test_scores]
        neg_excess = predictions[predictions < test_scores] - test_scores[predictions < test_scores]
        expected_pos_excess = np.mean(pos_excess) if pos_excess.size > 0 else 0
        expected_neg_excess = np.mean(neg_excess) if neg_excess.size > 0 else 0

        self.assertAlmostEqual(metrics["quantile_loss"], expected_quantile_loss)
        self.assertAlmostEqual(metrics["absolute_loss"], expected_abs_loss)
        self.assertAlmostEqual(metrics["square_loss"], expected_sq_loss)
        self.assertAlmostEqual(metrics["set_size"], expected_set_size)
        self.assertAlmostEqual(metrics["coverage"], expected_coverage)
        self.assertAlmostEqual(metrics["pos_excess"], expected_pos_excess)
        self.assertAlmostEqual(metrics["neg_excess"], expected_neg_excess)


if __name__ == "__main__":
    unittest.main()
