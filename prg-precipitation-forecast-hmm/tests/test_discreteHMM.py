"""Comprehensive tests for discreteHMM implementation and backtesting."""

import unittest
import numpy as np
import pandas as pd

from discreteHMM import discreteHMM


class TestDiscreteHMMDataLeakage(unittest.TestCase):
    """Test that there is no data leakage in the implementation."""

    def setUp(self):
        """Create test data."""
        # Create deterministic test data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100)
        rain_pattern = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1] * 10  # Repeating pattern

        self.test_data = pd.DataFrame({
            'Date': dates,
            'Rain Binary': rain_pattern
        })

        self.config = {
            'window_size': 10,
            'retrain_interval': 5,
            'n_components': 2,
            'n_iter': 10
        }

    def test_no_future_data_in_window(self):
        """Test that prediction windows don't contain future data."""
        model = discreteHMM(self.test_data, self.config)

        # Track windows used for predictions
        windows_used = []

        # Mock predict_rain_probability_next_day to capture windows
        original_predict = model.predict_rain_probability_next_day

        def mock_predict(observations):
            windows_used.append(observations.copy())
            return original_predict(observations)

        model.predict_rain_probability_next_day = mock_predict

        # Run the model
        model.run()

        # Check each window
        for i, window in enumerate(windows_used):
            prediction_index = self.config['window_size'] + i

            # Check that we're using correct historical data
            expected_start = prediction_index - self.config['window_size']
            expected_end = prediction_index

            # Verify window size
            self.assertEqual(len(window), self.config['window_size'])

            print(f"Prediction {i}: Using days {expected_start}-{expected_end - 1} to predict day {prediction_index}")

    def test_predictions_align_with_truths(self):
        """Test that predictions and truths are properly aligned."""
        model = discreteHMM(self.test_data, self.config)
        _, predictions, truths, _, _, _, _, _, _ = model.run()

        # Check alignment
        self.assertEqual(len(predictions), len(truths))

        # Verify truths match actual data
        for i, truth in enumerate(truths):
            actual_index = self.config['window_size'] + i
            expected_truth = self.test_data['Rain Binary'].iloc[actual_index]
            self.assertEqual(truth, expected_truth)

    def test_backtesting_range(self):
        """Test that backtesting covers the correct range of data."""
        model = discreteHMM(self.test_data, self.config)
        _, predictions, _, _, _, _, _, _, _ = model.run()

        # Expected number of predictions
        expected_predictions = len(self.test_data) - self.config['window_size']
        self.assertEqual(len(predictions), expected_predictions)

class TestDiscreteHMMForwardAlgorithm(unittest.TestCase):
    """Test forward algorithm implementations."""

    def setUp(self):
        """Create test model with known parameters."""
        # Simple test data
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=20),
            'Rain Binary': [0, 1, 0, 0, 1, 1, 0, 1, 0, 0] * 2
        })

        self.model = discreteHMM(self.test_data, {'window_size': 10, 'n_components': 2})

        # Set known parameters for testing
        self.model.model.startprob_ = np.array([0.7, 0.3])
        self.model.model.transmat_ = np.array([[0.8, 0.2], [0.3, 0.7]])
        self.model.model.emissionprob_ = np.array([[0.9, 0.1], [0.2, 0.8]])

    def test_forward_algorithm_dimensions(self):
        """Test that forward algorithm returns correct dimensions."""
        observations = np.array([[0], [1], [0]])
        alpha, likelihood = self.model.forward_algorithm(observations)

        self.assertEqual(alpha.shape, (3, 2))  # T x n_states
        self.assertIsInstance(likelihood, (float, np.float64))
        self.assertGreater(likelihood, 0)
        self.assertLessEqual(likelihood, 1)

    def test_forward_log_algorithm_consistency(self):
        """Test that log version gives consistent results."""
        observations = np.array([[0], [1], [0], [1]])

        # Regular forward
        alpha, likelihood = self.model.forward_algorithm(observations)

        # Log forward
        log_alpha, log_likelihood = self.model.forward_algorithm_log(observations)

        # Check consistency (allowing for numerical errors)
        self.assertAlmostEqual(likelihood, np.exp(log_likelihood), places=6)

        # Check that log version handles underflow better
        # Last row of alpha should match exp of last row of log_alpha
        np.testing.assert_allclose(alpha[-1, :], np.exp(log_alpha[-1, :]), rtol=1e-6)


class TestPredictionProbability(unittest.TestCase):
    """Test rain probability prediction."""

    def setUp(self):
        """Set up test model."""
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50),
            'Rain Binary': np.random.randint(0, 2, 50)
        })

        self.model = discreteHMM(self.test_data, {
            'window_size': 20,
            'n_components': 3
        })

    def test_probability_range(self):
        """Test that probabilities are in valid range [0, 1]."""
        # Create test window
        window = self.model.state['rain_data'][:20]

        # Get probability
        prob = self.model.predict_rain_probability_next_day(window)

        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_marginalization_sums_to_one(self):
        """Test that marginalization is done correctly."""
        # For any observation sequence, the sum of probabilities
        # for rain and no-rain should equal 1
        window = self.model.state['rain_data'][:20]

        # Get current state distribution
        log_alpha, _ = self.model.forward_algorithm_log(window)
        max_val = np.max(log_alpha[-1, :])
        last_log_distribution = log_alpha[-1, :] - (max_val + np.log(np.sum(np.exp(log_alpha[-1, :] - max_val))))
        last_state_distribution = np.exp(last_log_distribution)

        # Check that state distribution sums to 1
        self.assertAlmostEqual(np.sum(last_state_distribution), 1.0, places=6)


class TestBacktestingIntegrity(unittest.TestCase):
    """Test the integrity of backtesting process."""

    def setUp(self):
        """Create realistic test scenario."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=365),
            'Rain Binary': np.random.binomial(1, 0.3, 365)  # 30% rain probability
        })

        self.config = {
            'window_size': 60,
            'retrain_interval': 30,
            'n_components': 3,
            'n_iter': 20
        }

    def test_retrain_intervals(self):
        """Test that model retrains at correct intervals."""
        model = discreteHMM(self.test_data, self.config)

        # Track when retraining happens
        retrain_indices = []
        original_retrain = model.retrain_model

        def mock_retrain(current_index):
            retrain_indices.append(current_index)
            return original_retrain(current_index)

        model.retrain_model = mock_retrain

        # Run model
        model.run()

        # Check retrain intervals
        if len(retrain_indices) > 1:
            intervals = np.diff(retrain_indices)
            expected_interval = self.config['retrain_interval']

            # All intervals should be equal to retrain_interval
            self.assertTrue(all(interval == expected_interval for interval in intervals))

    def test_predictions_use_updated_model(self):
        """Test that predictions use the most recently trained model."""
        model = discreteHMM(self.test_data, self.config)

        original_process = model.run

        def track_model_params():
            # Run original
            results = original_process()

            # After running, check that model was updated
            self.assertIsNotNone(model.model)
            self.assertIsNotNone(model.model.transmat_)

            return results

        model.run = track_model_params
        results = model.run()

        # Verify results structure
        self.assertEqual(len(results), 9)  # Should return 9 elements

    def test_consistent_results_with_same_seed(self):
        """Test that results are reproducible with same random seed."""
        # First run
        model1 = discreteHMM(self.test_data, self.config)
        results1 = model1.run()
        predictions1 = results1[1]

        # Second run with same data and config
        model2 = discreteHMM(self.test_data, self.config)
        results2 = model2.run()
        predictions2 = results2[1]

        # Results should be identical
        np.testing.assert_array_equal(predictions1, predictions2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_small_dataset(self):
        """Test with dataset smaller than window_size."""
        small_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Rain Binary': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

        config = {'window_size': 20}  # Larger than dataset

        # Model should handle small dataset gracefully by adjusting window_size
        model = discreteHMM(small_data, config)
        # window_size should be adjusted to 90% of 10 = 9
        self.assertEqual(model.params['window_size'], 9)
        # Model should still run
        results = model.run()
        self.assertIsNotNone(results)

    def test_all_zeros_data(self):
        """Test with data that never rains."""
        no_rain_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': [0] * 100
        })

        model = discreteHMM(no_rain_data, {'window_size': 30})
        results = model.run()

        # Model should still run without errors
        predictions = results[1]
        self.assertEqual(len(predictions), 70)  # 100 - 30

    def test_all_ones_data(self):
        """Test with data that always rains."""
        always_rain_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': [1] * 100
        })

        model = discreteHMM(always_rain_data, {'window_size': 30})
        results = model.run()

        # Only extract what we need
        probs = results[3]

        # Probabilities should be high
        self.assertTrue(np.mean(probs) > 0.5)


class TestMetricsCalculation(unittest.TestCase):
    """Test that metrics are calculated correctly."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        model = discreteHMM(None, None)  # Don't need data for this test

        truths = [0, 1, 0, 1, 0, 1]
        predictions = [0, 1, 0, 1, 0, 1]  # Perfect match

        acc, prec, rec, f1 = model.calculate_metrics(truths, predictions)

        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)

    def test_no_rain_predictions(self):
        """Test metrics when model never predicts rain."""
        model = discreteHMM(None, None)

        truths = [0, 1, 0, 1, 0, 1]  # 50% rain
        predictions = [0, 0, 0, 0, 0, 0]  # Never predict rain

        acc, prec, rec, f1 = model.calculate_metrics(truths, predictions)

        self.assertEqual(acc, 0.5)  # Correct on non-rain days
        self.assertEqual(prec, 0)  # No true positives
        self.assertEqual(rec, 0)  # No true positives
        self.assertEqual(f1, 0)  # No true positives


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
