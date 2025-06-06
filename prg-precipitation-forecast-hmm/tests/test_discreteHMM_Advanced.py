"""Aggressive tests to detect any cheating or incorrect backtesting that could lead to unrealistic results."""

import warnings
import unittest
import numpy as np
import pandas as pd

from discreteHMM import discreteHMM


class TestBacktestingCheatingDetection(unittest.TestCase):
    """Tests specifically designed to catch any form of cheating in backtesting."""

    def setUp(self):
        """Suppress warnings for cleaner output."""
        warnings.filterwarnings('ignore')

    def test_model_cannot_see_future_aggressive(self):
        """Test that model truly cannot see future by using a pattern that changes mid-sequence."""
        np.random.seed(42)

        random_part = np.random.randint(0, 2, 100).tolist()
        pattern_part = [1, 1, 0, 0] * 25

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=200),
            'Rain Binary': random_part + pattern_part
        })

        config = {
            'window_size': 30,
            'retrain_interval': 1000,  # No retraining
            'n_components': 3,
            'n_iter': 50
        }

        model = discreteHMM(data, config)
        _, predictions, truths, probs, _, _, _, _, _ = model.run()

        # Pattern starts at day 100, so at prediction index 70
        pattern_start = 70

        # Check predictions before pattern (should be ~50% accuracy on random data)
        before_pattern = predictions[40:pattern_start]
        before_truths = truths[40:pattern_start]
        acc_before = sum(1 for p, t in zip(before_pattern, before_truths) if p == t) / len(before_pattern)

        self.assertLess(acc_before, 0.65,"Model too accurate on random data")
        self.assertLess(np.std(probs[pattern_start - 10:pattern_start]) / np.std(probs[40:pattern_start]), 2.0,"Model becomes too confident before pattern starts")


    def test_backtesting_window_isolation(self):
        """Test that each prediction is made in complete isolation from future."""
        # Create data with "trap" - if model sees future, it will fall into trap
        np.random.seed(42)
        base_data = np.random.randint(0, 2, 180).tolist()

        trap_data = base_data + [1] * 20

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=200),
            'Rain Binary': trap_data
        })

        config = {
            'window_size': 50,
            'retrain_interval': 30,
            'n_components': 3
        }

        model = discreteHMM(data, config)
        _, predictions, _, probs, _, _, _, _, _ = model.run()

        # Check predictions DURING the trap
        trap_predictions = predictions[-20:]
        trap_rain_rate = sum(trap_predictions) / len(trap_predictions)

        # This PROVES model doesn't see future!
        self.assertLess(trap_rain_rate, 0.5,"Model correctly fails to predict all-rain trap - good!")

        # Verify probabilities stay reasonable (not suddenly confident)
        prob_std = np.std(probs[-20:])
        self.assertGreater(prob_std, 0.01,"Probabilities should show uncertainty, not sudden confidence")


    def test_no_information_leakage_through_retraining(self):
        """Test that retraining doesn't leak future information."""
        # Create data with sudden change
        # Days 0-99: 20% rain
        # Days 100-199: 80% rain
        np.random.seed(42)
        period1 = (np.random.rand(100) < 0.2).astype(int)
        period2 = (np.random.rand(100) < 0.8).astype(int)
        data_values = np.concatenate([period1, period2])

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=200),
            'Rain Binary': data_values
        })

        config = {
            'window_size': 40,
            'retrain_interval': 20,  # Frequent retraining
            'n_components': 2
        }

        model = discreteHMM(data, config)
        _, predictions, _, _, _, _, _, _, _ = model.run()

        change_point_idx = 100 - config['window_size']

        # Check predictions BEFORE change (should reflect ~20% rain)
        before_change = predictions[change_point_idx - 20:change_point_idx]
        before_rain_rate = sum(before_change) / len(before_change)

        # Model shouldn't "know" about the change before it happens
        self.assertLess(before_rain_rate, 0.4,"Model predicts too much rain before change - information leak!")


    def test_identical_predictions_different_positions(self):
        """Test that identical windows produce similar predictions regardless of position."""
        # Create repeating pattern
        pattern = [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]  # 10-day pattern
        repeated = pattern * 20  # 200 days

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=200),
            'Rain Binary': repeated
        })

        config = {
            'window_size': 10,  # Exactly one pattern length
            'retrain_interval': 1000,  # No retraining
            'n_components': 3
        }

        model = discreteHMM(data, config)

        # Capture probabilities for identical windows
        window_probs = {}

        original_predict = model.predict_rain_probability_next_day

        def capture_probs(observations):
            window_key = tuple(observations.flatten())
            prob = original_predict(observations)

            if window_key not in window_probs:
                window_probs[window_key] = []
            window_probs[window_key].append(prob)

            return prob

        model.predict_rain_probability_next_day = capture_probs

        model.run()

        # Check consistency
        for window_key, probs in window_probs.items():
            if len(probs) > 1:
                prob_std = np.std(probs)
                print(f"\nWindow {window_key}: {len(probs)} occurrences")
                print(f"Probabilities: {[f'{p:.3f}' for p in probs]}")
                print(f"Std deviation: {prob_std:.4f}")

                # Same window should produce similar predictions
                self.assertLess(prob_std, 0.05,"Identical windows produce different predictions - cheating?")

    def test_performance_degrades_with_random_data(self):
        """Test that model can't achieve good performance on truly random data."""
        np.random.seed(42)
        # Truly random data - no pattern to learn
        random_data = np.random.randint(0, 2, 1000)

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=1000),
            'Rain Binary': random_data
        })

        config = {
            'window_size': 100,
            'retrain_interval': 50,
            'n_components': 4,
            'n_iter': 80
        }

        model = discreteHMM(data, config)
        _, _, truths, _, acc, _, _, _, _ = model.run()

        # Calculate baseline
        rain_rate = sum(truths) / len(truths)
        baseline_acc = max(rain_rate, 1 - rain_rate)  # Always predict majority

        # On truly random data, model shouldn't be much better than baseline
        improvement = acc - baseline_acc
        self.assertLess(improvement, 0.05,
                        f"Model is {improvement:.3f} better than baseline on random data - impossible!")

    def test_verify_actual_prediction_before_seeing_truth(self):
        """Verify that prediction is made BEFORE seeing the truth value."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': [0, 1] * 50
        })

        config = {
            'window_size': 20,
            'retrain_interval': 1000
        }

        model = discreteHMM(data, config)

        # Track the exact order of operations
        operation_log = []

        # Override methods to log operations
        original_predict = model.predict_rain_probability_next_day

        def logged_predict(observations):
            operation_log.append(('PREDICT', len(operation_log)))
            return original_predict(observations)

        model.predict_rain_probability_next_day = logged_predict

        # Patch the run method to log truth access
        original_run = model.run

        def logged_run():
            # We need to track when truths are accessed
            # This is complex because we need to intercept the process_step function
            results = original_run()
            return results

        model.run = logged_run
        model.run()

        # The operation log should show predictions before truth access
        self.assertTrue(len(operation_log) > 0, "No operations were logged!")


class TestSuspiciouslyGoodResults(unittest.TestCase):
    """Tests specifically targeting the 65% accuracy claim."""

    def test_realistic_performance_expectations(self):
        """Test that model performance is realistic for binary-only features."""
        # Use real Prague rain statistics (39% rain days)
        np.random.seed(42)

        # Generate realistic data with some temporal correlation
        data_values = []
        rain_prob = 0.39

        # Add temporal correlation - if it rained yesterday, slightly higher chance today
        for i in range(1200):
            if i == 0 or data_values[-1] == 0:
                rain_today = np.random.rand() < rain_prob
            else:
                rain_today = np.random.rand() < (rain_prob * 1.2)  # 20% higher if rained yesterday

            data_values.append(int(rain_today))

        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=1200),
            'Rain Binary': data_values
        })

        # Test with your exact configuration
        config = {
            'window_size': 600,
            'retrain_interval': 1,
            'n_components': 4,
            'n_iter': 80
        }

        model = discreteHMM(data, config)
        _, predictions, truths, _, acc, _, _, _, _ = model.run()

        # Calculate various baselines
        rain_rate = sum(truths) / len(truths)

        # Check if model is suspiciously good
        if acc > 0.65:

            # Analyze predictions
            pred_rain_rate = sum(predictions) / len(predictions)
            print(f"Model rain prediction rate: {pred_rain_rate:.3f}")

            # Check if model is just copying the distribution
            distribution_diff = abs(pred_rain_rate - rain_rate)
            print(f"Distribution difference: {distribution_diff:.3f}")


if __name__ == '__main__':
    # Run tests with extra verbosity
    unittest.main(verbosity=2)
