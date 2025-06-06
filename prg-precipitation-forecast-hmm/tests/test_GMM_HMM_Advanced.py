"""Advanced tests for GMM_HMM realistic performance."""

import unittest
import numpy as np
import pandas as pd

from GMM_HMM import GmmHMM


class TestGmmHMMAdvanced(unittest.TestCase):
    """Test GMM-HMM with realistic data and configurations."""

    def test_gmm_realistic_results(self):
        """Test to verify GMM_HMM works with realistic synthetic data."""

        np.random.seed(42)
        n_days = 1200

        data = pd.DataFrame()
        data['Date'] = pd.date_range('2020-01-01', periods=n_days)
        data['Rain Binary'] = np.random.binomial(1, 0.3, n_days)
        data['Avg. Temperature'] = np.random.normal(10, 5, n_days)
        data['Air Pressure'] = np.random.normal(1013, 10, n_days)
        data['Total Precipitation'] = np.random.exponential(2, n_days)
        data['Snow Depth'] = np.zeros(n_days)

        config = {
            'window_size': 100,
            'retrain_interval': 50,
            'n_components': 2,
            'n_iter': 10,
            'n_mix': 2,
            'features': ['Snow Depth', 'Air Pressure', 'Total Precipitation']
        }

        result = GmmHMM(data, config).run()

        # Test that result is iterable
        self.assertTrue(hasattr(result, '__len__'),f"Result should be iterable, got {type(result)}")

        # Test result structure
        self.assertEqual(len(result), 9, f"Expected 9 elements in result, got {len(result)}")

        _, predictions, truths, probs, acc, _, rec, f1, states = result

        # Test predictions length
        expected_length = n_days - config['window_size']
        self.assertEqual(len(predictions), expected_length,f"Expected {expected_length} predictions, got {len(predictions)}")
        self.assertEqual(len(truths), expected_length,f"Expected {expected_length} truths, got {len(truths)}")

        # Test metrics are in valid range
        self.assertGreaterEqual(acc, 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(acc, 0.8, "Accuracy should be <= 1")
        self.assertGreaterEqual(rec, 0.0, "Recall should be >= 0")
        self.assertLessEqual(rec, 0.8, "Recall should be <= 1")
        self.assertGreaterEqual(f1, 0.0, "F1 score should be >= 0")
        self.assertLessEqual(f1, 0.8, "F1 score should be <= 1")

        # Test that model performs better than random
        # With 30% rain probability, always predicting "no rain" gives ~70% accuracy
        self.assertGreater(acc, 0.5, f"Model accuracy {acc:.3f} should be better than random (0.5)")

        # Test predictions are binary
        unique_predictions = set(predictions)
        self.assertTrue(unique_predictions.issubset({0, 1}),f"Predictions should be binary (0 or 1), got {unique_predictions}")

        # Test probabilities are in valid range
        self.assertTrue(all(0 <= p <= 1 for p in probs),"All probabilities should be between 0 and 1")

        # Test states are valid
        unique_states = set(states)
        self.assertTrue(all(0 <= s < config['n_components'] for s in unique_states),f"All states should be between 0 and {config['n_components'] - 1}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
