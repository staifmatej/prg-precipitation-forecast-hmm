"""Tests for GMM_HMM.py implementation."""

import unittest
import warnings
from unittest.mock import Mock
import numpy as np
import pandas as pd

from GMM_HMM import GmmHMM, gmmHMM_func


class TestGmmHMMBasic(unittest.TestCase):
    """Test basic GMM-HMM functionality."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=200),
            'Rain Binary': np.random.randint(0, 2, 200),
            'Avg. Temperature': np.random.randn(200) * 10 + 15,
            'Air Pressure': np.random.randn(200) * 5 + 1013,
            'Total Precipitation': np.random.exponential(2, 200)
        })

    def test_initialization(self):
        """Test model initialization with default and custom config."""
        model = GmmHMM(self.test_data)
        self.assertEqual(model.params['window_size'], 180)
        self.assertEqual(model.params['n_components'], 3)
        self.assertIsNotNone(model.model)

        large_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=700),
            'Rain Binary': np.random.randint(0, 2, 700),
            'Avg. Temperature': np.random.randn(700) * 10 + 15,
            'Air Pressure': np.random.randn(700) * 5 + 1013,
            'Total Precipitation': np.random.exponential(2, 700)
        })
        model_large = GmmHMM(large_data)
        self.assertEqual(model_large.params['window_size'], 500)

        # Custom config s window_size 100
        config = {'window_size': 100, 'n_components': 4, 'features': ['Air Pressure', 'Total Precipitation']}
        model = GmmHMM(self.test_data, config)
        self.assertEqual(model.params['window_size'], 100)  # Nebude upraveno
        self.assertEqual(model.state['feature_data'].shape[1], 2)

    def test_small_dataset_adjustment(self):
        """Test automatic window size adjustment for small datasets."""
        small_data = self.test_data.iloc[:50]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GmmHMM(small_data, {'window_size': 100})
        self.assertEqual(model.params['window_size'], 45)

    def test_fix_covariance_matrix(self):
        """Test covariance matrix fixing."""
        model = GmmHMM(self.test_data, {'window_size': 100})

        # Mock model with problematic covariances
        mock_model = Mock()
        mock_model.covariance_type = "diag"
        mock_model.n_components = 2
        mock_model.n_mix = 2
        mock_model.covars_ = np.array([
            [[np.nan, 1e-10, 0.5], [0.5, 0.5, 0.5]],
            [[np.inf, -0.1, 0.5], [0.5, 0.0, 0.5]]
        ])

        fixed_model = model.fix_covariance_matrix(mock_model)

        # Check all values are valid and >= minimum
        self.assertFalse(np.isnan(fixed_model.covars_).any())
        self.assertFalse(np.isinf(fixed_model.covars_).any())
        self.assertTrue(np.all(fixed_model.covars_ >= 1e-3))

    def test_compute_emission_probability(self):
        """Test emission probability computation."""
        model = GmmHMM(self.test_data, {'window_size': 100, 'n_components': 2})
        observation = np.array([15.0, 1013.0, 2.0])  # Temperature, Pressure, Precipitation

        for state in range(2):
            prob = model.compute_emission_probability(state, observation)
            self.assertGreater(prob, 0)
            self.assertLessEqual(prob, 1.0)
            self.assertTrue(np.isfinite(prob))


class TestGmmHMMAlgorithms(unittest.TestCase):
    """Test forward algorithms."""

    def setUp(self):
        """Set up test model."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100),
            'Avg. Temperature': np.random.randn(100) * 10 + 15,
            'Air Pressure': np.random.randn(100) * 5 + 1013,
            'Total Precipitation': np.random.randn(100) * 2
        })
        self.model = GmmHMM(self.test_data, {'window_size': 50})

    def test_forward_algorithms(self):
        """Test forward and forward_log algorithms."""
        observations = np.random.randn(10, 3)  # 3 features

        # Regular forward
        alpha, likelihood = self.model.forward_algorithm(observations)
        self.assertEqual(alpha.shape, (10, 3))
        self.assertGreater(likelihood, 0)

        # Log forward
        log_alpha, log_likelihood = self.model.forward_algorithm_log(observations)
        self.assertEqual(log_alpha.shape, (10, 3))
        self.assertTrue(np.isfinite(log_likelihood))

        # Check consistency
        self.assertAlmostEqual(likelihood, np.exp(log_likelihood), places=5)

    def test_get_rainy_state(self):
        """Test rainy state identification."""
        rainy_state = self.model.get_rainy_state(self.model.model)
        self.assertIn(rainy_state, range(self.model.params['n_components']))

    def test_predict_rain_probability(self):
        """Test rain probability prediction."""
        window = self.model.scaler.transform(self.model.state['feature_data'][:self.model.params['window_size']])
        prob = self.model.predict_rain_probability_next_day(window)
        self.assertTrue(0 <= prob <= 1)


class TestGmmHMMBacktesting(unittest.TestCase):
    """Test backtesting functionality."""

    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=150),
            'Rain Binary': np.random.randint(0, 2, 150),
            'Avg. Temperature': np.random.randn(150) * 10 + 15,
            'Air Pressure': np.random.randn(150) * 5 + 1013,
            'Total Precipitation': np.random.exponential(2, 150)
        })

    def test_run_basic(self):
        """Test basic run functionality."""
        model = GmmHMM(self.test_data, {
            'window_size': 50,
            'retrain_interval': 25,
            'n_components': 2
        })

        results = model.run()

        # Check return structure
        self.assertEqual(len(results), 9)
        _, predictions, truths, _, acc, prec, rec, f1, _ = results

        # Check lengths
        expected_length = len(self.test_data) - 50
        self.assertEqual(len(predictions), expected_length)
        self.assertEqual(len(truths), expected_length)

        # Check metrics
        for metric in [acc, prec, rec, f1]:
            self.assertTrue(0 <= metric <= 1)

    def test_retrain_model(self):
        """Test model retraining."""
        model = GmmHMM(self.test_data, {'window_size': 50})
        new_model, new_rainy_state = model.retrain_model(100)

        self.assertIsNotNone(new_model)
        self.assertIn(new_rainy_state, range(model.params['n_components']))

    def test_gmmHMM_func(self):
        """Test wrapper function."""
        config = {'window_size': 50, 'n_components': 2}
        results = gmmHMM_func(self.test_data, config)

        self.assertEqual(len(results), 9)
        self.assertIsInstance(results[1], list)  # predictions
        self.assertIsInstance(results[4], float)  # accuracy


class TestGmmHMMEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_missing_features(self):
        """Test with missing feature columns."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100)
        })

        with self.assertRaises(KeyError):
            GmmHMM(data, {'features': ['NonExistent']})

    def test_partial_nan_features(self):
        """Test handling of partial NaN values."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100),
            'Avg. Temperature': np.concatenate([[np.nan] * 50, np.random.randn(50) * 10 + 15]),
            'Air Pressure': np.concatenate([[np.nan] * 30, np.random.randn(70) * 5 + 1013]),
            'Total Precipitation': np.random.exponential(2, 100)
        })

        # Should handle gracefully with fillna
        model = GmmHMM(data, {'window_size': 50})
        self.assertIsNotNone(model.model)

    def test_single_component(self):
        """Test with single component."""
        data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100),
            'Avg. Temperature': np.random.randn(100) * 10 + 15,
            'Air Pressure': np.random.randn(100) * 5 + 1013,
            'Total Precipitation': np.random.randn(100)
        })

        model = GmmHMM(data, {'window_size': 50, 'n_components': 1})
        results = model.run()
        self.assertGreater(len(results[1]), 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
