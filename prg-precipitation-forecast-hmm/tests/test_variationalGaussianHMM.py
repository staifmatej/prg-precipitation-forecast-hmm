"""Tests for variationalGaussianHMM.py implementation."""

import unittest
import warnings
import numpy as np
import pandas as pd

from variationalGaussianHMM import VGHMM_func, VGHMM


class TestVariationalGHMM(unittest.TestCase):
    """Test VariationalGHMM class methods."""

    def setUp(self):
        """Set up test data."""
        warnings.filterwarnings('ignore')

        # Create simple test data
        self.n_days = 200
        self.data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=self.n_days),
            'Rain Binary': np.random.binomial(1, 0.3, self.n_days),
            'Avg. Temperature': np.random.normal(10, 5, self.n_days),
            'Air Pressure': np.random.normal(1013, 10, self.n_days),
            'Total Precipitation': np.random.exponential(2, self.n_days),
            'Snow Depth': np.zeros(self.n_days)
        })

        self.config = {
            'window_size': 50,
            'retrain_interval': 25,
            'n_components': 2,
            'n_iter': 5,
            'covariance_type': 'full',
            'tol': 1e-3,
            'features': ['Avg. Temperature', 'Air Pressure', 'Total Precipitation']
        }

    def test_init(self):
        """Test __init__ method initializes correctly."""
        model = VGHMM(self.data, self.config)
        self.assertEqual(model.params['window_size'], 50)
        self.assertEqual(model.params['n_components'], 2)
        self.assertIsNotNone(model.state['feature_data'])
        self.assertIsNotNone(model.state['rain_data'])
        self.assertIsNotNone(model.scaler)
        self.assertIsNotNone(model.model)

    def test_compute_emission_probability(self):
        """Test compute_emission_probability method."""
        model = VGHMM(self.data, self.config)

        # Create test observation
        observation = np.array([10.0, 1013.0, 2.0])
        prob = model.compute_emission_probability(0, observation)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_forward_algorithm(self):
        """Test forward_algorithm method."""
        model = VGHMM(self.data, self.config)

        # Get some observations
        observations = model.state['feature_data'][:10]

        # Run forward algorithm
        alpha, likelihood = model.forward_algorithm(observations)

        # Check output
        self.assertEqual(alpha.shape, (10, model.params['n_components']))
        self.assertIsInstance(likelihood, float)
        self.assertGreater(likelihood, 0)

    def test_forward_algorithm_log(self):
        """Test forward_algorithm_log method."""
        model = VGHMM(self.data, self.config)

        # Get some observations
        observations = model.state['feature_data'][:10]

        # Run log forward algorithm
        log_alpha, log_likelihood = model.forward_algorithm_log(observations)

        # Check output
        self.assertEqual(log_alpha.shape, (10, model.params['n_components']))
        self.assertIsInstance(log_likelihood, float)
        self.assertLess(log_likelihood, 0)  # Log likelihood should be negative

    def test_initialize_model(self):
        """Test initialize_model method."""
        model = VGHMM(self.data, self.config)

        # Re-initialize model
        new_model = model.initialize_model()

        # Check model has required attributes
        self.assertTrue(hasattr(new_model, 'transmat_'))
        self.assertTrue(hasattr(new_model, 'means_'))
        self.assertTrue(hasattr(new_model, 'covars_'))
        self.assertEqual(new_model.n_components, 2)

    def test_get_rainy_state(self):
        """Test get_rainy_state method."""
        model = VGHMM(self.data, self.config)

        # Get rainy state
        rainy_state = model.get_rainy_state(model.model)

        # Check it's a valid state
        self.assertIn(rainy_state, [0, 1])

    def test_retrain_model(self):
        """Test retrain_model method."""
        model = VGHMM(self.data, self.config)
        result = model.retrain_model(60)

        # Check that method returns tuple of (model, rainy_state)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

        new_model, rainy_state = result

        # Check new model is returned
        self.assertIsNotNone(new_model)
        self.assertTrue(hasattr(new_model, 'transmat_'))
        self.assertTrue(hasattr(new_model, 'means_'))

        # Check rainy state is valid
        self.assertIn(rainy_state, [0, 1])

    def test_predict_rain_probability_next_day(self):
        """Test predict_rain_probability_next_day method."""
        model = VGHMM(self.data, self.config)

        # Get observations for prediction
        observations = model.state['feature_data'][:model.params['window_size']]

        # Predict rain probability
        prob = model.predict_rain_probability_next_day(observations)

        # Check probability is valid
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_run(self):
        """Test run method executes successfully."""

        small_data = self.data[:100].copy()
        small_config = self.config.copy()
        small_config['window_size'] = 30
        small_config['retrain_interval'] = 20

        model = VGHMM(small_data, small_config)

        # Run the model
        result = model.run()

        # Check result structure
        self.assertEqual(len(result), 9)
        _, predictions, truths, probs, acc, _, _, _, states = result

        # Check outputs
        self.assertGreater(len(predictions), 0)
        self.assertEqual(len(predictions), len(truths))
        self.assertEqual(len(predictions), len(probs))
        self.assertEqual(len(predictions), len(states))

        # Check metrics
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


class TestVariationalGHMMFunc(unittest.TestCase):
    """Test VGHMM_func function."""

    def test_variational_ghmm_func(self):
        """Test VGHMM_func executes correctly."""

        n_days = 150
        rain_binary = np.random.binomial(1, 0.3, n_days)

        # Construct DataFrame column by column
        data = pd.DataFrame()
        data['Date'] = pd.date_range('2020-01-01', periods=n_days)
        data['Rain Binary'] = rain_binary
        data['Avg. Temperature'] = np.random.normal(10, 5, n_days)
        data['Air Pressure'] = np.random.normal(1013, 10, n_days)
        data['Total Precipitation'] = np.where(rain_binary == 1, np.random.exponential(2, n_days), 0.0)
        data['Snow Depth'] = np.zeros(n_days)

        config = {
            'window_size': 40,
            'retrain_interval': 20,
            'n_components': 2,
            'n_iter': 5,
            'features': ['Avg. Temperature', 'Air Pressure']
        }

        # Call function
        result = VGHMM_func(data, config)

        # Check result
        self.assertEqual(len(result), 9)
        self.assertIsInstance(result[1], list)  # predictions
        self.assertIsInstance(result[2], list)  # truths
        self.assertIsInstance(result[3], list)  # probabilities
        self.assertIsInstance(result[4], float)  # accuracy


if __name__ == '__main__':
    unittest.main()
