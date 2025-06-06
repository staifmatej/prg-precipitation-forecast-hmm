"""Tests for utils_hmm.py module."""

import unittest
from unittest.mock import Mock
import numpy as np

from utils.utils_hmm import BaseHMM


class ConcreteHMM(BaseHMM):
    """Concrete implementation for testing."""

    def __init__(self, data=None, config=None):
        """Initialize test HMM with proper state setup."""
        # For tests we'll set minimal state
        self.params = self._get_default_config()
        if config:
            self.params.update(config)

        self.state = {
            'predictions': [],
            'truths': [],
            'states': [],
            'probs': [],
            'rain_data': np.array([0, 1, 0, 1]) if data is None else data["Rain Binary"].values,
            'observation_data': np.array([[0], [1], [0], [1]]) if data is None else None,
            'feature_data': None
        }

        if data is not None:
            super().__init__(data, config)

    def _get_default_config(self):
        """Mock implementation."""
        return {'window_size': 10, 'n_components': 2}

    def _prepare_window_data(self, window_start, window_end):
        """Mock implementation."""
        return np.array([[1], [0]])

    def _create_retrain_model(self):
        """Mock implementation."""
        return Mock()

    def compute_emission_probability(self, state, observation):
        """Mock implementation."""
        return 0.5

    def initialize_model(self):
        """Mock implementation."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.n_components = 2
        return mock_model

    def predict_rain_probability_next_day(self, observations):
        """Mock implementation."""
        return 0.6


class TestBaseHMM(unittest.TestCase):
    """Test BaseHMM abstract class and methods."""

    def test_is_abstract_class(self):
        """Test that BaseHMM is properly defined as abstract class."""
        # We'll verify that it has abstract methods
        abstract_methods = BaseHMM.__abstractmethods__
        self.assertIn('_get_default_config', abstract_methods)
        self.assertIn('initialize_model', abstract_methods)
        self.assertIn('_create_retrain_model', abstract_methods)
        self.assertIn('compute_emission_probability', abstract_methods)

    def test_get_rainy_state(self):
        """Test get_rainy_state method."""
        hmm = ConcreteHMM(data=None)
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.n_components = 2

        hmm.state['rain_data'] = np.array([0, 1, 0, 1])
        hmm.params['window_size'] = 4
        hmm.params['n_components'] = 2

        rainy_state = hmm.get_rainy_state(mock_model)
        self.assertIn(rainy_state, [0, 1])

    def test_save_prob_for_threshold(self):
        """Test save_prob_for_threshold method."""
        hmm = ConcreteHMM(data=None)

        # Mock model with transition matrix
        model = Mock()
        model.transmat_ = np.array([[0.7, 0.3],[0.4, 0.6]])

        prob_list = []
        result = hmm.save_prob_for_threshold(model, 0, 1, prob_list)

        self.assertEqual(result, [0.3])
        self.assertEqual(len(result), 1)

    def test_calculate_metrics_perfect(self):
        """Test calculate_metrics with perfect predictions."""
        hmm = ConcreteHMM(data=None)
        truths = [1, 1, 0, 0, 1]
        predictions = [1, 1, 0, 0, 1]

        acc, prec, rec, f1 = hmm.calculate_metrics(truths, predictions)

        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)

    def test_calculate_metrics_imperfect(self):
        """Test calculate_metrics with imperfect predictions."""
        hmm = ConcreteHMM(data=None)
        truths = [1, 1, 0, 0, 1]
        predictions = [1, 0, 0, 1, 1]

        acc, prec, rec, f1 = hmm.calculate_metrics(truths, predictions)

        self.assertEqual(acc, 0.6)  # 3/5 correct
        self.assertAlmostEqual(prec, 0.667, places=3)  # 2/3
        self.assertAlmostEqual(rec, 0.667, places=3)  # 2/3
        self.assertAlmostEqual(f1, 0.667, places=3)

    def test_calculate_metrics_zero_division(self):
        """Test calculate_metrics handles zero division."""
        hmm = ConcreteHMM(data=None)
        truths = [0, 0, 0]
        predictions = [0, 0, 0]

        acc, prec, rec, f1 = hmm.calculate_metrics(truths, predictions)

        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 0)
        self.assertEqual(rec, 0)
        self.assertEqual(f1, 0)


class TestConcreteImplementation(unittest.TestCase):
    """Test concrete HMM implementation methods."""

    def test_default_config_used(self):
        """Test that default config is properly used in constructor."""
        hmm = ConcreteHMM(data=None)
        # Test that params contain expected values from _get_default_config
        self.assertIn('window_size', hmm.params)
        self.assertIn('n_components', hmm.params)
        self.assertEqual(hmm.params['window_size'], 10)
        self.assertEqual(hmm.params['n_components'], 2)

    def test_config_override(self):
        """Test that provided config overrides defaults."""
        custom_config = {'window_size': 20, 'n_components': 3}
        hmm = ConcreteHMM(data=None, config=custom_config)
        self.assertEqual(hmm.params['window_size'], 20)
        self.assertEqual(hmm.params['n_components'], 3)


if __name__ == '__main__':
    unittest.main()
