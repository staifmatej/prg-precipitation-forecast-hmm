""" Variational Gaussian Hidden Markov Model for forecasting daily precipitation using continuous emissions. """

import warnings
import logging

import numpy as np
from hmmlearn.vhmm import VariationalGaussianHMM
from scipy.stats import multivariate_normal

# Import from utils module
from utils.utils import (manual_bayesianOptimization,
                   data_preprocessing,
                   manual_choice,
                   S_BOLD,
                   E_BOLD)

from utils.utils_hmm import BaseHMM, DEFAULT_HMM_CONFIG, DEFAULT_CONTINUOUS_FEATURES
from utils.utils_common import get_manual_mode_input, run_optimization, run_auto_mode, execute_mode_choice, DEFAULT_FEATURES


warnings.filterwarnings('ignore')
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
np.seterr(all='ignore')

class VGHMM(BaseHMM):
    """Train and evaluate Variational Gaussian HMM for rainfall prediction using sliding window approach."""

    # __init__ is handled by parent class BaseHMM

    def _has_invalid_parameters(self, model):
        """Check if model has invalid (NaN or Inf) parameters"""
        return (np.isnan(model.startprob_).any() or np.isinf(model.startprob_).any() or
                np.isnan(model.transmat_).any() or np.isinf(model.transmat_).any() or
                np.isnan(model.means_).any() or np.isinf(model.means_).any() or
                np.isnan(model.covars_).any() or np.isinf(model.covars_).any())

    def compute_emission_probability(self, state, observation):
        """
        Compute emission probability for a specific state and observation with enhanced numerical stability.
        """
        emission_prob = 0.0
        eps = 1e-10  # For numerical stability

        # Get mean and covariance for this state
        mean = self.model.means_[state]

        # Handle covariance based on covariance_type
        if self.model.covariance_type == 'full':
            cov = self.model.covars_[state]
        elif self.model.covariance_type == 'diag':
            cov = np.diag(self.model.covars_[state])
        elif self.model.covariance_type == 'tied':
            cov = self.model.covars_
        elif self.model.covariance_type == 'spherical':
            cov = np.eye(len(mean)) * self.model.covars_[state]

        # Handling singular covariances
        if self.model.covariance_type in ['diag', 'spherical']:
            cov = np.maximum(cov, eps)

        try:
            # Use safe probability computation
            emission_prob = multivariate_normal.pdf(
                observation,
                mean=mean,
                cov=cov,
                allow_singular=True  # Key: allow singular matrices
            )
        except (ValueError, TypeError, AttributeError):
            emission_prob = eps

        return np.maximum(emission_prob, eps)  # Ensures it will never be zero

    # The forward_algorithm() and forward_algorithm_log() methods are inherited from BaseHMM

    def initialize_model(self):
        """Initialize and train the model on the first window"""
        model = VariationalGaussianHMM(
            n_components=self.params['n_components'],
            n_iter=self.params['n_iter'],
            covariance_type=self.params['covariance_type'],
            tol=self.params['tol'],
            random_state=42,
            init_params="kmeans",
            verbose=False
        )

        # Fit model on scaled data
        X_train = self.scaler.transform(self.state['feature_data'][:self.params['window_size']])

        # Add small noise for stability
        np.random.seed(42)
        X_train = X_train + np.random.normal(0, 1e-4, X_train.shape)

        # Limit extreme values
        X_train = np.clip(X_train, -10, 10)

        model.fit(X_train)

        return model

    # The get_rainy_state() method is inherited from BaseHMM

    def _postprocess_model(self, model):
        """Apply Variational Gaussian HMM-specific post-processing"""
        # Check for invalid parameters
        if self._has_invalid_parameters(model):
            raise ValueError("Model contains NaN or Inf parameters")
        return model

    # predict_rain_probability_next_day is inherited from BaseHMM
    # Variational Gaussian HMM doesn't use seasonal adjustment

    # The run() method is inherited from BaseHMM

    def _get_default_config(self):
        """Get default configuration for Variational Gaussian HMM"""
        config = DEFAULT_HMM_CONFIG.copy()
        config.update({
            'covariance_type': 'full',
            'tol': 1e-6,
            'features': DEFAULT_CONTINUOUS_FEATURES
        })
        return config

    # _prepare_window_data uses the default implementation from BaseHMM

    def _create_retrain_model(self):
        """Create a new Variational Gaussian HMM model instance"""
        return VariationalGaussianHMM(
            n_components=self.params['n_components'],
            n_iter=self.params['n_iter'],
            covariance_type=self.params['covariance_type'],
            tol=self.params['tol'],
            random_state=42,
            init_params="kmeans",
            verbose=False
        )


# Function for applying class VGHMM
def VGHMM_func(data, config=None):
    """
    Train and evaluate Variational Gaussian HMM for rainfall prediction using sliding window approach.

    Args:
        data: DataFrame with weather data
        config: Dictionary with configuration parameters or None to use defaults

    Returns:
        tuple: Model, predictions, actual values, probabilities, metrics, and states
    """
    model_instance = VGHMM(data, config)
    return model_instance.run()


def main(short_dataset=False):
    """Run precipitation forecasting with either preset optimal parameters or user-defined Bayesian optimization settings."""

    data = data_preprocessing(short_dataset=short_dataset)
    choice = manual_choice(name="Variational Gaussian Hidden Markov Model")

    def auto_mode(data):
        print(f"\n{S_BOLD}Starting Variational Gaussian HMM Model Backtesting...{E_BOLD}\n")
        features = DEFAULT_FEATURES
        model_params = {
            'window_size': 720,
            'retrain_interval': 1,
            'n_components': 4,
            'n_iter': 150,
            'covariance_type': 'full',
            'tol': 1e-8
        }
        run_auto_mode([data, VGHMM_func, "Variational Gaussian HMM (auto)", features, model_params])

    def manual_mode(data):
        """Manual parameter optimization mode"""
        manual_bayesianOptimization(n_tol=True)
        params = get_manual_mode_input(data, "Variational Gaussian HMM")
        if params:
            config = {
                'covariance_type': 'full',
                'features': DEFAULT_FEATURES,
            }
            run_optimization(data, VGHMM_func, params, "Variational Gaussian HMM (manual)", config)

    execute_mode_choice(choice, data, auto_mode, manual_mode)


if __name__ == "__main__":
    main(short_dataset=False)
