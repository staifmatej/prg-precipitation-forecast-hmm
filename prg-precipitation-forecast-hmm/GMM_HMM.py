""" Gaussian Mixture Hidden Markov Model for forecasting daily precipitation using continuous emissions. """

import warnings
import logging

import numpy as np
from hmmlearn.hmm import GMMHMM
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
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # Suppresses warning messages from hmmlearn
np.seterr(all='ignore')  # Suppresses NumPy warnings

class GmmHMM(BaseHMM):
    """Train and evaluate Gaussian Mixture HMM for rainfall prediction using sliding window approach."""

    # __init__ is handled by parent class BaseHMM

    def fix_covariance_matrix(self, model):
        """Fixes covariance matrices to maintain model stability if the matrix is not positive definite."""

        # Minimum value to ensure positive definiteness
        min_value = 1e-3

        if model.covariance_type == "diag":
            # For diag we must ensure all values are positive
            for i in range(model.n_components):
                for j in range(model.n_mix):
                    # Replace any NaN/Inf values
                    nan_mask = np.isnan(model.covars_[i, j])
                    inf_mask = np.isinf(model.covars_[i, j])
                    if np.any(nan_mask) or np.any(inf_mask):
                        model.covars_[i, j] = np.ones_like(model.covars_[i, j]) * min_value

                    # Ensure minimum values for numerical stability
                    model.covars_[i, j] = np.maximum(model.covars_[i, j], min_value)

        return model

    def compute_emission_probability(self, state, observation):
        """
        Compute emission probability for a specific state and observation with
        enhanced numerical stability.
        """
        emission_prob = 0.0
        eps = 1e-10  # For numerical stability

        # For each mixture component
        for c in range(self.params['n_mix']):
            # Get weight of this component
            weight = self.model.weights_[state][c]
            if weight < eps:
                continue  # Skip components with negligible weight

            # Get mean and covariance for this component
            mean = self.model.means_[state][c]
            cov = self.model.covars_[state][c]

            # Handling singular covariances
            cov = np.maximum(cov, eps)

            try:
                # For diagonal covariances, we create a diagonal matrix
                cov_matrix = np.diag(cov)

                # Safe PDF calculation with numerical instability handling
                component_prob = multivariate_normal.pdf(
                    observation,
                    mean=mean,
                    cov=cov_matrix,
                    allow_singular=True
                )
                emission_prob += weight * component_prob
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                # Fallback for error case - add small probability
                emission_prob += weight * eps

        return np.maximum(emission_prob, eps)  # Ensures it will never be zero

    def _has_invalid_parameters(self, model):
        """
        Check if model parameters contain NaN or Inf values.
        
        Args:
            model: The HMM model to check
            
        Returns:
            bool: True if any parameter contains NaN or Inf values
        """
        parameters_to_check = [
            model.startprob_,
            model.transmat_,
            model.means_,
            model.covars_,
            model.weights_
        ]

        for param in parameters_to_check:
            if np.isnan(param).any() or np.isinf(param).any():
                return True
        return False

    # The forward_algorithm() and forward_algorithm_log() methods are inherited from BaseHMM

    def initialize_model(self):
        """Initialize and train the model on the first window"""

        # Explicit setting of initial probabilities for numerical stability
        n_components = self.params['n_components']
        startprob = np.ones(n_components) / n_components

        # Explicit setting of transition matrix
        transmat = np.ones((n_components, n_components)) / n_components

        model = GMMHMM(
            n_components=n_components,
            n_iter=self.params['n_iter'],
            n_mix=self.params['n_mix'],
            covariance_type="diag",
            tol=self.params['tol'],
            random_state=42,
            init_params="mcw",
            params="stmcw",
            verbose=False,
            startprob_prior=1.0,
            transmat_prior=1.0
        )

        # Explicit setting of initial values
        model.startprob_ = startprob
        model.transmat_ = transmat

        # Add regularization - adapted for diag
        model.covars_prior = 0.05  # Increase regularization
        model.means_prior = np.zeros((n_components,
                                      self.params['n_mix'],
                                      len(self.params['features'])))

        # Fit model on scaled data with explicit regularization
        X_train = self.scaler.transform(self.state['feature_data'][:self.params['window_size']])

        # Add small noise for stability
        np.random.seed(42)
        X_train = X_train + np.random.normal(0, 1e-5, X_train.shape)

        # Limit extreme values
        X_train = np.clip(X_train, -10, 10)

        try:
            model.fit(X_train)
            # Fix covariance matrices to ensure stability
            model = self.fix_covariance_matrix(model)

            # Check for NaN/Inf
            if self._has_invalid_parameters(model):
                # Reset model with default values if it contains invalid values
                model.startprob_ = startprob
                model.transmat_ = transmat
                model.covars_ = np.ones((n_components,
                                         self.params['n_mix'],
                                         len(self.params['features']))) * 0.1
                model.weights_ = np.ones((n_components, self.params['n_mix'])) / self.params['n_mix']
        except (ValueError, np.linalg.LinAlgError, RuntimeError, AttributeError) as e:
            print(f"Warning: Model initialization failed: {e}")
            # Creation of backup model with default parameters
            model.startprob_ = startprob
            model.transmat_ = transmat
            model.covars_ = np.ones((n_components,
                                     self.params['n_mix'],
                                     len(self.params['features']))) * 0.1
            model.weights_ = np.ones((n_components, self.params['n_mix'])) / self.params['n_mix']

            # Initialize means using random sampling from data
            indices = np.random.choice(len(X_train), size=n_components * self.params['n_mix'], replace=False)
            model.means_ = X_train[indices].reshape(n_components, self.params['n_mix'], -1)

        return model

    # The get_rainy_state() method is inherited from BaseHMM

    def _postprocess_model(self, model):
        """Apply GMM-specific post-processing: fix covariance matrices"""
        model = self.fix_covariance_matrix(model)

        # Check for invalid parameters
        if self._has_invalid_parameters(model):
            n_comp = self.params['n_components']
            model.startprob_ = np.ones(n_comp) / n_comp
            model.transmat_ = np.ones((n_comp, n_comp)) / n_comp
            model.covars_ = np.ones((self.params['n_components'],
                                     self.params['n_mix'],
                                     len(self.params['features']))) * 0.1
            model.weights_ = np.ones((self.params['n_components'], self.params['n_mix'])) / self.params['n_mix']

            # Initialize means using random sampling from data
            indices = np.random.choice(len(self.scaler.transform(self.state['feature_data'][:self.params['window_size']])),
                                     size=self.params['n_components'] * self.params['n_mix'],
                                     replace=False)
            model.means_ = self.scaler.transform(self.state['feature_data'][:self.params['window_size']])[indices].reshape(
                self.params['n_components'], self.params['n_mix'], -1)

        return model

    def _apply_seasonal_adjustment(self, rain_probability, current_month):
        """Apply GMM-specific seasonal adjustment"""
        if 5 <= current_month <= 7 or current_month == 1:
            return min(1.0, rain_probability * 1.2)
        return rain_probability

    # The run() method is inherited from BaseHMM

    def _get_default_config(self):
        """Get default configuration for GMM HMM"""
        config = DEFAULT_HMM_CONFIG.copy()
        config.update({
            'n_mix': 2,
            'tol': 1e-6,
            'features': DEFAULT_CONTINUOUS_FEATURES
        })
        return config

    # _prepare_window_data uses the default implementation from BaseHMM

    def _create_retrain_model(self):
        """Create a new GMM HMM model instance"""
        n_components = self.params['n_components']
        startprob = np.ones(n_components) / n_components
        transmat = np.ones((n_components, n_components)) / n_components

        model = GMMHMM(
            n_components=n_components,
            n_iter=self.params['n_iter'],
            n_mix=self.params['n_mix'],
            covariance_type="diag",
            tol=self.params['tol'],
            random_state=42,
            init_params="mcw",
            params="stmcw",
            verbose=False,
            startprob_prior=1.0,
            transmat_prior=1.0
        )

        model.startprob_ = startprob
        model.transmat_ = transmat
        model.covars_prior = 0.05
        model.means_prior = np.zeros((n_components, self.params['n_mix'],len(self.params['features'])))
        return model


# Function for applying class GmmHMM
def gmmHMM_func(data, config=None):
    """
    Train and evaluate Gaussian Mixture HMM for rainfall prediction using sliding window approach.

    Args:
        data: DataFrame with weather data
        config: Dictionary with configuration parameters or None to use defaults

    Returns:
        tuple: Model, predictions, actual values, probabilities, metrics, and states
    """
    model_instance = GmmHMM(data, config)
    return model_instance.run()


def main(short_dataset=False):
    """ Run precipitation forecasting with either preset optimal parameters or user-defined Bayesian optimization settings. """

    data = data_preprocessing(short_dataset=short_dataset)
    choice = manual_choice(name="Gaussian Mixture Hidden Markov Model")

    def auto_mode(data):
        print(f"\n{S_BOLD}Starting GMM-HMM Model Backtesting...{E_BOLD}\n")
        features = DEFAULT_FEATURES
        model_params = {
            'window_size': 1132,
            'retrain_interval': 1,
            'n_components': 4,
            'n_iter': 131,
            'n_mix': 4,
            'tol': 1e-8
        }
        run_auto_mode([data, gmmHMM_func, "GMM-HMM (auto)", features, model_params, True])

    def manual_mode(data):
        """Manual parameter optimization mode"""
        manual_bayesianOptimization(n_mix=True, n_tol=True)
        params = get_manual_mode_input(data, "GMM-HMM")
        if params:
            config = {
                'features': DEFAULT_FEATURES,
            }
            run_optimization(data, gmmHMM_func, params, "GMM-HMM (manual)", config)

    execute_mode_choice(choice, data, auto_mode, manual_mode)


if __name__ == "__main__":
    main(short_dataset=False)
