""" Discrete Hidden Markov Model for forecasting daily precipitation using categorical (discrete) emissions. """

import numpy as np
from hmmlearn.hmm import CategoricalHMM

from utils.utils import (manual_bayesianOptimization,
                  data_preprocessing,
                  manual_choice,
                  S_BOLD,
                  E_BOLD,
                  Optimization)

from utils.utils_hmm import BaseHMM, DEFAULT_HMM_CONFIG
from utils.utils_common import execute_mode_choice, run_auto_mode, validate_basic_params


class discreteHMM(BaseHMM):
    """Train and evaluate HMM for rainfall prediction using sliding window approach."""

    def __init__(self, data, config=None):
        """
        Initialize HMM model with data and configuration.

        Args:
            data: DataFrame with weather data
            config: Dictionary with model configuration parameters or None to use defaults
        """
        # Initialize state attribute to avoid pylint warning
        self.state = None

        # Override _initialize_data_structures before calling super().__init__
        self._original_initialize = BaseHMM._initialize_data_structures
        BaseHMM._initialize_data_structures = self._initialize_data_structures_override

        try:
            # Call parent class constructor
            super().__init__(data, config)
        finally:
            # Restore original method
            BaseHMM._initialize_data_structures = self._original_initialize

    def _initialize_data_structures_override(self, data):
        """Override data structure initialization for discrete HMM."""
        # Initialize state dictionary first
        self.state = self._create_empty_state()

        if data is not None:
            # Dataset size check
            if len(data) < self.params['window_size']:
                raise ValueError(
                    f"Dataset size ({len(data)}) is smaller than window_size ({self.params['window_size']})")

            self.state['rain_data'] = data["Rain Binary"].values.reshape(-1, 1)
            self.state['observation_data'] = data["Rain Binary"].values.reshape(-1, 1)  
        else:
            # For tests that don't need data
            self.state['rain_data'] = None
            self.state['observation_data'] = None
    def _get_default_config(self):
        """Get default configuration for discrete HMM."""
        return DEFAULT_HMM_CONFIG.copy()

    # The forward_algorithm() and forward_algorithm_log() methods are inherited from BaseHMM
    # DiscreteHMM uses them with its compute_emission_probability implementation

    def initialize_model(self):
        """Initialize and train the model on the first window"""
        if self.state['rain_data'] is None:
            # For tests without data
            return None

        model = CategoricalHMM(
            n_components=self.params['n_components'],
            n_iter=self.params['n_iter'],
            random_state=42,
            n_features=2,  # binary output
            init_params="ste"
        )
        model.fit(self.state['rain_data'][:self.params['window_size']])
        return model

    def get_rainy_state(self, model):
        """Determine the rainy state from emission probabilities"""
        if model is None:
            return 0
        return np.argmax(model.emissionprob_[:, 1])

    def retrain_model(self, current_index):
        """Retrain the model on the most recent window"""
        model = CategoricalHMM(
            n_components=self.params['n_components'],
            n_iter=self.params['n_iter'],
            random_state=42,
            n_features=2,
            init_params="ste"
        )
        window_data = self.state['rain_data'][current_index - self.params['window_size']: current_index]
        model.fit(window_data)
        return model, self.get_rainy_state(model)

    def run(self):
        """Execute training and backtesting process"""
        self._reset_run_state()

        # Initialize model and rainy_state
        model = self.model

        # Helper method for processing a single prediction step
        def process_step(i, model_state):
            # Determine the hidden states on the last window
            window = self.state['rain_data'][i - self.params['window_size']: i]
            h_states = model_state.predict(window)

            # Retrieve the last state (for visualization and debugging)
            last = h_states[-1]
            self.state['states'].append(last)

            # Save actual/real value
            self.state['truths'].append(self.state['rain_data'][i][0])

            # Using marginalization for prediction
            rain_prob = self.predict_rain_probability_next_day(window)

            # Saving probability for threshold optimization
            self.state['probs'].append(rain_prob)

            # For now we'll use threshold 0.5 - will be optimized later
            self.state['predictions'].append(1 if rain_prob >= 0.5 else 0)

        for i in range(self.params['window_size'], len(self.state['rain_data']), self.params['step']):
            # Every retrain_interval days, retrain the model on the most recent window
            if (i - self.params['window_size']) % self.params['retrain_interval'] == 0 and i != self.params['window_size']:
                self.model, self.state['rainy_state'] = self.retrain_model(i)

            # Process actual situation - we pass only model, not rainy_state
            try:
                process_step(i, self.model)
            except (ValueError, IndexError, KeyError, AttributeError) as e:
                print(f"Warning: Failed to process step at index {i}: {e}")

        # Calculate metrics
        metrics = self.calculate_metrics(self.state['truths'], self.state['predictions'])

        return self._create_run_result(model, metrics)

    def predict_rain_probability_next_day(self, observations):
        log_alpha, _ = self.forward_algorithm_log(observations)

        max_val = np.max(log_alpha[-1, :])
        last_log_distribution = log_alpha[-1, :] - (max_val + np.log(np.sum(np.exp(log_alpha[-1, :] - max_val))))

        last_state_distribution = np.exp(last_log_distribution)

        rain_probability = 0
        for j in range(self.model.n_components):
            emission_prob_rain = self.model.emissionprob_[j, 1]
            for i in range(self.model.n_components):
                transition_prob = self.model.transmat_[i, j]
                rain_probability += emission_prob_rain * transition_prob * last_state_distribution[i]

        return rain_probability


    def _prepare_window_data(self, window_start, window_end):
        """Prepare window data for discrete HMM"""
        return self.state['rain_data'][window_start:window_end]

    def _create_retrain_model(self):
        """Create a new discrete HMM model instance"""
        return CategoricalHMM(
            n_components=self.params['n_components'],
            n_iter=self.params['n_iter'],
            random_state=42,
            n_features=2,
            init_params="ste"
        )

    def compute_emission_probability(self, state, observation):
        """Compute emission probability for discrete observations"""
        obs_value = observation[0] if hasattr(observation, '__getitem__') else observation
        return self.model.emissionprob_[state, obs_value]

# Function for applying class discreteHMM
def discreteHMM_func(data, config=None):
    """
    Train and evaluate HMM for rainfall prediction using sliding window approach.

    Args:
        data: DataFrame with weather data
        config: Dictionary with configuration parameters (window_size, retrain_interval,
                step, n_components, n_iter) or None to use defaults

    Returns:
        tuple: Model, predictions, actual values, evaluation metrics, and states
    """
    model_instance = discreteHMM(data, config)
    return model_instance.run()


def main(short_dataset):
    """ Run precipitation forecasting with either preset optimal parameters or user-defined Bayesian optimization settings. """

    data = data_preprocessing(short_dataset=short_dataset)
    choice = manual_choice(name="Hidden Markov Model with categorical (discrete) emissions")

    def auto_mode(data):
        print(f"\n{S_BOLD}Starting Discrete HMM Model Backtesting...{E_BOLD}\n")
        # Discrete HMM doesn't use features, only empty list
        run_auto_mode([data, discreteHMM_func, "Discrete HMM (auto)", [], {}])

    def manual_mode(data):
        manual_bayesianOptimization(n_mix=False, n_tol=False)

        try:
            opt_by = input(f"Optimize by '{S_BOLD}f1{E_BOLD}' or '{S_BOLD}acc{E_BOLD}'? (recommended acc): ").strip().lower()
            ws_range = tuple(map(int, input(f"Window size range ({S_BOLD}min max{E_BOLD}): ").split()))
            ri_range = tuple(map(int, input(f"Retrain interval range ({S_BOLD}min max{E_BOLD}): ").split()))
            nc_range = tuple(map(int, input(f"Num components range ({S_BOLD}min max{E_BOLD}): ").split()))
            ni_range = tuple(map(int, input(f"Num iterations range ({S_BOLD}min max{E_BOLD}): ").split()))
            trials = int(input(f"Number of trials ({S_BOLD}Expected Decimal Value{E_BOLD}): "))
            print_res = input(f"Print every trial ({S_BOLD}y/n{E_BOLD}): ").strip().lower()

        except (ValueError, IndexError, NameError):
            print("Wrong input.")
            return

        try:
            if ws_range[1] >= len(data):
                print(f"{S_BOLD}ERROR: Maximum window size ({ws_range[1]}) must be smaller than the dataset size ({len(data)}).{E_BOLD}")
                return

            if all(validate_basic_params({
                'opt_by': opt_by,
                'ws_range': ws_range,
                'ri_range': ri_range,
                'nc_range': nc_range,
                'ni_range': ni_range,
                'trials': trials,
                'print_res': print_res
            })):
                print_res = print_res == "y"
                print(f"\n{S_BOLD}Starting Discrete HMM Model Backtesting...{E_BOLD}\n")

                config_ranges = {
                    'window_size_range': (ws_range[0], ws_range[1]),
                    'retrain_interval_range': (ri_range[0], ri_range[1]),
                    'n_components_range': (nc_range[0], nc_range[1]),
                    'n_iter_range': (ni_range[0], ni_range[1])
                }

                params = {
                    'title': "Discrete HMM (manual)",
                    'data': data,
                    'model_func': discreteHMM_func,
                    'optimization_by': opt_by,
                    'config_param_ranges': config_ranges,
                    'n_trials': trials,
                    'print_every_trial': print_res,
                    'threshold_optimization': True
                }

                optimizer = Optimization(params)
                optimizer.bayesian_optimization()
            else:
                print("Wrong input.")
                return

        except (ValueError, IndexError, NameError):
            print("Wrong input.")
            return

    execute_mode_choice(choice, data, auto_mode, manual_mode)

if __name__ == "__main__":
    main(short_dataset=False)
