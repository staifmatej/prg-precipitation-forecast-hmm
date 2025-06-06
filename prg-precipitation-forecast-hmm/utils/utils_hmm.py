"""Module for shared functionality across Hidden Markov Model implementations."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Default configuration values shared across all HMM models
DEFAULT_HMM_CONFIG = {
    'window_size': 500,
    'retrain_interval': 10,
    'step': 1,
    'n_components': 3,
    'n_iter': 20
}

# Default features for continuous models
DEFAULT_CONTINUOUS_FEATURES = ['Avg. Temperature', 'Air Pressure', 'Total Precipitation']

class BaseHMM(ABC):
    """
    Abstract base class for Hidden Markov Models used in precipitation forecasting.

    Provides common functionality that can be shared across different
    implementations of HMM models (Categorical, Supervised, Gaussian, etc.)
    """

    def __init__(self, data, config=None):
        """
        Initialize base HMM with common setup.

        Args:
            data: DataFrame with weather data
            config: Dictionary with configuration parameters
        """
        # Store configuration
        self.params = self._get_default_config()
        if config is not None:
            self.params.update(config)

        # Store reference to data
        self.params['data'] = data

        # Check and adjust window size if needed
        self._check_and_adjust_window_size(data)

        # Initialize data structures
        self._initialize_data_structures(data)

        # Initialize scaler if continuous features are used
        if hasattr(self, 'state') and 'feature_data' in self.state:
            self.scaler = StandardScaler()
            self.scaler.fit(self.state['feature_data'][:self.params['window_size']])

        # Initialize model and rainy state
        self.model = self.initialize_model()
        self.state['rainy_state'] = self.get_rainy_state(self.model)

    @abstractmethod
    def _get_default_config(self):
        """
        Get default configuration for the specific HMM implementation.

        Returns:
            dict: Default configuration parameters
        """

    def _create_empty_state(self):
        """Create empty state dictionary for tracking predictions"""
        return {
            'predictions': [],
            'truths': [],
            'states': [],
            'probs': []
        }

    def _initialize_data_structures(self, data):
        """
        Initialize common data structures for HMM models.

        Args:
            data: DataFrame with weather data
        """
        # Initialize state dictionary
        self.state = self._create_empty_state()

        # For continuous models, initialize feature data
        if 'features' in self.params:
            feature_df = data[self.params['features']].copy()

            # Handle missing values
            for col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())

            self.state['feature_data'] = feature_df.values

        # Always store rain data
        self.state['rain_data'] = data["Rain Binary"].values

    def _check_and_adjust_window_size(self, data):
        """Check if dataset is large enough for window_size and adjust if needed."""
        if data is not None and len(data) <= self.params['window_size']:
            print(f"Warning: Dataset too small ({len(data)} rows) for window_size {self.params['window_size']}")
            # Automatically reduce window_size to 90% of data length
            self.params['window_size'] = int(len(data) * 0.9)
            print(f"Adjusted window_size to {self.params['window_size']}")

    @abstractmethod
    def initialize_model(self):
        """
        Initialize and train the model on the first window.
        Must be implemented by subclasses.

        Returns:
            Trained model instance
        """

    def get_rainy_state(self, model):
        """
        Determine which hidden state corresponds to rainy days.

        Args:
            model: Trained HMM model

        Returns:
            int: Index of the state most associated with rain
        """
        # Get appropriate training data based on model type
        if hasattr(self, 'scaler') and 'feature_data' in self.state:
            # Continuous model - scale the training data
            X_train = self.scaler.transform(self.state['feature_data'][:self.params['window_size']])
        else:
            # Discrete model - use raw data
            X_train = self.state.get('observation_data', self.state.get('feature_data'))[:self.params['window_size']]

        # Predict hidden states on training data
        hidden_states = model.predict(X_train)

        # Calculate rain probability for each state
        state_rain_probs = {}
        for state in range(self.params['n_components']):
            mask = hidden_states == state
            if np.sum(mask) > 0:
                state_rain_probs[state] = np.mean(self.state['rain_data'][:self.params['window_size']][mask])
            else:
                state_rain_probs[state] = 0

        # Return the state with highest rain probability
        rainy_state = max(state_rain_probs, key=state_rain_probs.get)
        return rainy_state

    @abstractmethod
    def compute_emission_probability(self, state, observation):
        """
        Compute emission probability for a specific state and observation.
        This method must be implemented by subclasses based on their emission model.

        Args:
            state: Hidden state index
            observation: Single observation (format depends on model type)

        Returns:
            float: Emission probability P(observation|state)
        """

    def forward_algorithm(self, observations):
        """
        Implementation of the forward algorithm for HMM.
        Works with any emission model by using compute_emission_probability.

        Args:
            observations: Sequence of observations (array of shape [T, ...] for continuous data)

        Returns:
            forward_probs: Matrix of forward probabilities (shape [T, n_components])
            likelihood: Total probability of the observation sequence
        """
        # Extract model parameters
        n_states = self.model.n_components
        n_observations = len(observations)

        # Initialize forward probability matrix
        # alpha[t, i] = P(o_1, o_2, ..., o_t, q_t = i | model)
        alpha = np.zeros((n_observations, n_states))

        # Step 1: Initialization (t=0)
        for i in range(n_states):
            # Compute the emission probability using subclass implementation
            emission_prob = self.compute_emission_probability(i, observations[0])
            alpha[0, i] = self.model.startprob_[i] * emission_prob

        # Step 2: Recursion (for t=1,2,...,T-1)
        for t in range(1, n_observations):
            for j in range(n_states):
                # Sum over all possible previous states
                for i in range(n_states):
                    alpha[t, j] += alpha[t - 1, i] * self.model.transmat_[i, j]
                # Multiply by emission probability
                emission_prob = self.compute_emission_probability(j, observations[t])
                alpha[t, j] *= emission_prob

        # Step 3: Termination - compute total probability
        likelihood = np.sum(alpha[-1, :])

        return alpha, likelihood

    def forward_algorithm_log(self, observations):
        """
        Implementation of the forward algorithm using log probabilities for numerical stability.
        Works with any emission model by using compute_emission_probability.

        Args:
            observations: Sequence of observations (array of shape [T, ...] for continuous data)

        Returns:
            log_alpha: Matrix of log forward probabilities (shape [T, n_components])
            log_likelihood: Log of the total probability of the observation sequence
        """
        # Extract model parameters
        n_states = self.model.n_components
        n_observations = len(observations)

        # Initialize log forward probability matrix
        log_alpha = np.zeros((n_observations, n_states)) - np.inf  # log(0) = -inf

        # Small constant for log stability
        eps = 1e-10

        # Step 1: Initialization (t=0) with handling of zeros
        for i in range(n_states):
            emission_prob = self.compute_emission_probability(i, observations[0])
            log_alpha[0, i] = np.log(np.maximum(self.model.startprob_[i], eps)) + \
                              np.log(np.maximum(emission_prob, eps))

        # Step 2: Recursion (for t=1,2,...,T-1)
        for t in range(1, n_observations):
            for j in range(n_states):
                # Safe log-space addition
                log_sum = np.array([log_alpha[t - 1, i] +
                                    np.log(np.maximum(self.model.transmat_[i, j], eps))
                                    for i in range(n_states)])
                max_val = np.max(log_sum)

                if max_val > -np.inf:  # Avoid cases with -inf
                    log_alpha[t, j] = max_val + np.log(np.sum(np.exp(log_sum - max_val)))
                    emission_prob = self.compute_emission_probability(j, observations[t])
                    log_alpha[t, j] += np.log(np.maximum(emission_prob, eps))

        # Step 3: Termination - compute log likelihood using log-sum-exp trick
        max_val = np.max(log_alpha[-1, :])
        log_likelihood = max_val + np.log(np.sum(np.exp(log_alpha[-1, :] - max_val)))

        return log_alpha, log_likelihood

    def compute_sequence_likelihood(self, observations):
        """
        Compute the likelihood of an observation sequence given the model.

        Args:
            observations: Array of observations appropriate for the model type

        Returns:
            likelihood: Probability of the observation sequence given the model
        """
        _, likelihood = self.forward_algorithm(observations)
        return likelihood

    def compute_sequence_log_likelihood(self, observations):
        """
        Compute the log likelihood of an observation sequence given the model.

        Args:
            observations: Array of observations appropriate for the model type

        Returns:
            log_likelihood: Log probability of the observation sequence given the model
        """
        _, log_likelihood = self.forward_algorithm_log(observations)
        return log_likelihood

    def save_prob_for_threshold(self, model, current_state, rainy_state, prob_list):
        """Save transition probability to rainy state"""
        rainy_prob = model.transmat_[current_state, rainy_state]
        prob_list.append(rainy_prob)
        return prob_list

    def get_last_state_distribution(self, observations):
        """
        Calculate normalized state distribution from forward algorithm.
        
        Args:
            observations: Sequence of observations
            
        Returns:
            numpy.ndarray: Normalized probability distribution over states at last time step
        """
        log_alpha, _ = self.forward_algorithm_log(observations)
        max_val = np.max(log_alpha[-1, :])
        last_log_distribution = log_alpha[-1, :] - (max_val + np.log(np.sum(np.exp(log_alpha[-1, :] - max_val))))
        return np.exp(last_log_distribution)

    def marginalize_rain_probability(self, last_state_distribution, state_rain_probs):
        """
        Calculate rain probability by marginalizing over all possible next states.
        
        Args:
            last_state_distribution: Probability distribution over current states
            state_rain_probs: Dictionary mapping state indices to rain probabilities
            
        Returns:
            float: Marginalized rain probability for next time step
        """
        rain_probability = 0

        for j in range(self.model.n_components):
            # Probability of being in state j at time t+1
            next_state_prob = 0
            for i in range(self.model.n_components):
                next_state_prob += self.model.transmat_[i, j] * last_state_distribution[i]

            # Add contribution of state j to total rain probability
            rain_probability += next_state_prob * state_rain_probs.get(j, 0)

        return rain_probability

    def calculate_state_rain_probabilities(self, current_index, current_month=None):
        """
        Calculate rain probabilities for each hidden state.

        Args:
            current_index: Current position in the data
            current_month: Current month for seasonal filtering (optional)

        Returns:
            dict: Mapping from state index to rain probability
        """
        window_start = current_index - self.params['window_size']
        window_end = current_index

        # Get appropriate data based on model type
        if hasattr(self, 'scaler') and 'feature_data' in self.state:
            # Continuous model
            window_data = self.state['feature_data'][window_start:window_end]
            train_data = self.scaler.transform(window_data)
        else:
            # Discrete model
            train_data = self.state.get('observation_data', self.state.get('feature_data'))[window_start:window_end]

        # Predict states
        hidden_states = self.model.predict(train_data)
        window_rain_data = self.state['rain_data'][window_start:window_end]

        # Apply seasonal filtering if month is provided
        if current_month is not None and 'data' in self.params:
            seasonal_mask = self._create_seasonal_mask(window_start, window_end, current_month)
        else:
            seasonal_mask = np.ones(self.params['window_size'], dtype=bool)

        # Calculate probabilities
        state_rain_probs = {}
        for state in range(self.params['n_components']):
            # Try seasonal data first
            state_mask = (hidden_states == state) & seasonal_mask
            if np.sum(state_mask) > 0:
                state_rain_probs[state] = np.mean(window_rain_data[state_mask])
            else:
                # Fall back to all data for this state
                mask = hidden_states == state
                if np.sum(mask) > 0:
                    state_rain_probs[state] = np.mean(window_rain_data[mask])
                else:
                    state_rain_probs[state] = 0

        return state_rain_probs

    def _create_seasonal_mask(self, window_start, window_end, current_month):
        """
        Create mask for seasonal data filtering.

        Args:
            window_start: Start index of window
            window_end: End index of window
            current_month: Current month (1-12)

        Returns:
            numpy.ndarray: Boolean mask for seasonal filtering
        """
        seasonal_mask = np.zeros(self.params['window_size'], dtype=bool)

        for i, date in enumerate(self.params['data']['Date'].iloc[window_start:window_end]):
            month = date.month
            # Include similar months (handle December/January wrap-around)
            if abs(month - current_month) <= 1 or abs(month - current_month) >= 11:
                seasonal_mask[i] = True

        return seasonal_mask

    def calculate_metrics(self, truths, predictions):
        """Calculate evaluation metrics"""
        acc = accuracy_score(truths, predictions)
        precision = precision_score(truths, predictions, zero_division=0)
        recall = recall_score(truths, predictions, zero_division=0)
        f1 = f1_score(truths, predictions, zero_division=0)
        return acc, precision, recall, f1

    def _prepare_window_data(self, window_start, window_end):
        """
        Prepare window data for model operations.

        Args:
            window_start: Start index of the window
            window_end: End index of the window

        Returns:
            Prepared data suitable for the model (scaled for continuous, raw for discrete)
        """
        # Default implementation for continuous models
        if hasattr(self, 'scaler') and 'feature_data' in self.state:
            window_data = self.state['feature_data'][window_start:window_end]
            return self.scaler.transform(window_data)
        # For discrete models, this should be overridden
        return self.state.get('observation_data', [])[window_start:window_end]

    @abstractmethod
    def _create_retrain_model(self):
        """
        Create a new model instance for retraining.
        
        Returns:
            New model instance with proper configuration
        """

    def predict_rain_probability_next_day(self, observations):
        """
        Predict the probability of rain for the next day based on observations.
        
        Args:
            observations: Sequence of observations
            
        Returns:
            float: Probability of rain for the next day
        """
        # Get current index and month
        current_index = len(self.state['predictions']) + self.params['window_size']
        current_date = self.params['data']['Date'].iloc[current_index]
        current_month = current_date.month

        # Use BaseHMM methods
        last_state_distribution = self.get_last_state_distribution(observations)
        state_rain_probs = self.calculate_state_rain_probabilities(current_index, current_month)
        rain_probability = self.marginalize_rain_probability(last_state_distribution, state_rain_probs)

        # Apply seasonal adjustment if implemented by subclass
        return self._apply_seasonal_adjustment(rain_probability, current_month)

    def _apply_seasonal_adjustment(self, rain_probability, current_month):
        """
        Apply seasonal adjustment to rain probability. Override in subclass if needed.
        
        Args:
            rain_probability: Base rain probability
            current_month: Current month (1-12)
            
        Returns:
            float: Adjusted rain probability
        """
        # Default implementation - no adjustment
        # Subclasses can override to use current_month for seasonal adjustments
        _ = current_month  # Mark as intentionally unused in base implementation
        return rain_probability

    def _postprocess_model(self, model):
        """
        Perform model-specific post-processing after fitting.
        Override in subclasses if needed (e.g., fix_covariance_matrix for GMM).
        
        Args:
            model: The fitted model
            
        Returns:
            The post-processed model
        """
        return model

    def _reset_run_state(self):
        """Reset state arrays for a new run"""
        self.state['predictions'] = []
        self.state['truths'] = []
        self.state['states'] = []
        self.state['probs'] = []

    def run(self):
        """Execute training and backtesting process"""
        # Reset state
        self._reset_run_state()

        # Initialize model and rainy_state
        model = self.model

        # Helper method for processing a single prediction step
        def process_step(i, model_state):
            window_data = self._prepare_window_data(i - self.params['window_size'], i)

            # Determine the hidden states on the last window
            h_states = model_state.predict(window_data)

            # Save the last state for visualization
            last = h_states[-1]
            self.state['states'].append(last)

            # Save actual/real value for the next day
            self.state['truths'].append(self.state['rain_data'][i])

            # Predict rain probability for next day
            rain_prob = self.predict_rain_probability_next_day(window_data)

            # Save probability for threshold optimization
            self.state['probs'].append(rain_prob)

            # Use 0.5 threshold initially - will be optimized later
            self.state['predictions'].append(1 if rain_prob >= 0.5 else 0)

        # Backtest from window_size to the end
        data_length = len(self.state.get('feature_data', self.state.get('observation_data', [])))
        for i in range(self.params['window_size'], data_length, self.params['step']):

            # DEBUGGING
            #if i % 100 == 0:
                #print(f"Processing step {i}/{data_length}")

            # Every retrain_interval days, retrain the model on the most recent window
            if (i - self.params['window_size']) % self.params['retrain_interval'] == 0 and i != self.params['window_size']:
                try:
                    self.model, self.state['rainy_state'] = self.retrain_model(i)
                except (ValueError, AttributeError, np.linalg.LinAlgError):
                    # print(f"Warning: Failed to retrain model at index {i}: {e}")
                    # Continue with previous model
                    pass

            # Process step
            try:
                process_step(i, self.model)
            except (ValueError, IndexError, KeyError) as e:
                print(f"Warning: Failed to process step at index {i}: {e}")
                # Skip this step if it fails

        # Calculate metrics
        metrics = self.calculate_metrics(self.state['truths'], self.state['predictions'])

        return self._create_run_result(model, metrics)

    def _create_run_result(self, model, metrics):
        """Create the standard run result tuple"""
        return (model,
                self.state['predictions'],
                self.state['truths'],
                self.state['probs'],
                *metrics,  # unpacking metrics tuple
                self.state['states'])

    def retrain_model(self, current_index):
        """Retrain the model on the most recent window with additional stability measures"""
        model = self._create_retrain_model()

        window_data = self.state.get('feature_data', self.state.get('observation_data'))[current_index - self.params['window_size']: current_index]

        # Prepare data (scaling for continuous models)
        if hasattr(self, 'scaler'):
            X_train = self.scaler.transform(window_data)

            # Add small noise and limit extreme values for continuous models
            np.random.seed(current_index)  # Seed dependent on index for reproducibility
            X_train = X_train + np.random.normal(0, 1e-4, X_train.shape)
            X_train = np.clip(X_train, -10, 10)

            # Fit model with explicit NaN/Inf check
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                raise ValueError("Training data contains NaN or Inf values")
        else:
            # Discrete model - use raw data
            X_train = window_data

        model.fit(X_train)

        # Model-specific post-processing
        model = self._postprocess_model(model)

        if hasattr(model, 'means_'):
            # Check for invalid probabilities
            has_invalid_probs = (np.isnan(model.startprob_).any() or np.isinf(model.startprob_).any() or
                                np.isnan(model.transmat_).any() or np.isinf(model.transmat_).any())
            has_invalid_params = (np.isnan(model.means_).any() or np.isinf(model.means_).any() or
                                 np.isnan(model.covars_).any() or np.isinf(model.covars_).any())
            if has_invalid_probs or has_invalid_params:
                raise ValueError("Model contains NaN or Inf parameters")

        # Continue with rainy state calculation
        hidden_states = model.predict(X_train)
        window_rain = self.state['rain_data'][current_index - self.params['window_size']: current_index]

        state_rain_probs = {}
        for state in range(self.params['n_components']):
            mask = hidden_states == state
            if np.sum(mask) > 0:
                state_rain_probs[state] = np.mean(window_rain[mask])
            else:
                state_rain_probs[state] = 0

        rainy_state = max(state_rain_probs, key=state_rain_probs.get) if state_rain_probs else 0
        return model, rainy_state
