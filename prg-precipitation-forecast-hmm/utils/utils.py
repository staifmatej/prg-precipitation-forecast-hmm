"""Module for HMM models"""

import warnings
import os
import inspect

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.visualization import plot_optimization_history, plot_param_importances

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


# Global Constants for bold text.
S_BOLD = "\033[1m"
E_BOLD = "\033[0m"


class optimalizationThreshold:
    """Optimalization Threshold for HMM models"""

    @ staticmethod
    def bayesianOptimalization_onlyThreshold(optimization_by, truths, probabilities, n_trials=30):
        """Optimize threshold using Bayesian optimization."""

        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")

        def objective(trial):
            threshold = trial.suggest_float("threshold", 0.01, 0.99)
            preds = [1 if p >= threshold else 0 for p in probabilities]

            acc = accuracy_score(truths, preds)
            prec = precision_score(truths, preds, zero_division=0)
            rec = recall_score(truths, preds, zero_division=0)
            f1 = f1_score(truths, preds, zero_division=0)

            trial.set_user_attr("acc", acc)
            trial.set_user_attr("precision", prec)
            trial.set_user_attr("recall", rec)
            trial.set_user_attr("f1", f1)

            if optimization_by == "f1":
                return f1
            return acc

        # Limits parallel jobs to one less than the total CPU cores
        n_jobs = max(1, os.cpu_count() - 1)

        # Run the optimization
        with tqdm(total=n_trials, desc="Optimization Progress") as pbar:
            study.optimize(objective, n_trials=n_trials, callbacks=[lambda study, trial: pbar.update(1)], n_jobs=n_jobs)

        return study

    @ staticmethod
    def bayesianOptimalization_onlyThreshold_results(truths, probabilities, study):
        """print results for Optimize threshold using Bayesian optimization."""

        # Get the best threshold value
        best_threshold = study.best_trial.params["threshold"]

        # Print optimization results
        print("\n===== Results with Optimal Threshold =====")
        print(f"Optimal threshold: {S_BOLD}{best_threshold:.4f}{E_BOLD}")

        # Calculate metrics with the best threshold
        best_preds = [1 if p >= best_threshold else 0 for p in probabilities]

        print(f"Accuracy:       {S_BOLD}{accuracy_score(truths, best_preds):.4f}{E_BOLD}")
        print(f"Precision:      {S_BOLD}{precision_score(truths, best_preds, zero_division=1):.4f}{E_BOLD}")
        print(f"Recall:         {S_BOLD}{recall_score(truths, best_preds, zero_division=1):.4f}{E_BOLD}")
        print(f"F1 Score:       {S_BOLD}{f1_score(truths, best_preds, zero_division=1):.4f}{E_BOLD}")
        print("==========================================", end="\n")


def manual_choice(name):
    """Manual Choice"""

    print(f"{S_BOLD}Forecasting Daily Precipitation Using {name}.{E_BOLD}",end="\n\n")
    print("Would you like run the model directly with the best hyperparameters found through Bayesian optimization or set hyperparameters yourself?")
    choice = input("Type 'auto' to Run with best predefined parameters find by Bayesian Optimization. Bayesian optimization or 'manual' to set hyperparameters yourself:\n").strip().lower()
    return choice


def manual_bayesianOptimization(n_mix=False, n_tol=False):
    """Manual for bayesian optimization"""

    print(f"{S_BOLD}Manual Bayesian Optimization{E_BOLD}")
    print("================ Hyperparameter Settings Guide ================")
    print("1. Which metric to maximize during Bayesian optimization.")
    print("     Recommended: F1-score")
    print("2. The minimum and maximum length (in days) of the rolling window used to train each HMM.")
    print("     Recommended are values between 90 and 720.")
    print(f"!    {S_BOLD}WARNING{E_BOLD}: 360 days isn't always the best solution. Every model is different and needs to be fine‑tuned.")
    print("3. How often (in days) to refit the HMM as you step through the data.")
    print("     Recommended are values between 1 and 10.")
    print("     Effect:\n     a). A shorter interval → more frequent retraining\n     b). A longer interval → less compute overhead")
    print("4. The range for the number of hidden components.")
    print("     Recommended are values between 2 and 5.")
    print("     Effect:\n     a). More components → can capture more nuanced regimes\n     b). Fewer components → simpler rain/no-rain structure")
    print(f"!    {S_BOLD}WARNING{E_BOLD}: The more components you use, the more iterations you'll need to set.")
    print("5. The range for the maximum EM iterations per fit.")
    print("     Recommended are values between 5 and 75.")
    print("     Effect:\n     a). More iterations → better convergence (slower)\n     b). Fewer iterations → faster but may under‑fit")
    if n_tol:
        print("6. The tolerance for convergence.")
        print("     Recommended are values between 1e-6 and 1e-3.")
    if n_mix:
        print("7. The number of Gaussian mixtures per hidden state.")
        print("     Recommended are values between 1 and 5.")
        print("     Effect:\n     a). More mixtures → can model more complex emission distributions\n     b). Fewer mixtures → simpler model, less prone to overfitting")
        print(f"!    {S_BOLD}WARNING{E_BOLD}: More mixtures require more training data and iterations.")
    print("================================================================", end="\n\n")

def data_preprocessing(short_dataset):
    """
    Load weather datasets, combine them, and prepare binary rain indicators
    with advanced meteorological features for improved prediction.

    Returns:
        DataFrame: Processed weather data with rainfall indicators and advanced features
    """

    # Data retrieved from "https://meteostat.net/" for Prague-Ruzyne meteorological station
    data1 = pd.read_csv('data_set/meteostat-01.01.2000-31.12.2003.csv')
    data2 = pd.read_csv('data_set/meteostat-01.01.2004-31.12.2009.csv')
    data3 = pd.read_csv('data_set/meteostat-01.01.2010-31.12.2015.csv')
    data4 = pd.read_csv('data_set/meteostat-01.01.2016-31.12.2024.csv')

    weather_data = pd.concat([data1, data2, data3, data4], ignore_index=True)
    weather_data.to_csv("data_set/meteostat-01.01.2000-31.12.2024", index=False)

    weather_data.rename(columns={
        'date': 'Date',
        'tavg': 'Avg. Temperature',
        'tmin': 'Min. Temperature',
        'tmax': 'Max. Temperature',
        'prcp': 'Total Precipitation',
        'snow': 'Snow Depth',
        'wdir': 'Wind Direction',
        'wspd': 'Wind Speed',
        'wpgt': 'Peak Gust',
        'pres': 'Air Pressure',
        'tsun': 'Sunshine Duration'
    }, inplace=True)

    def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:

        conv = pd.to_numeric(df["Total Precipitation"], errors="coerce")
        df.loc[conv.isna(), "Total Precipitation"] = 0
        return df

    weather_data = clean_weather_data(weather_data)

    weather_data['Date'] = pd.to_datetime(weather_data['Date'])

    # Adding cyclic transformations for seasonality
    month = weather_data['Date'].dt.month
    weather_data['month_sin'] = np.sin(2 * np.pi * month / 12)
    weather_data['month_cos'] = np.cos(2 * np.pi * month / 12)

    day_of_year = weather_data['Date'].dt.dayofyear
    weather_data['day_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    weather_data['day_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    weather_data["Rain Binary"] = np.where(weather_data["Total Precipitation"].fillna(0) > 0, 1, 0)

    if short_dataset:
        total_rows = len(weather_data)
        start_index = int(total_rows * 0.80)
        weather_data = weather_data.iloc[start_index:].reset_index(drop=True)

    return weather_data


class HMMEvaluator:
    """Class for creating comprehensive visualizations for HMM model evaluation."""

    def __init__(self, params_dict):

        for key in ['model', 'predictions', 'truths']:
            if key not in params_dict:
                raise ValueError(f"Missing required parameter: {key}")

        self.model = params_dict['model']
        self.predictions = params_dict['predictions']
        self.truths = params_dict['truths']
        self.dates = params_dict.get('dates', None)
        self.states = params_dict.get('states', None)
        self.title = params_dict.get('title', "Evaluation")
        self.plot_objects = {}

    def _check_data(self):
        """Check if we have enough data to create visualizations."""
        if not self.predictions or not self.truths:
            print(f"Warning: Not enough data to create visualizations for {self.title}")
            return False
        return True

    def _setup_plots(self):
        """Setup matplotlib figures for visualization."""
        # Prepare figure for states and rain prediction
        if self.states is not None:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
            self.plot_objects['ax1'] = ax1
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(15, 5))

        self.plot_objects['fig'] = fig
        self.plot_objects['ax2'] = ax2

        # Prepare X values
        x_values = self.dates.to_numpy() if self.dates is not None else np.arange(len(self.truths))
        self.plot_objects['x_values'] = x_values

    def plot_confusion_matrix(self):
        """Plot confusion matrix of predictions vs truths."""
        cm = confusion_matrix(self.truths, self.predictions)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(f"{self.title} — Confusion Matrix")
        plt.show()

    def plot_states(self):
        """Plot the HMM states if available."""
        if self.states is None or len(self.states) == 0:
            return

        ax1 = self.plot_objects.get('ax1')
        x_values = self.plot_objects.get('x_values')
        if ax1 is None or x_values is None or len(x_values) == 0:
            return

        unique_states = sorted(set(self.states))
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
        state_labels = [f"State {s}" for s in unique_states]

        for i, state in enumerate(unique_states):
            mask = np.array(self.states) == state
            indices = np.where(mask)[0]
            if len(indices) > 0:
                # Filter indices that could be out of range
                valid_indices = indices[indices < len(x_values)]
                if len(valid_indices) > 0:
                    x_points = x_values[valid_indices]
                    y_points = np.full(len(valid_indices), state)
                    ax1.scatter(x_points, y_points, color=colors[i % len(colors)], label=state_labels[i], s=10, alpha=0.7)

        ax1.set_title(f"{self.title} — Predicted States")
        ax1.set_ylabel("State")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks(unique_states)

    def plot_rain_prediction(self):
        """Plot the actual vs predicted rain."""
        ax2 = self.plot_objects.get('ax2')
        fig = self.plot_objects.get('fig')
        x_values = self.plot_objects.get('x_values')
        if ax2 is None or fig is None or x_values is None or len(x_values) == 0:
            return

        actual_bin = np.array(self.truths, dtype=int)
        pred_bin = np.array(self.predictions, dtype=int)

        # Make sure we have the correct length of data
        min_len = min(len(x_values), len(actual_bin), len(pred_bin))
        x_values_trimmed = x_values[:min_len]
        actual_bin = actual_bin[:min_len]
        pred_bin = pred_bin[:min_len]

        ax2.plot(x_values_trimmed, actual_bin - 0.05, 'b-',
                 drawstyle='steps-post', label='Actual Rain', alpha=0.7)
        ax2.plot(x_values_trimmed, pred_bin + 0.05, 'r-',
                 drawstyle='steps-post', label='Predicted Rain', alpha=0.7)
        ax2.set_title(f"{self.title} — Actual vs Predicted Rain")
        ax2.set_ylabel("Rain (1 = Yes, 0 = No)")
        ax2.set_ylim(-0.2, 1.2)

        if self.dates is not None:
            fig.autofmt_xdate()

        ax2.set_xlabel("Date" if self.dates is not None else "Time Index")
        ax2.legend(loc='lower right')

        plt.tight_layout()
        plt.show()

    def plot_transition_matrix(self):
        """Plot the transition matrix of the HMM model."""
        if self.model is None or not hasattr(self.model, 'transmat_'):
            print("Model or transition matrix not available for visualization")
            return

        plt.figure(figsize=(8, 6))
        plt.imshow(self.model.transmat_, cmap='viridis', interpolation='none')
        plt.colorbar(label='Probability')
        plt.title(f"{self.title} — Transition Matrix")
        plt.xlabel("To State")
        plt.ylabel("From State")

        n_components = self.model.transmat_.shape[0]
        for i in range(n_components):
            for j in range(n_components):
                plt.text(j, i, f'{self.model.transmat_[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if self.model.transmat_[i, j] > 0.5 else 'black')

        plt.xticks(range(n_components))
        plt.yticks(range(n_components))
        plt.tight_layout()
        plt.show()

    def create_all_plots(self):
        """Create all visualizations for the HMM model evaluation."""
        if not self._check_data():
            return

        self.plot_confusion_matrix()
        self._setup_plots()

        if self.states is not None:
            self.plot_states()

        self.plot_rain_prediction()
        self.plot_transition_matrix()

class Optimization:
    """Class for performing Bayesian optimization of model hyperparameters."""

    def __init__(self, params_dict):
        """Initialize the optimization with parameters from a dictionary.

        Args:
            params_dict: Dictionary containing optimization parameters
        """
        # Core parameters
        self.config = {
            'title': params_dict.get('title', ''),
            'data': params_dict.get('data'),
            'model_func': params_dict.get('model_func'),
            'optimization_by': params_dict.get('optimization_by', 'f1'),
            'n_trials': params_dict.get('n_trials', 10),
            'print_every_trial': params_dict.get('print_every_trial', False),
            'threshold_optimization': params_dict.get('threshold_optimization', False)
        }


        if 'config_params' in params_dict:
            self.config['config_params'] = params_dict['config_params']

        if 'config_param_ranges' in params_dict:
            self.config['config_param_ranges'] = params_dict['config_param_ranges']

        # Range parameters
        self.ranges = {
            'window_size': params_dict.get('window_size_range', (1, 10)),
            'retrain_interval': params_dict.get('retrain_interval_range', (1, 10)),
            'n_components': params_dict.get('n_components_range', (2, 5)),
            'n_iter': params_dict.get('n_iter_range', (10, 100)),
            'tol': params_dict.get('tol_range'),
            'n_mix': params_dict.get('n_mix_range')
        }

        # Storage for results
        self.results = {
            'model_store': {},
            'pred_store': {},
            'actual_store': {},
            'states_store': {}
        }

        self.study = None

    def suggest_parameters(self, trial):
        """Suggest parameters for the trial."""

        if 'config_param_ranges' in self.config:
            config = {}
            ranges = self.config['config_param_ranges']

            for param_name, range_data in ranges.items():
                if param_name.endswith('_range'):
                    actual_param = param_name[:-6]
                else:
                    actual_param = param_name

                # Skip covariance_type as it's not a numeric range
                if actual_param == 'covariance_type':
                    continue

                min_val, max_val = range_data
                if param_name == 'tol_range' or actual_param == 'tol':
                    config[actual_param] = trial.suggest_float(actual_param, min_val, max_val, log=True)
                else:
                    config[actual_param] = trial.suggest_int(actual_param, min_val, max_val)

            config['step'] = 1

            if 'config_params' in self.config:
                if 'features' in self.config['config_params']:
                    config['features'] = self.config['config_params']['features']
                if 'covariance_type' in self.config['config_params']:
                    config['covariance_type'] = self.config['config_params']['covariance_type']

            return {'config': config}

        if 'config_params' in self.config:
            return {'config': self.config['config_params']}

        def suggest_range_params():
            """Process range parameters and suggest values using trial."""
            params = {}
            # Core parameters
            for param_name, param_range in self.ranges.items():
                if param_range is None:
                    continue

                if param_name == 'tol':
                    params[param_name] = trial.suggest_float(
                        param_name, param_range[0], param_range[1], log=True
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name, param_range[0], param_range[1]
                    )
            return params

        params = suggest_range_params()

        # Check if parameters are needed based on model signature
        sig = inspect.signature(self.config['model_func'])
        params = {k: v for k, v in params.items() if k in sig.parameters}

        # Add fixed step parameter
        params['step'] = 1

        # For consistency with new format, wrap parameters in 'config' key
        return {'config': params}

    def call_model_func(self, params):
        """Call the model function with the provided parameters."""
        return self.config['model_func'](
            data=self.config['data'],
            **params
        )

    def store_trial_results(self, trial, result):
        """Store trial results."""
        # Unpack results
        model, preds, actual, _, acc, prec, rec, f1, states = result

        # Save metrics to the trial
        trial.set_user_attr("acc", acc)
        trial.set_user_attr("precision", prec)
        trial.set_user_attr("recall", rec)
        trial.set_user_attr("f1", f1)

        # Store the model and data externally to avoid serialization issues
        trial_key = trial.number
        self.results['model_store'][trial_key] = model
        self.results['pred_store'][trial_key] = preds
        self.results['actual_store'][trial_key] = actual
        self.results['states_store'][trial_key] = states

        # Save just the reference key
        trial.set_user_attr("model_key", trial_key)

    def objective(self, trial):
        """Define the objective function for optimization."""
        params = self.suggest_parameters(trial)

        # Call the actual modeling function
        try:
            result = self.call_model_func(params)
            self.store_trial_results(trial, result)

            # What we optimized
            optimization_by = self.config['optimization_by']
            return result[7] if optimization_by == "f1" else result[4]  # f1 or acc

        except (ValueError, TypeError) as e:
            print(f"Trial failed with parameters {trial.params}: {e}")
            return -1.0  # Penalty for failed trials

    def setup_study(self):
        """Set up the optimization study."""
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(direction="maximize")

    def run_optimization(self):
        """Run the optimization process with progress bar."""
        # Limits parallel jobs to one less than the total CPU cores (minimum one)
        n_jobs = max(1, os.cpu_count() - 1)

        # Do optimization
        print(f"{S_BOLD}Optimizing hyperparameters...{E_BOLD}")
        with tqdm(total=self.config['n_trials'], desc="Optimization Progress") as pbar:
            self.study.optimize(
                self.objective,
                n_trials=self.config['n_trials'],
                callbacks=[lambda study, trial: pbar.update(1)],
                n_jobs=n_jobs
            )

    def print_trial_results(self):
        """Print results for each trial if requested."""
        if not self.config['print_every_trial']:
            return

        print("")
        for trial in self.study.trials:
            if trial.value is None or trial.value < 0:  # Skip failed trials
                continue

            print(f"Trial number:   {S_BOLD}{trial.number}{E_BOLD}")
            print(f" Params:        {trial.params}")
            print(f" F1-score:      {S_BOLD}{trial.value:.4f}{E_BOLD}")
            print(f" Accuracy:      {S_BOLD}{trial.user_attrs.get('acc', 0.0):.4f}{E_BOLD}")
            print(f" Precision:     {S_BOLD}{trial.user_attrs.get('precision', 0.0):.4f}{E_BOLD}")
            print(f" Recall:        {S_BOLD}{trial.user_attrs.get('recall', 0.0):.4f}{E_BOLD}")
            print("")

    def print_optimization_header(self):
        """Print optimization header based on trials count."""
        n_trials = self.config['n_trials']
        optimization_by = self.config['optimization_by']

        if n_trials == 1:
            print(f"\n{S_BOLD}===== Results with Best Parameters ====={E_BOLD}")
            print("Note: optimized for best F1 score.")
        else:
            print(f"\n{S_BOLD}====== Best trial optimized by {optimization_by.upper()} ======{E_BOLD}")

    def extract_trial_data(self, trial_key, best_trial):
        """Extract trial data from storage."""
        model = self.results['model_store'][trial_key]
        predictions = self.results['pred_store'][trial_key]
        truths = self.results['actual_store'][trial_key]
        states = self.results['states_store'][trial_key]

        # Getting window_size - can be in various formats depending on optimization type
        if 'config' in best_trial.params:
            # If we use new approach with configuration dictionary
            window_size = best_trial.params['config']['window_size']  # Loads from trial parameters
        else:
            # Backward compatibility with older approach
            window_size = best_trial.params.get("window_size", 500)

        # Determine appropriate dates
        data_source = states if states is not None and len(states) > 0 else predictions
        if not data_source:
            print("Warning: No valid data available for visualization.")
            return None

        dates = self.config['data']['Date'].iloc[window_size: window_size + len(data_source)]

        return model, predictions, truths, states, dates

    def visualize_results(self, result_data):
        """Visualize results based on available data.

        Args:
            result_data: Dictionary or tuple containing visualization data
        """
        try:
            # If input is tuple, convert to dictionary
            if isinstance(result_data, tuple):
                model, predictions, truths, dates, states = result_data
                params_dict = {
                    'model': model,
                    'predictions': predictions,
                    'truths': truths,
                    'dates': dates,
                    'states': states
                }
            else:
                # If it's already a dictionary, use it directly
                params_dict = result_data

            # Add title
            params_dict['title'] = self.config['title']

            # Create evaluator and generate graphs
            evaluator = HMMEvaluator(params_dict)
            evaluator.create_all_plots()

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Warning: Visualization failed due to invalid data: {e}")
        except (ImportError, RuntimeError) as e:
            print(f"Warning: Visualization failed due to plotting error: {e}")


    def process_best_trial_data(self, best_trial):
        """Process and visualize data from the best trial."""
        trial_key = best_trial.user_attrs.get('model_key')

        # Check if we have the model data
        if trial_key is None or trial_key not in self.results['model_store']:
            print("Warning: Could not retrieve model data for visualization.")
            return None

        # Extract stored data
        trial_data = self.extract_trial_data(trial_key, best_trial)
        if trial_data is None:
            return None

        model, predictions, truths, states, dates = trial_data

        # Visualize results
        self.visualize_results({
            'model': model,
            'predictions': predictions,
            'truths': truths,
            'dates': dates,
            'states': states
        })

        return truths, predictions, model

    def print_metrics(self, best_trial):
        """Print evaluation metrics."""
        optimization_by = self.config['optimization_by']

        if optimization_by == "f1":
            print(f" F1-score:      {S_BOLD}{best_trial.value:.4f}{E_BOLD}")
            print(f" Accuracy:      {S_BOLD}{best_trial.user_attrs['acc']:.4f}{E_BOLD}")
        elif optimization_by == "acc":
            print(f" Accuracy:      {S_BOLD}{best_trial.value:.4f}{E_BOLD}")
            print(f" F1-score:      {S_BOLD}{best_trial.user_attrs['f1']:.4f}{E_BOLD}")

        print(f" Precision:     {S_BOLD}{best_trial.user_attrs['precision']:.4f}{E_BOLD}")
        print(f" Recall:        {S_BOLD}{best_trial.user_attrs['recall']:.4f}{E_BOLD}")

    def handle_best_trial(self):
        """Process the results of the best trial."""
        best_trial = self.study.best_trial

        if not self.config['threshold_optimization']:
            # Print header
            self.print_optimization_header()

            # Print parameters - display parameters either from config or directly
            if 'config' in best_trial.params:
                print(f" Parameters:    {best_trial.params['config']}")
            else:
                print(f" Parameters:    {best_trial.params}")

        # Process trial data if valid
        if best_trial.value >= 0:  # Ensure it's not a failed trial
            result_data = self.process_best_trial_data(best_trial)

            if result_data:
                truths, predictions, _ = result_data
                if not self.config['threshold_optimization']:
                    self.print_metrics(best_trial)

                if not self.config['threshold_optimization']:
                    print("========================================", end="\n")
                return truths, predictions

        if not self.config['threshold_optimization']:
            print("========================================", end="\n")
        return None, None

    def optimize_threshold(self, truths, probabilities):
        """Perform threshold optimization if requested."""
        if not self.config['threshold_optimization']:
            return

        print(f"\n{S_BOLD}Optimizing threshold...{E_BOLD}")

        # Use the optimalizationThreshold class methods
        study = optimalizationThreshold.bayesianOptimalization_onlyThreshold(self.config['optimization_by'],truths,probabilities,n_trials=30)

        # Print the results
        optimalizationThreshold.bayesianOptimalization_onlyThreshold_results(truths,probabilities,study)

    def plot_optimization_results(self):
        """Plot the optimization history and parameter importances."""
        n_trials = self.config['n_trials']

        if n_trials < 2:
            return

        try:
            fig_history = plot_optimization_history(self.study)
            # Plotly way of setting title
            fig_history.update_layout(title="Optimization History")
            fig_history.show()
        except (ValueError, AttributeError, TypeError) as e:
            print("Could not plot optimization history. Possibly need more trials:", e)

        try:
            fig_importances = plot_param_importances(self.study)
            # Plotly way of setting title
            fig_importances.update_layout(title="Parameter Importances")
            fig_importances.show()
        except (ValueError, AttributeError, TypeError) as e:
            print("Param importances could not be computed. Possibly need more trials:", e)

    def bayesian_optimization(self):
        """Run the full Bayesian optimization process."""

        # Setup the study
        self.setup_study()

        # Run the optimization
        self.run_optimization()

        # Print results for each trial if requested
        self.print_trial_results()

        truths, predictions = self.handle_best_trial()

        # Perform threshold optimization if requested
        if truths is not None and predictions is not None and self.config['threshold_optimization']:
            self.optimize_threshold(truths, predictions)

        # Plot optimization results
        self.plot_optimization_results()

        return self.study.best_trial
