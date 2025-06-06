"""Common UI functions for HMM models to reduce code duplication."""

from utils.utils import Optimization

# Bold formatting
S_BOLD = '\033[1m'
E_BOLD = '\033[0m'

# Common feature sets for models
DEFAULT_FEATURES = [
    'Air Pressure',
    'Total Precipitation',
    'Snow Depth',
]

# Common validation criteria
VALIDATION_CRITERIA = [
    lambda opt_by: opt_by in ("f1", "acc"),
    lambda ws_range: ws_range[0] <= ws_range[1],
    lambda ri_range: ri_range[0] <= ri_range[1],
    lambda nc_range: nc_range[0] <= nc_range[1],
    lambda ni_range: ni_range[0] <= ni_range[1],
    lambda trials: trials >= 1,
    lambda print_res: print_res in ("y", "n")
]

def validate_basic_params(params):
    """Validate basic parameters using VALIDATION_CRITERIA"""

    return [
        VALIDATION_CRITERIA[0](params['opt_by']),
        VALIDATION_CRITERIA[1](params['ws_range']),
        VALIDATION_CRITERIA[2](params['ri_range']),
        VALIDATION_CRITERIA[3](params['nc_range']),
        VALIDATION_CRITERIA[4](params['ni_range']),
        VALIDATION_CRITERIA[5](params['trials']),
        VALIDATION_CRITERIA[6](params['print_res'])
    ]


def get_manual_mode_input(data, model_name):
    """Get user input for manual mode optimization parameters.

    Args:
        data: DataFrame with weather data
        model_name: Name of the model for display

    Returns:
        dict: Parameters for optimization or None if invalid input
    """

    try:
        opt_by = input(
            f"Optimize by '{S_BOLD}f1{E_BOLD}' or '{S_BOLD}acc{E_BOLD}'? (recommended f1): ").strip().lower()
        ws_range = tuple(map(int, input(f"Window size range ({S_BOLD}min max{E_BOLD}): ").split()))
        ri_range = tuple(map(int, input(f"Retrain interval range ({S_BOLD}min max{E_BOLD}): ").split()))
        nc_range = tuple(map(int, input(f"Num components range ({S_BOLD}min max{E_BOLD}): ").split()))
        ni_range = tuple(map(int, input(f"Num iterations range ({S_BOLD}min max{E_BOLD}): ").split()))

        extra_params = {}
        if model_name == "GMM-HMM":
            nm_range = tuple(map(int, input(f"Num mixtures range ({S_BOLD}min max{E_BOLD}): ").split()))
            tol_range = tuple(map(float, input(f"Tolerance range ({S_BOLD}min max{E_BOLD}): ").split()))
            extra_params['n_mix_range'] = nm_range
            extra_params['tol_range'] = tol_range
        elif model_name == "Variational Gaussian HMM":
            tol_range = tuple(map(float, input(f"Tolerance range ({S_BOLD}min max{E_BOLD}): ").split()))
            # Set default covariance type to 'diag' (fast and stable)
            default_cov_type = 'full'
            extra_params['tol_range'] = tol_range
            extra_params['covariance_type'] = default_cov_type

        trials = int(input(f"Number of trials ({S_BOLD}Expected Decimal Value{E_BOLD}): "))
        print_res = input(f"Print every trial ({S_BOLD}y/n{E_BOLD}): ").strip().lower()

    except (ValueError, IndexError, NameError):
        print("Wrong input.")
        return None

    try:
        if ws_range[1] >= len(data):
            print(
                f"{S_BOLD}ERROR: Maximum window size ({ws_range[1]}) must be smaller than the dataset size ({len(data)}).{E_BOLD}")
            return None

        validations = validate_basic_params({
            'opt_by': opt_by,
            'ws_range': ws_range,
            'ri_range': ri_range,
            'nc_range': nc_range,
            'ni_range': ni_range,
            'trials': trials,
            'print_res': print_res
        })

        if model_name == "GMM-HMM":
            validations.extend([
                extra_params['n_mix_range'][0] <= extra_params['n_mix_range'][1],
                extra_params['tol_range'][0] <= extra_params['tol_range'][1]
            ])
        elif model_name == "Variational Gaussian HMM":
            validations.append(extra_params['tol_range'][0] <= extra_params['tol_range'][1])

        if all(validations):
            print_res = print_res == "y"
            print(f"\n{S_BOLD}Starting {model_name} Model Backtesting...{E_BOLD}\n")

            # Configuration parameters ranges
            config_ranges = {
                'window_size_range': (ws_range[0], ws_range[1]),
                'retrain_interval_range': (ri_range[0], ri_range[1]),
                'n_components_range': (nc_range[0], nc_range[1]),
                'n_iter_range': (ni_range[0], ni_range[1]),
            }
            config_ranges.update(extra_params)

            return {
                'optimization_by': opt_by,
                'config_ranges': config_ranges,
                'n_trials': trials,
                'print_every_trial': print_res
            }

        print("Wrong input.")
        return None

    except (ValueError, IndexError, NameError):
        print("Wrong input.")
        return None


def run_optimization(data, model_func, params, title, config=None):
    """Run Bayesian optimization for a model.

    Args:
        data: DataFrame with weather data
        model_func: Model function to optimize
        params: Parameters from get_manual_mode_input
        title: Title for the optimization run
        config: Additional configuration parameters
    """

    if config is None:
        config = {}

    optimization_params = {
        'title': title,
        'data': data,
        'model_func': model_func,
        'optimization_by': params['optimization_by'],
        'config_params': config,
        'config_param_ranges': params['config_ranges'],
        'n_trials': params['n_trials'],
        'print_every_trial': params['print_every_trial'],
        'threshold_optimization': True
    }

    optimizer = Optimization(optimization_params)
    optimizer.bayesian_optimization()


def create_auto_mode_config(features_list, model_specific_params=None):
    """Create configuration for auto mode.
    
    Args:
        features_list: List of features to use
        model_specific_params: Dictionary of model-specific parameters
        
    Returns:
        dict: Configuration dictionary
    """

    config = {
        'window_size': 600,
        'retrain_interval': 1,
        'n_components': 4,
        'n_iter': 80,
        'features': features_list
    }

    if model_specific_params:
        config.update(model_specific_params)

    return config


def run_auto_mode(params_list):
    """Run auto mode optimization.
    
    Args:
        params_list: List containing parameters in order:
            [0] data: DataFrame with weather data  
            [1] model_func: Model function to optimize
            [2] title: Title for the optimization run
            [3] features_list: List of features to use
            [4] model_specific_params: Dictionary of model-specific parameters (optional)
            [5] threshold_optimization: Whether to optimize threshold (optional, default: True)
    """

    data = params_list[0]
    model_func = params_list[1]
    title = params_list[2]
    features_list = params_list[3]
    model_specific_params = params_list[4] if len(params_list) > 4 else None
    threshold_optimization = params_list[5] if len(params_list) > 5 else True
    config = create_auto_mode_config(features_list, model_specific_params)

    params = {
        'title': title,
        'data': data,
        'model_func': model_func,
        'config_params': config,
        'optimization_by': "acc",
        'n_trials': 1,
        'print_every_trial': False,
        'threshold_optimization': threshold_optimization
    }

    optimizer = Optimization(params)
    optimizer.bayesian_optimization()


def execute_mode_choice(choice, data, auto_mode_func, manual_mode_func):
    """Execute user's choice between auto and manual mode.
    
    Args:
        choice: User's choice ('auto' or 'manual')
        data: DataFrame with weather data
        auto_mode_func: Function to execute in auto mode
        manual_mode_func: Function to execute in manual mode
    """

    if choice == "auto":
        auto_mode_func(data)
    elif choice == 'manual':
        manual_mode_func(data)
    else:
        print("Wrong input.")
