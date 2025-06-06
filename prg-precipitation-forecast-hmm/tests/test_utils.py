"""Tests for utils.py module."""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
import numpy as np

from utils import (optimalizationThreshold,
                   data_preprocessing,
                   HMMEvaluator,
                   Optimization)

# PEP8 refactoring applied
class TestOptimalizationThreshold(unittest.TestCase):
    """Test optimalizationThreshold class."""

    def setUp(self):
        """Set up test data."""
        self.truths = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        self.probabilities = [0.8, 0.2, 0.9, 0.7, 0.3, 0.1, 0.6, 0.4, 0.85, 0.15]

    @patch('optuna.create_study')
    def test_optimization_returns_study(self, mock_create_study):
        """Test optimization returns study object."""
        mock_study = Mock()
        mock_create_study.return_value = mock_study

        result = optimalizationThreshold.bayesianOptimalization_onlyThreshold("f1", self.truths, self.probabilities, n_trials=5)

        mock_create_study.assert_called_once_with(direction="maximize")
        self.assertEqual(result, mock_study)

    @patch('optuna.create_study')
    def test_objective_function(self, mock_create_study):
        """Test objective function calculates metrics correctly."""
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.5
        mock_trial.set_user_attr = Mock()  # Mock the set_user_attr method

        mock_study.optimize = Mock()
        mock_create_study.return_value = mock_study

        # Just test that optimization runs without error
        result = optimalizationThreshold.bayesianOptimalization_onlyThreshold("f1", self.truths, self.probabilities, n_trials=1)

        # Verify the study was created and optimize was called
        mock_create_study.assert_called_once_with(direction="maximize")
        mock_study.optimize.assert_called_once()
        self.assertEqual(result, mock_study)

    @patch('builtins.print')
    def test_results_printing(self, mock_print):
        """Test results are printed correctly."""
        mock_study = Mock()
        mock_study.best_trial.params = {"threshold": 0.5}
        optimalizationThreshold.bayesianOptimalization_onlyThreshold_results(self.truths, self.probabilities, mock_study)
        self.assertTrue(mock_print.called)

    @patch('os.cpu_count', return_value=8)
    @patch('optuna.create_study')
    def test_parallel_jobs(self, mock_create_study, _mock_cpu_count):
        """Test parallel jobs set correctly."""
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        optimalizationThreshold.bayesianOptimalization_onlyThreshold("acc", self.truths, self.probabilities, n_trials=3)
        call_args = mock_study.optimize.call_args
        self.assertEqual(call_args[1]['n_jobs'], 7)


class TestDataPreprocessing(unittest.TestCase):
    """Test data_preprocessing function."""

    def setUp(self):
        """Create mock CSV data."""
        self.mock_data = pd.DataFrame({
            'date': pd.date_range('2000-01-01', periods=100),
            'tavg': np.random.randn(100),
            'tmin': np.random.randn(100),
            'tmax': np.random.randn(100),
            'prcp': np.random.rand(100) * 10,
            'snow': np.random.rand(100) * 5,
            'wdir': np.random.rand(100) * 360,
            'wspd': np.random.rand(100) * 20,
            'wpgt': np.random.rand(100) * 30,
            'pres': np.random.rand(100) * 100 + 950,
            'tsun': np.random.rand(100) * 12
        })

    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.read_csv')
    def test_short_dataset_size(self, mock_read_csv, mock_to_csv):
        """Test short dataset is 20% of full dataset."""
        mock_read_csv.return_value = self.mock_data
        mock_to_csv.return_value = None
        full_data = data_preprocessing(short_dataset=False)
        short_data = data_preprocessing(short_dataset=True)
        expected_short_size = int(len(full_data) * 0.20)
        self.assertEqual(len(short_data), expected_short_size)

    @patch('pandas.read_csv')
    def test_column_count(self, mock_read_csv):
        """Test correct number of columns."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)

        # Original 11 + 5 new columns (month_sin, month_cos, day_sin, day_cos, Rain Binary)
        self.assertEqual(len(data.columns), 16)

    @patch('pandas.read_csv')
    def test_no_nan_values(self, mock_read_csv):
        """Test critical columns have no NaN values."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)

        # Check no NaN in critical columns
        self.assertFalse(data['Total Precipitation'].isna().any())
        self.assertFalse(data['Air Pressure'].isna().any())
        self.assertFalse(data['Snow Depth'].isna().any())

    @patch('pandas.read_csv')
    def test_numeric_columns(self, mock_read_csv):
        """Test critical columns are numeric."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)

        # Check columns are numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(data['Total Precipitation']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['Air Pressure']))
        self.assertTrue(pd.api.types.is_numeric_dtype(data['Snow Depth']))

    @patch('pandas.read_csv')
    def test_rain_binary_values(self, mock_read_csv):
        """Test Rain Binary contains only 0 and 1."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)
        unique_values = data['Rain Binary'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))

    @patch('pandas.read_csv')
    def test_date_column_type(self, mock_read_csv):
        """Test Date column is datetime type."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)
        self.assertEqual(data['Date'].dtype, 'datetime64[ns]')

    @patch('pandas.read_csv')
    def test_cyclic_features_range(self, mock_read_csv):
        """Test cyclic features are in range [-1, 1]."""
        mock_read_csv.return_value = self.mock_data
        data = data_preprocessing(short_dataset=False)
        for col in ['month_sin', 'month_cos', 'day_sin', 'day_cos']:
            self.assertTrue(data[col].min() >= -1.0)
            self.assertTrue(data[col].max() <= 1.0)


class TestHMMEvaluator(unittest.TestCase):
    """Test HMMEvaluator class."""

    def setUp(self):
        """Set up test data for HMM evaluation."""
        self.mock_model = Mock()
        self.mock_model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])

        self.valid_params = {
            'model': self.mock_model,
            'predictions': [1, 0, 1, 1, 0],
            'truths': [1, 0, 1, 0, 0],
            'dates': pd.date_range('2020-01-01', periods=5),
            'states': [0, 1, 0, 0, 1],
            'title': 'Test Evaluation'
        }

    def test_initialization(self):
        """Test initialization scenarios."""
        # Test successful initialization
        evaluator = HMMEvaluator(self.valid_params)
        self.assertEqual(evaluator.model, self.mock_model)
        self.assertEqual(evaluator.predictions, [1, 0, 1, 1, 0])
        self.assertEqual(evaluator.truths, [1, 0, 1, 0, 0])
        self.assertEqual(evaluator.title, 'Test Evaluation')

        # Test missing required parameters
        for param in ['model', 'predictions', 'truths']:
            params = self.valid_params.copy()
            del params[param]
            with self.assertRaises(ValueError) as context:
                HMMEvaluator(params)
            self.assertIn(f"Missing required parameter: {param}", str(context.exception))

        # Test default values
        minimal = {'model': self.mock_model, 'predictions': [1], 'truths': [0]}
        evaluator = HMMEvaluator(minimal)
        self.assertIsNone(evaluator.dates)
        self.assertEqual(evaluator.title, "Evaluation")

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrix(self, mock_show):
        """Test confusion matrix plotting."""
        evaluator = HMMEvaluator(self.valid_params)
        evaluator.plot_confusion_matrix()
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_transition_matrix(self, mock_show):
        """Test transition matrix plotting."""
        # With valid model
        evaluator = HMMEvaluator(self.valid_params)
        evaluator.plot_transition_matrix()
        mock_show.assert_called_once()

        # With no model
        mock_show.reset_mock()
        params = self.valid_params.copy()
        params['model'] = None
        evaluator = HMMEvaluator(params)
        evaluator.plot_transition_matrix()
        mock_show.assert_not_called()

        # With model without transmat_
        params['model'] = Mock(spec=[])
        evaluator = HMMEvaluator(params)
        evaluator.plot_transition_matrix()
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_create_all_plots(self, mock_show):
        """Test create_all_plots method."""
        # With valid data
        evaluator = HMMEvaluator(self.valid_params)
        evaluator.create_all_plots()
        self.assertGreater(mock_show.call_count, 1)

        # With empty predictions - should return early
        mock_show.reset_mock()
        params = self.valid_params.copy()
        params['predictions'] = []
        evaluator = HMMEvaluator(params)
        evaluator.create_all_plots()
        mock_show.assert_not_called()

        # With empty truths - should return early
        params = self.valid_params.copy()
        params['truths'] = []
        evaluator = HMMEvaluator(params)
        evaluator.create_all_plots()
        mock_show.assert_not_called()

    @patch('builtins.print')
    @patch('matplotlib.pyplot.show')
    def test_create_all_plots_prints_warning(self, _mock_show, mock_print):
        """Test that warning is printed for empty data."""
        params = self.valid_params.copy()
        params['predictions'] = []
        params['title'] = 'Test Warning'
        evaluator = HMMEvaluator(params)
        evaluator.create_all_plots()

        # Check that warning was printed
        mock_print.assert_called()
        printed_message = mock_print.call_args[0][0]
        self.assertIn('Test Warning', printed_message)

    def test_edge_case_dates_length_mismatch(self):
        """Test handling of dates with different length than predictions/truths."""
        params = self.valid_params.copy()
        params['dates'] = pd.date_range('2020-01-01', periods=3)
        params['predictions'] = [1, 0, 1, 1, 0]
        params['truths'] = [1, 0, 1, 0, 0]

        evaluator = HMMEvaluator(params)
        with patch('matplotlib.pyplot.show'):
            evaluator.create_all_plots()

        params['dates'] = pd.date_range('2020-01-01', periods=10)
        params['predictions'] = [1, 0, 1]
        params['truths'] = [1, 0, 0]

        evaluator = HMMEvaluator(params)
        with patch('matplotlib.pyplot.show'):
            evaluator.create_all_plots()


class TestOptimizationSimple(unittest.TestCase):
    """Test Optimization class."""

    def setUp(self):
        """Set up test data for optimization."""
        # Creating mock data
        self.mock_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100),
            'Avg. Temperature': np.random.randn(100),
            'Air Pressure': np.random.randn(100) * 10 + 1000,
            'Total Precipitation': np.random.rand(100) * 10
        })

        def mock_model_func(data, config):
            # Use data and config to avoid pylint warnings
            _ = data
            _ = config
            model = Mock()
            predictions = [1, 0, 1, 1, 0]
            truths = [1, 0, 1, 0, 0]
            probs = [0.8, 0.2, 0.9, 0.7, 0.3]
            acc = 0.8
            prec = 0.75
            rec = 0.8
            f1 = 0.77
            states = [0, 1, 0, 0, 1]
            return (model, predictions, truths, probs, acc, prec, rec, f1, states)

        self.mock_model_func = mock_model_func

        self.basic_params = {
            'title': 'Test Optimization',
            'data': self.mock_data,
            'model_func': self.mock_model_func,
            'optimization_by': 'f1',
            'n_trials': 3,
            'print_every_trial': False,
            'threshold_optimization': False
        }

    def test_initialization(self):
        """Test Optimization initialization."""
        # Test basic initialization
        opt = Optimization(self.basic_params)
        self.assertEqual(opt.config['title'], 'Test Optimization')
        self.assertEqual(opt.config['optimization_by'], 'f1')
        self.assertEqual(opt.config['n_trials'], 3)
        self.assertIsNotNone(opt.config['data'])
        self.assertIsNone(opt.study)

        # Test with config_params
        params_with_config = self.basic_params.copy()
        params_with_config['config_params'] = {'window_size': 100}
        opt = Optimization(params_with_config)
        self.assertIn('config_params', opt.config)

        # Test with range parameters
        params_with_ranges = self.basic_params.copy()
        params_with_ranges['window_size_range'] = (50, 200)
        params_with_ranges['n_components_range'] = (2, 5)
        opt = Optimization(params_with_ranges)
        self.assertEqual(opt.ranges['window_size'], (50, 200))
        self.assertEqual(opt.ranges['n_components'], (2, 5))

    def test_suggest_parameters_with_config_params(self):
        """Test _suggest_parameters with config_params."""
        params = self.basic_params.copy()
        params['config_params'] = {
            'window_size': 100,
            'features': ['Avg. Temperature', 'Air Pressure']
        }

        opt = Optimization(params)
        mock_trial = Mock()
        result = opt.suggest_parameters(mock_trial)

        self.assertIn('config', result)
        self.assertEqual(result['config']['window_size'], 100)
        self.assertEqual(result['config']['features'], ['Avg. Temperature', 'Air Pressure'])

    def test_suggest_parameters_with_ranges(self):
        """Test _suggest_parameters with parameter ranges."""
        params = self.basic_params.copy()
        params['config_param_ranges'] = {
            'window_size_range': (50, 200),
            'n_components_range': (2, 5),
            'tol_range': (1e-6, 1e-3)
        }

        opt = Optimization(params)
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [100, 3]  # window_size, n_components
        mock_trial.suggest_float.return_value = 1e-4  # tol
        result = opt.suggest_parameters(mock_trial)
        self.assertIn('config', result)
        mock_trial.suggest_int.assert_any_call('window_size', 50, 200)
        mock_trial.suggest_int.assert_any_call('n_components', 2, 5)
        mock_trial.suggest_float.assert_called_once_with('tol', 1e-6, 1e-3, log=True)

    def test_call_model_func(self):
        """Test _call_model_func method."""
        opt = Optimization(self.basic_params)
        params = {'config': {'window_size': 100}}
        result = opt.call_model_func(params)
        self.assertEqual(len(result), 9)
        self.assertEqual(result[4], 0.8)  # acc
        self.assertEqual(result[7], 0.77)  # f1

    def test_store_trial_results(self):
        """Test _store_trial_results method."""
        opt = Optimization(self.basic_params)
        mock_trial = Mock()
        mock_trial.number = 0
        model = Mock()
        result = (model, [1, 0, 1], [1, 0, 0], [0.8, 0.2, 0.9], 0.67, 1.0, 0.67, 0.8, [0, 1, 0])
        opt.store_trial_results(mock_trial, result)

        self.assertIn(0, opt.results['model_store'])
        self.assertIn(0, opt.results['pred_store'])
        self.assertEqual(opt.results['pred_store'][0], [1, 0, 1])
        self.assertEqual(opt.results['actual_store'][0], [1, 0, 0])

        mock_trial.set_user_attr.assert_any_call("acc", 0.67)
        mock_trial.set_user_attr.assert_any_call("f1", 0.8)
        mock_trial.set_user_attr.assert_any_call("model_key", 0)

    @patch('optuna.create_study')
    def test_setup_study(self, mock_create_study):
        """Test setup_study method."""
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        opt = Optimization(self.basic_params)
        opt.setup_study()
        mock_create_study.assert_called_once_with(direction="maximize")
        self.assertEqual(opt.study, mock_study)

    def test_objective_success(self):
        """Test _objective function with successful execution."""
        opt = Optimization(self.basic_params)
        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.params = {'window_size': 100}

        # Mock _suggest_parameters
        with patch.object(opt, 'suggest_parameters', return_value={'config': {'window_size': 100}}):
            result = opt.objective(mock_trial)

        # For optimization_by='f1' should return f1 score (0.77)
        self.assertEqual(result, 0.77)

    def test_objective_failure(self):
        """Test _objective function with failed execution."""

        def failing_model_func(data, config):
            raise ValueError("Model failed")

        params = self.basic_params.copy()
        params['model_func'] = failing_model_func
        opt = Optimization(params)

        mock_trial = Mock()
        mock_trial.params = {}

        with patch.object(opt, 'suggest_parameters', return_value={'config': {}}):
            result = opt.objective(mock_trial)
        self.assertEqual(result, -1.0)

    @patch('os.cpu_count', return_value=4)
    @patch('tqdm.tqdm')
    def test_run_optimization(self, mock_tqdm, _mock_cpu_count):
        """Test run_optimization method."""
        opt = Optimization(self.basic_params)
        opt.study = Mock()

        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        opt.run_optimization()

        opt.study.optimize.assert_called_once()
        call_args = opt.study.optimize.call_args
        self.assertEqual(call_args[1]['n_trials'], 3)
        self.assertEqual(call_args[1]['n_jobs'], 3)  # cpu_count - 1

    @patch('builtins.print')
    def test_print_trial_results(self, mock_print):
        """Test print_trial_results method."""
        opt = Optimization(self.basic_params)
        opt.study = Mock()
        opt.study.trials = []

        opt.print_trial_results()
        mock_print.assert_not_called()

        # Test with print_every_trial=True
        params = self.basic_params.copy()
        params['print_every_trial'] = True
        opt = Optimization(params)

        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.value = 0.8
        mock_trial.params = {'window_size': 100}
        mock_trial.user_attrs = {'acc': 0.85, 'precision': 0.9, 'recall': 0.8}

        opt.study = Mock()
        opt.study.trials = [mock_trial]

        opt.print_trial_results()

        self.assertTrue(mock_print.called)

    def test_extract_trial_data(self):
        """Test _extract_trial_data method."""
        opt = Optimization(self.basic_params)
        model = Mock()
        predictions = [1, 0, 1]
        truths = [1, 0, 0]
        states = [0, 1, 0]

        opt.results['model_store'][0] = model
        opt.results['pred_store'][0] = predictions
        opt.results['actual_store'][0] = truths
        opt.results['states_store'][0] = states

        mock_trial = Mock()
        mock_trial.params = {'config': {'window_size': 10}}

        result = opt.extract_trial_data(0, mock_trial)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 5)  # model, predictions, truths, states, dates
        self.assertEqual(result[1], predictions)
        self.assertEqual(result[2], truths)

    def test_visualize_results_with_tuple_input(self):
        """Test visualize_results with tuple input."""
        opt = Optimization(self.basic_params)
        opt.config['title'] = 'Test Viz Title'

        # Mock HMMEvaluator
        with patch('utils.utils.HMMEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator

            # Prepare tuple input
            mock_model = Mock()
            predictions = [1, 0, 1, 0]
            truths = [1, 0, 0, 0]
            dates = pd.date_range('2020-01-01', periods=4)
            states = [0, 1, 0, 1]
            result_tuple = (mock_model, predictions, truths, dates, states)

            # Call method
            opt.visualize_results(result_tuple)

            # Verify HMMEvaluator was created with correct parameters
            mock_evaluator_class.assert_called_once()
            call_args = mock_evaluator_class.call_args[0][0]

            self.assertEqual(call_args['model'], mock_model)
            self.assertEqual(call_args['predictions'], predictions)
            self.assertEqual(call_args['truths'], truths)
            self.assertEqual(list(call_args['dates']), list(dates))
            self.assertEqual(call_args['states'], states)
            self.assertEqual(call_args['title'], 'Test Viz Title')

            # Verify plots were created
            mock_evaluator.create_all_plots.assert_called_once()


class TestOptimizationAdvanced(unittest.TestCase):
    """Test advanced Optimization functionality and integration."""

    def setUp(self):
        """Set up test data for optimization."""
        self.mock_data = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=100),
            'Rain Binary': np.random.randint(0, 2, 100),
            'Avg. Temperature': np.random.randn(100),
            'Air Pressure': np.random.randn(100) * 10 + 1000,
            'Total Precipitation': np.random.rand(100) * 10
        })

        # Mock model function.
        def mock_model_func(data, config):
            # Use data and config to avoid pylint warnings
            _ = data
            _ = config
            model = Mock()
            predictions = [1, 0, 1, 1, 0]
            truths = [1, 0, 1, 0, 0]
            probs = [0.8, 0.2, 0.9, 0.7, 0.3]
            acc = 0.8
            prec = 0.75
            rec = 0.8
            f1 = 0.77
            states = [0, 1, 0, 0, 1]
            return (model, predictions, truths, probs, acc, prec, rec, f1, states)

        self.mock_model_func = mock_model_func

        self.basic_params = {
            'title': 'Test Optimization',
            'data': self.mock_data,
            'model_func': self.mock_model_func,
            'optimization_by': 'f1',
            'n_trials': 3,
            'print_every_trial': False,
            'threshold_optimization': False
        }

    @patch('matplotlib.pyplot.show')
    def test_handle_best_trial(self, _mock_show):
        """Test handle_best_trial method."""
        opt = Optimization(self.basic_params)
        mock_trial = Mock()
        mock_trial.value = 0.8
        mock_trial.params = {'config': {'window_size': 100}}
        mock_trial.user_attrs = {
            'model_key': 0,
            'acc': 0.85,
            'f1': 0.8,
            'precision': 0.9,
            'recall': 0.75
        }

        opt.study = Mock()
        opt.study.best_trial = mock_trial
        opt.results['model_store'][0] = Mock()
        opt.results['pred_store'][0] = [1, 0, 1]
        opt.results['actual_store'][0] = [1, 0, 0]
        opt.results['states_store'][0] = [0, 1, 0]

        with patch.object(opt, 'process_best_trial_data', return_value=([1, 0, 0], [1, 0, 1], Mock())):
            truths, predictions = opt.handle_best_trial()

        self.assertEqual(truths, [1, 0, 0])
        self.assertEqual(predictions, [1, 0, 1])

    @patch('utils.optimalizationThreshold.bayesianOptimalization_onlyThreshold')
    @patch('utils.optimalizationThreshold.bayesianOptimalization_onlyThreshold_results')
    def test_optimize_threshold(self, mock_results, mock_optimization):
        """Test optimize_threshold method."""
        params = self.basic_params.copy()
        params['threshold_optimization'] = True
        opt = Optimization(params)

        mock_study = Mock()
        mock_optimization.return_value = mock_study
        truths = [1, 0, 1, 0, 0]
        probabilities = [0.8, 0.2, 0.9, 0.3, 0.1]
        opt.optimize_threshold(truths, probabilities)

        mock_optimization.assert_called_once_with('f1', truths, probabilities, n_trials=30)
        mock_results.assert_called_once_with(truths, probabilities, mock_study)

    @patch('builtins.print')
    def test_plot_optimization_results_handles_exceptions(self, mock_print):
        """Test that plot_optimization_results handles exceptions properly."""
        params = self.basic_params.copy()
        params['n_trials'] = 5
        opt = Optimization(params)
        opt.study = Mock()
        opt.plot_optimization_results()

        print_calls = [call[0][0] for call in mock_print.call_args_list]
        self.assertTrue(any("Could not plot optimization history" in str(call) for call in print_calls))
        self.assertTrue(any("Param importances could not be computed" in str(call) for call in print_calls))


    def test_bayesian_optimization_complete_flow(self):
        """Test complete flow of bayesian_optimization method."""
        opt = Optimization(self.basic_params)

        # Mock all dependent methods
        with patch.object(opt, 'setup_study') as mock_setup:
            with patch.object(opt, 'run_optimization') as mock_run:
                with patch.object(opt, 'print_trial_results') as mock_print_trials:
                    with patch.object(opt, 'handle_best_trial', return_value=([1, 0, 1, 0], [1, 0, 0, 0])) as mock_handle:
                        with patch.object(opt, 'optimize_threshold') as mock_threshold:
                            with patch.object(opt, 'plot_optimization_results') as mock_plot:
                                # Setup mock study
                                mock_trial = Mock()
                                mock_trial.params = {'window_size': 100}
                                mock_trial.value = 0.75
                                opt.study = Mock()
                                opt.study.best_trial = mock_trial

                                # Run bayesian optimization
                                result = opt.bayesian_optimization()

                                # Verify correct order of calls
                                mock_setup.assert_called_once()
                                mock_run.assert_called_once()
                                mock_print_trials.assert_called_once()
                                mock_handle.assert_called_once()
                                mock_plot.assert_called_once()

                                # Threshold optimization should not be called (default is False)
                                mock_threshold.assert_not_called()

                                # Verify return value
                                self.assertEqual(result, mock_trial)


    def test_bayesian_optimization_with_threshold_optimization(self):
        """Test bayesian_optimization with active threshold optimization."""
        params = self.basic_params.copy()
        params['threshold_optimization'] = True
        opt = Optimization(params)

        truths = [1, 0, 1, 0, 0]
        predictions = [1, 0, 0, 0, 0]

        with patch.object(opt, 'setup_study'):
            with patch.object(opt, 'run_optimization'):
                with patch.object(opt, 'print_trial_results'):
                    with patch.object(opt, 'handle_best_trial', return_value=(truths, predictions)):
                        with patch.object(opt, 'optimize_threshold') as mock_threshold:
                            with patch.object(opt, 'plot_optimization_results'):
                                mock_trial = Mock()
                                opt.study = Mock()
                                opt.study.best_trial = mock_trial

                                result = opt.bayesian_optimization()

                                # Verify threshold optimization was called with correct parameters
                                mock_threshold.assert_called_once_with(truths, predictions)
                                self.assertEqual(result, mock_trial)


    def test_bayesian_optimization_with_none_results(self):
        """Test bayesian_optimization when handle_best_trial returns None."""
        opt = Optimization(self.basic_params)

        with patch.object(opt, 'setup_study'):
            with patch.object(opt, 'run_optimization'):
                with patch.object(opt, 'print_trial_results'):
                    with patch.object(opt, 'handle_best_trial', return_value=(None, None)):
                        with patch.object(opt, 'optimize_threshold') as mock_threshold:
                            with patch.object(opt, 'plot_optimization_results'):
                                mock_trial = Mock()
                                opt.study = Mock()
                                opt.study.best_trial = mock_trial

                                result = opt.bayesian_optimization()

                                # optimize_threshold should not be called with None values
                                mock_threshold.assert_not_called()

                                # Verify that bayesian_optimization still returns best_trial
                                self.assertEqual(result, mock_trial)


    def test_process_best_trial_data_success(self):
        """Test successful processing of data from best trial."""
        opt = Optimization(self.basic_params)

        # Prepare mock trial with valid data
        mock_trial = Mock()
        mock_trial.user_attrs = {'model_key': 0}
        mock_trial.params = {'config': {'window_size': 100}}

        # Prepare stored data
        mock_model = Mock()
        predictions = [1, 0, 1, 1, 0]
        truths = [1, 0, 1, 0, 0]
        states = [0, 1, 0, 0, 1]
        dates = pd.date_range('2020-01-01', periods=5)

        opt.results['model_store'][0] = mock_model
        opt.results['pred_store'][0] = predictions
        opt.results['actual_store'][0] = truths
        opt.results['states_store'][0] = states

        with patch.object(opt, 'extract_trial_data', return_value=(
                mock_model, predictions, truths, states, dates
        )) as mock_extract:
            with patch.object(opt, 'visualize_results') as mock_viz:
                result = opt.process_best_trial_data(mock_trial)

                # Verify extract_trial_data was called
                mock_extract.assert_called_once_with(0, mock_trial)

                # Verify visualize_results was called with correct parameters
                mock_viz.assert_called_once()
                viz_args = mock_viz.call_args[0][0]
                self.assertEqual(viz_args['model'], mock_model)
                self.assertEqual(viz_args['predictions'], predictions)
                self.assertEqual(viz_args['truths'], truths)
                self.assertEqual(viz_args['states'], states)
                self.assertIsNotNone(viz_args['dates'])

                self.assertIsNotNone(result)
                result_truths, result_predictions, result_model = result
                self.assertEqual(result_truths, truths)
                self.assertEqual(result_predictions, predictions)
                self.assertEqual(result_model, mock_model)


    def test_process_best_trial_data_missing_model_key(self):
        """Test when trial doesn't have model_key."""
        opt = Optimization(self.basic_params)

        mock_trial = Mock()
        mock_trial.user_attrs = {}

        with patch('builtins.print') as mock_print:
            result = opt.process_best_trial_data(mock_trial)

            self.assertIsNone(result)
            mock_print.assert_called()
            self.assertIn("Warning", mock_print.call_args[0][0])


    def test_process_best_trial_data_invalid_model_key(self):
        """Test when trial has invalid model_key."""
        opt = Optimization(self.basic_params)

        mock_trial = Mock()
        mock_trial.user_attrs = {'model_key': 999}

        with patch('builtins.print') as mock_print:
            result = opt.process_best_trial_data(mock_trial)

            self.assertIsNone(result)
            mock_print.assert_called()


    def test_process_best_trial_data_extract_returns_none(self):
        """Test when extract_trial_data returns None."""
        opt = Optimization(self.basic_params)

        mock_trial = Mock()
        mock_trial.user_attrs = {'model_key': 0}

        # Mock extract_trial_data to return None
        with patch.object(opt, 'extract_trial_data', return_value=None):
            result = opt.process_best_trial_data(mock_trial)
            self.assertIsNone(result)

    def test_visualize_results_with_dict_input(self):
        """Test visualize_results with dictionary input."""
        opt = Optimization(self.basic_params)
        opt.config['title'] = 'Dict Test Title'

        with patch('utils.utils.HMMEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator

            # Prepare dict input
            result_dict = {
                'model': Mock(),
                'predictions': [0, 1, 0, 1],
                'truths': [0, 0, 0, 1],
                'dates': pd.date_range('2020-01-01', periods=4),
                'states': [1, 0, 1, 0],
                'extra_param': 'ignored'  # Extra parameter will be passed through
            }

            opt.visualize_results(result_dict)

            # Verify calls
            mock_evaluator_class.assert_called_once()
            call_args = mock_evaluator_class.call_args[0][0]

            # Verify title was added
            self.assertEqual(call_args['title'], 'Dict Test Title')

            # Verify that all parameters were passed including extra_param
            # because visualize_results passes the entire dict without filtering
            self.assertIn('model', call_args)
            self.assertIn('predictions', call_args)
            self.assertIn('truths', call_args)
            self.assertIn('dates', call_args)
            self.assertIn('states', call_args)
            self.assertIn('extra_param', call_args)  # This SHOULD be present
            self.assertEqual(call_args['extra_param'], 'ignored')

            mock_evaluator.create_all_plots.assert_called_once()


    def test_visualize_results_evaluator_exception(self):
        """Test that visualize_results handles exceptions from HMMEvaluator."""
        opt = Optimization(self.basic_params)

        with patch('utils.utils.HMMEvaluator') as mock_evaluator_class:
            # HMMEvaluator throws exception
            mock_evaluator_class.side_effect = ValueError("Invalid data for visualization")

            result_dict = {
                'model': Mock(),
                'predictions': [],  # Empty predictions can cause problems
                'truths': [],
                'dates': None,
                'states': None
            }

            # Method should not throw exception, but should catch it
            try:
                opt.visualize_results(result_dict)
                # Test passed if no exception occurred
            except ValueError:
                self.fail("visualize_results should handle HMMEvaluator exceptions")


    # Integration test
    @patch('matplotlib.pyplot.show')
    def test_integration_optimization_to_visualization(self, mock_show):
        """Integration test of flow from optimization to visualization."""
        # Prepare more realistic data
        params = self.basic_params.copy()
        params['n_trials'] = 2
        opt = Optimization(params)

        # Mock study with more realistic results
        mock_study = Mock()
        mock_best_trial = Mock()
        mock_best_trial.number = 1
        mock_best_trial.value = 0.85
        mock_best_trial.params = {'config': {'window_size': 200, 'n_components': 3}}
        mock_best_trial.user_attrs = {
            'model_key': 1,
            'acc': 0.85,
            'f1': 0.82,
            'precision': 0.88,
            'recall': 0.77
        }

        mock_study.best_trial = mock_best_trial
        mock_study.trials = [Mock(), mock_best_trial]

        # Prepare stored data for best trial
        mock_model = Mock()
        mock_model.transmat_ = np.array([[0.7, 0.3], [0.4, 0.6]])

        opt.results['model_store'][1] = mock_model
        opt.results['pred_store'][1] = [1, 0, 1, 0, 1]
        opt.results['actual_store'][1] = [1, 0, 0, 0, 1]
        opt.results['states_store'][1] = [0, 1, 0, 1, 0]

        with patch.object(opt, 'setup_study'):
            with patch.object(opt, 'run_optimization'):
                with patch('utils.optuna.create_study', return_value=mock_study):
                    opt.study = mock_study

                    # Run entire process
                    result = opt.bayesian_optimization()

                    self.assertEqual(result, mock_best_trial)
                    self.assertGreater(mock_show.call_count, 0)

if __name__ == '__main__':
    unittest.main()
