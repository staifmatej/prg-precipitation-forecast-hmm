"""Tests for main.py module."""

import unittest
from unittest.mock import patch, Mock
import sys

class TestMain(unittest.TestCase):
    """Test main.py functionality."""

    def setUp(self):
        """Set up test by saving original modules and mocking them."""
        # Save original modules if they exist
        self.original_modules = {}
        self.mock_modules = ['discreteHMM', 'GMM_HMM', 'variationalGaussianHMM']

        for module_name in self.mock_modules:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]
            sys.modules[module_name] = Mock()

        # Import main after mocking
        from main import main
        self.main = main

    def tearDown(self):
        """Clean up by restoring original modules."""
        # Remove mocked modules
        for module_name in self.mock_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

        # Restore original modules
        for module_name, original_module in self.original_modules.items():
            sys.modules[module_name] = original_module

        # Remove main from modules to ensure clean import next time
        if 'main' in sys.modules:
            del sys.modules['main']

    def test_model_1_full(self):
        """Test Discrete HMM with full dataset."""
        with patch('builtins.input', side_effect=['1', 'n']):
            with patch('main.run_discrete_hmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=False)

    def test_model_2_short(self):
        """Test GMM HMM with 20% dataset."""
        with patch('builtins.input', side_effect=['2', 'y']):
            with patch('main.run_gmm_hmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=True)

    def test_model_3(self):
        """Test Variational Gaussian model."""
        with patch('builtins.input', side_effect=['3', 'n']):
            with patch('main.run_vghmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=False)

    def test_model_3_short(self):
        """Test Variational Gaussian model with 20% dataset."""
        with patch('builtins.input', side_effect=['3', 'y']):
            with patch('main.run_vghmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=True)

    def test_invalid_model_max_attempts(self):
        """Test max attempts for model choice - fails after 3 attempts."""
        with patch('builtins.input', side_effect=['x', 'y', 'z']):
            self.assertEqual(self.main(), 1)

    def test_invalid_dataset_max_attempts(self):
        """Test max attempts for dataset choice - fails after 3 attempts."""
        with patch('builtins.input', side_effect=['1', 'a', 'b', 'c']):
            self.assertEqual(self.main(), 1)

    def test_case_insensitive(self):
        """Test case insensitive input."""
        with patch('builtins.input', side_effect=['1', 'Y']):
            with patch('main.run_discrete_hmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=True)

    def test_whitespace_handling(self):
        """Test whitespace handling in inputs."""
        with patch('builtins.input', side_effect=['  1  ', '  n  ']):
            with patch('main.run_discrete_hmm') as mock:
                self.main()
                mock.assert_called_once_with(short_dataset=False)

    def test_invalid_then_valid_model(self):
        """Test recovery from invalid model inputs - max 3 attempts."""
        with patch('builtins.input', side_effect=['5', 'abc', '2', 'y']):
            with patch('builtins.print') as mock_print:
                with patch('main.run_gmm_hmm') as mock_model:
                    result = self.main()
                    self.assertIsNone(result)
                    mock_model.assert_called_once_with(short_dataset=True)
                    # Check error messages with [i/3] format
                    error_messages = [call for call in mock_print.call_args_list
                                      if 'Wrong Input. Try again.' in str(call)]
                    self.assertEqual(len(error_messages), 2)

    def test_invalid_then_valid_dataset(self):
        """Test recovery from invalid dataset inputs - max 3 attempts."""
        with patch('builtins.input', side_effect=['3', 'maybe', 'y', 'n']):
            with patch('main.run_vghmm') as mock_model:
                result = self.main()
                self.assertIsNone(result)
                mock_model.assert_called_once_with(short_dataset=True)

    def test_exactly_three_invalid_attempts_allowed(self):
        """Test that exactly 3 invalid attempts are allowed before failure."""
        # Test for model choice
        with patch('builtins.input', side_effect=['wrong1', 'wrong2', 'wrong3']):
            with patch('builtins.print') as mock_print:
                result = self.main()
                self.assertEqual(result, 1)
                # Check error messages
                print_calls = [str(call) for call in mock_print.call_args_list]
                # Should have 2x "Try again" and 1x final error
                try_again_count = sum(1 for call in print_calls if 'Try again' in call)
                final_error_count = sum(1 for call in print_calls if '[3/3]' in call and 'Try again' not in call)
                self.assertEqual(try_again_count, 2)
                self.assertEqual(final_error_count, 1)

    def test_successful_execution_returns_none(self):
        """Test that successful execution returns None."""
        with patch('builtins.input', side_effect=['1', 'n']):
            with patch('main.run_discrete_hmm'):
                result = self.main()
                self.assertIsNone(result)

    def test_error_messages_show_attempt_count(self):
        """Test that error messages show attempt count [i/3]."""
        with patch('builtins.input', side_effect=['x', 'y', 'z']):
            with patch('builtins.print') as mock_print:
                self.main()
                # Check that error messages contain [1/3], [2/3], [3/3]
                print_calls = [str(call) for call in mock_print.call_args_list]
                self.assertTrue(any('[1/3]' in call for call in print_calls))
                self.assertTrue(any('[2/3]' in call for call in print_calls))
                self.assertTrue(any('[3/3]' in call for call in print_calls))


if __name__ == '__main__':
    unittest.main()
