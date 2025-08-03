![FYI](FYI.png) <!-- Created in Figma with font: "Fuzzy Bubbles", size of the text: "30" and box sizes: "w: 991px & h: 199px"

# Prague Precipitation Forecasting using Hidden Markov Models


## Abstract

This project implements three Hidden Markov Model variants for **daily precipitation prediction in Prague** using 25 years of meteorological data (2000-2024) from Prague-Ruzyně station. The models include **Discrete HMM**, **Gaussian Mixture HMM**, and **Variational Gaussian HMM**, all trained to predict whether precipitation will occur the next day based on historical weather patterns. Using **Bayesian optimization** for hyperparameter tuning and backtesting with a **sliding window approach**, the **best performing model GMM HMM achieved 64.91% accuracy**, outperforming the naive baseline of 61.27%. The implementation leverages the hmmlearn library and demonstrates how HMMs can capture hidden weather states for precipitation forecasting despite **highly stochastic weather patterns**.

For a more detailed description of the methodology, results, and analysis, please refer to the [staifmatej-report.pdf](staifmatej-report.pdf) file included in this repository.


## Usage

Run the `main.py` file from the root folder and follow the instructions in the terminal.

When running `manual` mode with more trials, warnings may appear - this is normal and nothing to worry about. It simply means the model is searching for hyperparameter types that don't match, and the model reports this. This is expected behavior and the optimization will complete successfully regardless.

## Sample of program work:
```
===== Precipitation Forecasting using Hidden Markov Models =====

(1) - Discrete Hidden Markov Model
(2) - Gaussian Hidden Markov Model with Mixture Emissions
(3) - Variational Gaussian Hidden Markov Model

Press "1", "2" or "3" for choosing your preferable model: 3

Do you would like to backtest only at 20% of Dataset?
(Recommended options for faster backtesting and running time.)

Press "y" for yes or "n" for no: y

================================================================

Forecasting Daily Precipitation Using Variational Gaussian Hidden Markov Model.

Would you like run the model directly with the best hyperparameters found through Bayesian optimization or set hyperparameters yourself?
Type 'auto' to Run with best predefined parameters find by Bayesian Optimization. Bayesian optimization or 'manual' to set hyperparameters yourself:
auto

Starting Variational Gaussian HMM Model Backtesting...

Optimizing hyperparameters...
Optimization Progress: 100%|███████████████| 1/1 [07:40<00:00, 460.92s/it]

Optimizing threshold...
Optimization Progress: 100%|█████████████| 30/30 [00:00<00:00, 210.75it/s]

===== Results with Optimal Threshold =====
Optimal threshold: 0.0963
Accuracy:       0.6459
Precision:      0.5922
Recall:         0.5214
F1 Score:       0.5545
==========================================
```
**Recommendation:**
Run in faster mode with 20% of the training dataset for quicker program execution, otherwise the program runs quite long time.

**Notes:**
In `auto` mode, the most optimal hyperparameters found through Bayesian Optimization are preconfigured.

## Installation

- Clone the repository using SSH or HTTPS
    - **SSH:** `git@github.com:staifmatej/prg-precipitation-forecast-hmm.git`
    - **HTTPS:** `https://github.com/staifmatej/prg-precipitation-forecast-hmm.git`

- Navigate to the project directory

    - `cd prg-precipitation-forecast-hmm`

- Create virtual environment and install dependencies:

    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`

## Testing

To run the tests, execute `pytest` directly in the main project directory (**root folder**).

## Codestyle

To check code style compliance, run `pylint . --disable=C0301,C0103` from the main project directory. This will analyze all Python files while ignoring line length (C0301) and naming convention (C0103) warnings.
