"""Main auxiliary and provisional module for running Hidden Markov Models for precipitation forecasting."""

from discreteHMM import main as run_discrete_hmm
from GMM_HMM import main as run_gmm_hmm
from variationalGaussianHMM import main as run_vghmm

def main():
    """Main function for selecting and running HMM models."""

    # We will display the model selection options.
    print("\n===== Precipitation Forecasting using Hidden Markov Models =====\n")

    print("(1) - Discrete Hidden Markov Model")
    print("(2) - Gaussian Hidden Markov Model with Mixture Emissions")
    print("(3) - Variational Gaussian Hidden Markov Model")

    # Getting the model choice.
    i = 0
    while True:
        i += 1
        model_choice = input("\nPress \"1\", \"2\" or \"3\" for choosing your preferable model: ").strip()
        if model_choice in ["1", "2", "3"]:
            break
        if i < 3:
            print(f"Wrong Input. Try again. [{i}/3]")
        if i >= 3:
            print(f"Wrong Input. [{i}/3]")
            return 1

    # Ask about the dataset size.
    print("\nDo you would like to backtest only at 20% of Dataset?")
    print("(Recommended options for faster backtesting and running time.)")

    i = 0
    while True:
        i += 1
        dataset_choice = input("\nPress \"y\" for yes or \"n\" for no: ").strip().lower()
        if dataset_choice in ["y", "n"]:
            break
        if i < 3:
            print(f"Wrong Input. Try again. [{i}/3]")
        if i >= 3:
            print(f"Wrong Input. [{i}/3]")
            return 1

    short_dataset = dataset_choice == "y"
    print("\n================================================================\n")

    # Run select model.
    if model_choice == "1":
        run_discrete_hmm(short_dataset=short_dataset)

    elif model_choice == "2":
        run_gmm_hmm(short_dataset=short_dataset)

    elif model_choice == "3":
        run_vghmm(short_dataset=short_dataset)

    return None

if __name__ == "__main__":
    main()
