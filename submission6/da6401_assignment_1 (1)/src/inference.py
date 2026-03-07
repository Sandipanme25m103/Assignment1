
#Evaluate trained models on test sets

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


# Argument parsing  (same args as train.py)

def parse_arguments():
    """
    Parse command-line arguments for inference.

    Arguments
    ---------
    - model_path  : Path to saved model weights (relative path)
    - dataset     : Dataset to evaluate on
    - batch_size  : Batch size for inference
    - num_layers  : Number of hidden layers
    - hidden_size : Number of neurons in hidden layers
    - activation  : Activation function ('relu', 'sigmoid', 'tanh')
    - loss        : Loss function ('cross_entropy', 'mean_squared_error')
    - weight_init : Weight initialisation method
    - split       : Which split to evaluate ('train', 'val', 'test')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",         type=int,   default=15)
    parser.add_argument("-b",   "--batch_size",     type=int,   default=64)
    parser.add_argument("-lr",  "--learning_rate",  type=float, default=0.001)
    parser.add_argument("-o",   "--optimizer",      type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    parser.add_argument("-nhl", "--num_layers",     type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",    type=int,   nargs="+", default=[128])
    parser.add_argument("-a",   "--activation",     type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l",   "--loss",           type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"])
    parser.add_argument("-w_i", "--weight_init",    type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-wd",  "--weight_decay",   type=float, default=0.0)
    parser.add_argument("-w_p", "--wandb_project",  type=str,   default="da6401-assignment1")
    parser.add_argument("--model_path",             type=str,   default="best_model.npy",
                        help="Relative path to saved model weights (.npy).")
    parser.add_argument("--split",                  type=str,   default="test",
                        choices=["train", "val", "test"])


    return parser.parse_args()


# Model loading  (matches updated instructions verbatim)

def load_model(model_path: str) -> dict:
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


# Evaluation

def evaluate_model(model: NeuralNetwork, X_test: np.ndarray,
                   y_test: np.ndarray) -> dict:
    
    """
    Evaluate the model on test data.
    gives the dictionary with keys: logits, loss, accuracy, f1, precision, recall
    """
    return model.evaluate(X_test, y_test)


# Main function definition

def main() -> dict:
    """
    Main inference function.
    Returns the dictionary with keys: logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]


    # Loading data

    print(f"Loading {args.dataset}...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.dataset)

    split_map = {
        "train": (X_train, y_train),
        "val":   (X_val,   y_val),
        "test":  (X_test,  y_test),
    }
    X, y = split_map[args.split]
    print(f"Evaluating on '{args.split}' split: {X.shape}")

    # Building the model and loading weights

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    print(f"Loaded weights from '{args.model_path}'.")

    # Evaluation
    results = evaluate_model(model, X, y)

    print("\n" + "=" * 50)
    print(f"  Split      : {args.split}")
    print(f"  Accuracy   : {results['accuracy']:.5f}")
    print(f"  Precision  : {results['precision']:.5f}  (macro)")
    print(f"  Recall     : {results['recall']:.5f}  (macro)")
    print(f"  F1-score   : {results['f1']:.5f}  (macro)")
    print(f"  Loss       : {results['loss']:.5f}")
    print("=" * 50)

    print("Evaluation complete!")
    return results


if __name__ == '__main__':
    main()
