# Main Training Script

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer, NAG
from utils.data_loader import load_dataset, get_batches


# Argument parsing

def parse_arguments():
    """
    Arguments

    - dataset        : 'mnist' or 'fashion_mnist'
    - epochs         : Number of training epochs
    - batch_size     : Mini-batch size
    - learning_rate  : Learning rate for optimizer
    - optimizer      : 'sgd', 'momentum', 'nag', 'rmsprop'
    - num_layers     : Number of hidden layers
    - hidden_size    : Number of neurons in hidden layers
    - activation     : Activation function ('relu', 'sigmoid', 'tanh')
    - loss           : Loss function ('cross_entropy', 'mean_squared_error')
    - weight_init    : Weight initialization method
    - weight_decay   : L2 regularisation coefficient
    - wandb_project  : W&B project name
    - model_save_path: Path to save trained model (relative path)
    """
    
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-d",   "--dataset",         type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"],
                        help="Dataset to use.")
    parser.add_argument("-e",   "--epochs",           type=int,   default=20,
                        help="Number of training epochs.")
    parser.add_argument("-b",   "--batch_size",       type=int,   default=64,
                        help="Mini-batch size.")
    parser.add_argument("-lr",  "--learning_rate",    type=float, default=0.001,
                        help="Learning rate for optimizer.")
    parser.add_argument("-o",   "--optimizer",        type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop"],
                        help="Optimizer.")
    parser.add_argument("-nhl", "--num_layers",       type=int,   default=3,
                        help="Number of hidden layers.")
    parser.add_argument("-sz",  "--hidden_size",      type=int,   nargs="+", default=[128],
                        help="Neurons per hidden layer (single int or list).")
    parser.add_argument("-a",   "--activation",       type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"],
                        help="Activation function for hidden layers.")
    parser.add_argument("-l",   "--loss",             type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error"],
                        help="Loss function.")
    parser.add_argument("-w_i", "--weight_init",      type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"],
                        help="Weight initialisation strategy.")
    parser.add_argument("-wd",  "--weight_decay",     type=float, default=0.0,
                        help="L2 weight decay coefficient.")
    parser.add_argument("-w_p", "--wandb_project",    type=str,   default="da6401-assignment1",
                        help="Weights & Biases project name.")
    parser.add_argument("--wandb_entity",             type=str,   default=None,
                        help="W&B entity (username or team).")
    parser.add_argument("--no_wandb",                 action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--model_save_path",          type=str,   default="best_model.npy",
                        help="Relative path to save trained model weights.")
    parser.add_argument("--config_path",              type=str,   default="best_config.json",
                        help="Relative path to save best config JSON.")

    return parser.parse_args()


# W&B helpers

def _init_wandb(args):
    if args.no_wandb:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "loss":          args.loss,
                "optimizer":     args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay":  args.weight_decay,
                "num_layers":    args.num_layers,
                "hidden_size":   args.hidden_size,
                "activation":    args.activation,
                "weight_init":   args.weight_init,
            },
        )
        return run
    except Exception as e:
        print("W&B init failed: {}. Continuing without W&B.".format(e))
        return None


def _log(run, metrics, step=None):
    if run is None:
        return
    try:
        import wandb
        wandb.log(metrics, step=step)
    except Exception:
        pass


# Main

def main():
    # Main training function.
    args = parse_arguments()

    # Flatten hidden_size list of length 1 -> int (NeuralNetwork handles replication)
    
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    # -- Data --
    print("Loading {}...".format(args.dataset))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.dataset)
    print("  Train: {}  Val: {}  Test: {}".format(X_train.shape, X_val.shape, X_test.shape))

    # -- Model --
    model = NeuralNetwork(args)
    print("Architecture: {}".format([784] + model.hidden_sizes + [10]))

    # -- Optimizer --
    opt_kwargs = dict(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = get_optimizer(args.optimizer, **opt_kwargs)
    model._optimizer = optimizer
    is_nag = isinstance(optimizer, NAG)

    # -- W&B --
    run = _init_wandb(args)

    # -- Training loop --
    best_val_f1  = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        batch_losses = []

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            if is_nag:
                optimizer.apply_lookahead(model.layers)

            logits = model.forward(X_batch)
            batch_losses.append(model.compute_loss(logits, y_batch, args.weight_decay))
            model.backward(y_batch, logits)

            if is_nag:
                optimizer.restore_and_update(model.layers)
            else:
                optimizer.update(model.layers)

        # -- Epoch metrics --
        train_loss    = float(np.mean(batch_losses))
        val_metrics   = model.evaluate(X_val,   y_val)
        train_metrics = model.evaluate(X_train, y_train)

        print("Epoch {:3d}/{} | train_loss={:.5f} | train_acc={:.5f} | val_acc={:.5f} | val_f1={:.5f} | {:.2f}s".format(
            epoch, args.epochs, train_loss,
            train_metrics["accuracy"], val_metrics["accuracy"],
            val_metrics["f1"], time.time() - t0
        ))

        log_dict = {
            "epoch":          epoch,
            "train/loss":     train_loss,
            "train/accuracy": train_metrics["accuracy"],
            "val/loss":       val_metrics["loss"],
            "val/accuracy":   val_metrics["accuracy"],
            "val/f1":         val_metrics["f1"],
        }
        for k, v in model.get_gradient_norms().items():
            log_dict["grad/" + k] = v
        for k, v in model.get_activation_stats().items():
            log_dict["act/" + k] = v
        _log(run, log_dict, step=epoch)

        # Saving the best weights by validation F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_weights = model.get_weights()

    # -- Restoring best & evaluate on test --
    model.set_weights(best_weights)
    test_metrics = model.evaluate(X_test, y_test)
    print("\n--- Test Set ---")
    print("  Accuracy : {:.5f}".format(test_metrics["accuracy"]))
    print("  Precision: {:.5f}".format(test_metrics["precision"]))
    print("  Recall   : {:.5f}".format(test_metrics["recall"]))
    print("  F1-score : {:.5f}".format(test_metrics["f1"]))

    _log(run, {
        "test/accuracy":  test_metrics["accuracy"],
        "test/precision": test_metrics["precision"],
        "test/recall":    test_metrics["recall"],
        "test/f1":        test_metrics["f1"],
    })

    # -- Save model & config --
    np.save(args.model_save_path, best_weights)
    print("Saved model -> {}".format(args.model_save_path))

    config = {
        "dataset":       args.dataset,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "loss":          args.loss,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay":  args.weight_decay,
        "num_layers":    args.num_layers,
        "hidden_size":   args.hidden_size if isinstance(args.hidden_size, list) else [args.hidden_size],
        "activation":    args.activation,
        "weight_init":   args.weight_init,
        "val_f1":        best_val_f1,
        "test_f1":       test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"],
    }
    with open(args.config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Saved config -> {}".format(args.config_path))

    if run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("Training complete!")
    return model, test_metrics


if __name__ == '__main__':
    main()
























