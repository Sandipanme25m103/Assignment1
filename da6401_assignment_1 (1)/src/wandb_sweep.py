"""
W&B hyperparameter sweep configuration and runner.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
        "batch_size":    {"values": [32, 64, 128]},
        "num_layers":    {"values": [2, 3, 4, 5]},
        "hidden_size":   {"values": [32, 64, 128]},
        "activation":    {"values": ["sigmoid", "tanh", "relu"]},
        "optimizer":     {"values": ["sgd", "momentum", "nag", "rmsprop"]},
        "weight_init":   {"values": ["random", "xavier"]},
        "weight_decay":  {"values": [0.0, 1e-4, 1e-3]},
        "loss":          {"values": ["cross_entropy", "mean_squared_error"]},
        "epochs":        {"value": 20},
        "dataset":       {"value": "fashion_mnist"},
    },
}


def sweep_train():
    import wandb
    run = wandb.init()
    cfg = wandb.config

    class Args:
        dataset        = cfg.dataset
        epochs         = cfg.epochs
        batch_size     = cfg.batch_size
        loss           = cfg.loss
        optimizer      = cfg.optimizer
        learning_rate  = cfg.learning_rate
        weight_decay   = cfg.weight_decay
        num_layers     = cfg.num_layers
        hidden_size    = cfg.hidden_size
        activation     = cfg.activation
        weight_init    = cfg.weight_init
        wandb_project  = "da6401-assignment1"
        wandb_entity   = None
        no_wandb       = False
        model_save_path = f"sweep_model_{run.id}.npy"
        config_path    = f"sweep_config_{run.id}.json"

    from train import main as train_main
    import argparse

    # Monkey-patch parse_arguments to return our Args
    import train as train_module
    orig = train_module.parse_arguments
    train_module.parse_arguments = lambda: Args()
    try:
        train_main()
    finally:
        train_module.parse_arguments = orig
        for f in [Args.model_save_path, Args.config_path]:
            if os.path.exists(f):
                os.remove(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="da6401-assignment1")
    parser.add_argument("--entity",  type=str, default=None)
    parser.add_argument("--count",   type=int, default=100)
    args = parser.parse_args()

    import wandb
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project, entity=args.entity)
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, function=sweep_train, count=args.count)


if __name__ == "__main__":
    main()
