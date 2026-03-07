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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-d",   "--dataset",         type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",           type=int,   default=20)
    parser.add_argument("-b",   "--batch_size",       type=int,   default=64)
    parser.add_argument("-lr",  "--learning_rate",    type=float, default=0.001)
    parser.add_argument("-o",   "--optimizer",        type=str,   default="rmsprop",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-nhl", "--num_layers",       type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",      type=int,   nargs="+", default=[128])
    parser.add_argument("-a",   "--activation",       type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-l",   "--loss",             type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse", "mean_squared_error"])
    parser.add_argument("-w_i", "--weight_init",      type=str,   default="xavier",
                        choices=["random", "xavier", "zeros"])
    parser.add_argument("-wd",  "--weight_decay",     type=float, default=0.0)
    parser.add_argument("-w_p", "--wandb_project",    type=str,   default="assignment_1")
    parser.add_argument("--wandb_entity",             type=str,   default=None)
    parser.add_argument("--no_wandb",                 action="store_true")
    parser.add_argument("--model_save_path",          type=str,   default="best_model.npy")
    parser.add_argument("--config_path",              type=str,   default="best_config.json")
    parser.add_argument("--sweep",                    action="store_true")
    parser.add_argument("--sweep_count",              type=int,   default=100)
    parser.add_argument("--sweep_yaml",               type=str,   default="../sweep.yaml")

    return parser.parse_args()


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


def _log_confusion_matrix(run, model, X_test, y_test, dataset):
    if run is None:
        return
    try:
        import wandb
        from utils.data_loader import CLASS_NAMES
        preds = model.predict(X_test)
        names = CLASS_NAMES[dataset]
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test.tolist(),
                preds=preds.tolist(),
                class_names=names,
            )
        })
        print("Logged confusion matrix to W&B.")
    except Exception as e:
        print("Could not log confusion matrix: {}".format(e))


def run_training(args):
    if isinstance(args.hidden_size, list) and len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size[0]

    print("Loading {}...".format(args.dataset))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_dataset(args.dataset)
    print("  Train: {}  Val: {}  Test: {}".format(
        X_train.shape, X_val.shape, X_test.shape))

    model = NeuralNetwork(args)
    print("Architecture: {}".format([784] + model.hidden_sizes + [10]))

    opt_kwargs = dict(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = get_optimizer(args.optimizer, **opt_kwargs)
    model._optimizer = optimizer
    is_nag = isinstance(optimizer, NAG)

    run = _init_wandb(args)

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

        if val_metrics["f1"] > best_val_f1:
            best_val_f1  = val_metrics["f1"]
            best_weights = model.get_weights()

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

    _log_confusion_matrix(run, model, X_test, y_test, args.dataset)

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


def run_sweep(args):
    import yaml
    import wandb

    yaml_path = os.path.abspath(args.sweep_yaml)
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(__file__), args.sweep_yaml)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("sweep.yaml not found at: {}".format(yaml_path))

    with open(yaml_path, "r") as f:
        sweep_config = yaml.safe_load(f)

    print("Loaded sweep config from: {}".format(yaml_path))

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project,
                           entity=args.wandb_entity)
    print("Sweep ID: {}".format(sweep_id))

    def sweep_run():
        import wandb
        wandb.init()
        cfg = wandb.config

        args.dataset       = getattr(cfg, "dataset",       args.dataset)
        args.epochs        = getattr(cfg, "epochs",        args.epochs)
        args.batch_size    = getattr(cfg, "batch_size",    args.batch_size)
        args.learning_rate = getattr(cfg, "learning_rate", args.learning_rate)
        args.optimizer     = getattr(cfg, "optimizer",     args.optimizer)
        args.num_layers    = getattr(cfg, "num_layers",    args.num_layers)
        hs = getattr(cfg, "hidden_size", args.hidden_size)
        args.hidden_size   = hs[0] if isinstance(hs, (list, tuple)) else int(hs)
        args.activation    = getattr(cfg, "activation",    args.activation)
        args.loss          = getattr(cfg, "loss",          args.loss)
        args.weight_init   = getattr(cfg, "weight_init",   args.weight_init)
        args.weight_decay  = getattr(cfg, "weight_decay",  args.weight_decay)
        args.no_wandb      = True
        args.model_save_path = "sweep_model_{}.npy".format(wandb.run.id)
        args.config_path     = "sweep_config_{}.json".format(wandb.run.id)

        run_training(args)

        for f in [args.model_save_path, args.config_path]:
            if os.path.exists(f):
                os.remove(f)

    wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)


def main():
    args = parse_arguments()
    if args.sweep:
        run_sweep(args)
    else:
        run_training(args)


if __name__ == '__main__':
    main()






















