#handles forward and backward propagation loops

import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss
from ann.activations import softmax


class NeuralNetwork:
    
    INPUT_SIZE  = 784
    NUM_CLASSES = 10

    def __init__(self, cli_args):
        """
        Parameters
        cli_args : argparse.Namespace or dict with fields:
            num_layers    : number of hidden layers
            hidden_size   : int or list[int] neurons per hidden layer
            activation    :'relu','sigmoid', 'tanh'
            weight_init   :'random', 'xavier', 'zeros'
            loss          :'cross_entropy', 'mean_squared_error'
        """
        # Support both Namespace and dict
        if isinstance(cli_args, dict):
            cli_args = _DictNS(cli_args)

        self.args        = cli_args
        self.activation  = cli_args.activation
        self.weight_init = getattr(cli_args, "weight_init", "xavier")
        self.loss_name   = cli_args.loss
        self.input_size  = getattr(cli_args, "input_size", self.INPUT_SIZE)

        # Resolving the hidden layer sizes

        num_layers  = cli_args.num_layers
        hidden_size = cli_args.hidden_size

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size] * num_layers
        elif isinstance(hidden_size, (list, tuple)):
            if len(hidden_size) == 1:
                hidden_sizes = list(hidden_size) * num_layers
            else:
                assert len(hidden_size) == num_layers, (
                    f"hidden_size length {len(hidden_size)} != num_layers {num_layers}"
                )
                hidden_sizes = list(hidden_size)
        else:
            hidden_sizes = [int(hidden_size)] * num_layers

        self.hidden_sizes = hidden_sizes
        self.layers: list[NeuralLayer] = []
        self._build()

        self._loss_fn, self._loss_grad_fn = get_loss(self.loss_name)

        # Public gradient arrays (set after backward())
        self.grad_W = None
        self.grad_b = None

    #network establishment

    def _build(self):
        
        sizes = [self.input_size] + self.hidden_sizes + [self.NUM_CLASSES]
        for i in range(len(sizes) - 1):
            is_output = (i == len(sizes) - 2)
            act = None if is_output else self.activation
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i + 1],
                            activation=act, weight_init=self.weight_init)
            )

    # Forward pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all the layers.
        Returns logits (no softmax applied).
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out  # logits

    # Backward pass

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward propagation to compute the gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        """
        grad_W_list = []
        grad_b_list = []

        # Initial delta: gradient of loss with respect to the logits
        delta = self._loss_grad_fn(y_pred, y_true)

        # Backprop through layers in reverse direction; collect grads so that index 0 = last layer
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        # creating explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b

    # Weight update (called by train loop after backward pass)

    def update_weights(self):
        
        if hasattr(self, "_optimizer") and self._optimizer is not None:
            self._optimizer.update(self.layers)

    # Training loop

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 1, batch_size: int = 32):
        
        from utils.data_loader import get_batches
        N = X_train.shape[0]
        for epoch in range(1, epochs + 1):
            losses = []
            for Xb, yb in get_batches(X_train, y_train, batch_size):
                logits = self.forward(Xb)
                losses.append(self._loss_fn(logits, yb))
                self.backward(yb, logits)
                self.update_weights()
            print(f"Epoch {epoch}/{epochs}  loss={np.mean(losses):.4f}")

    # Evaluation

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_score, recall_score)
        logits = self.forward(X)
        loss   = self._loss_fn(logits, y)
        preds  = np.argmax(logits, axis=1)
        return {
            "logits":    logits,
            "loss":      float(loss),
            "accuracy":  float(accuracy_score(y, preds)),
            "f1":        float(f1_score(y, preds, average="macro", zero_division=0)),
            "precision": float(precision_score(y, preds, average="macro", zero_division=0)),
            "recall":    float(recall_score(y, preds,  average="macro", zero_division=0)),
        }


    def get_weights(self) -> dict:
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict: dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()


    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray,
                     weight_decay: float = 0.0) -> float:
        loss = self._loss_fn(logits, y_true)
        if weight_decay > 0:
            l2 = sum(np.sum(layer.W ** 2) for layer in self.layers)
            loss += 0.5 * weight_decay * l2
        return float(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def get_gradient_norms(self) -> dict:
        
        return {
            f"grad_norm_W_layer_{i}": float(np.linalg.norm(l.grad_W))
            for i, l in enumerate(self.layers)
        }

    def get_activation_stats(self) -> dict:

        # Mean, std, dead-fraction of activations per hidden layer
        
        stats = {}
        for i, layer in enumerate(self.layers[:-1]):
            if layer._activation_out is not None:
                act = layer._activation_out
                stats[f"act_mean_layer_{i}"]     = float(np.mean(act))
                stats[f"act_std_layer_{i}"]      = float(np.std(act))
                stats[f"dead_frac_layer_{i}"]    = float(np.mean(act == 0))
        return stats


class _DictNS:
    def __init__(self, d: dict):
        self.__dict__.update(d)
