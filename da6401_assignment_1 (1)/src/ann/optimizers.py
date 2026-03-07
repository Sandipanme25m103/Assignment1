
import numpy as np

#class generating for all types of optimizers

class Optimizer:

    def update(self, layers):
        raise NotImplementedError

# SGD

class SGD(Optimizer):
    """
    Mini-batch Stochastic Gradient Descent with L2 weight decay.
    """

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.wd = weight_decay

    def update(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.wd * layer.W)
            layer.b -= self.lr * layer.grad_b


# Momentum

class Momentum(Optimizer):
    """
    SGD with classical momentum.
    v_t = beta * v_(t-1) + grad
    w   = w - lr * v_t
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9,
                 weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta = beta
        self.wd = weight_decay
        self._v = {}   

    def _vel(self, i, layer):
        if i not in self._v:
            self._v[i] = {"W": np.zeros_like(layer.W),
                          "b": np.zeros_like(layer.b)}
        return self._v[i]

    def update(self, layers):
        for i, layer in enumerate(layers):
            v = self._vel(i, layer)
            v["W"] = self.beta * v["W"] + layer.grad_W + self.wd * layer.W
            v["b"] = self.beta * v["b"] + layer.grad_b
            layer.W -= self.lr * v["W"]
            layer.b -= self.lr * v["b"]


# NAG  (Nesterov Accelerated Gradient)

class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient.

    Three-step usage per training iteration:
      1. optimizer.apply_lookahead(model.layers) :shift to look-ahead point
      2. forward + backward pass
      3. optimizer.restore_and_update(model.layers) : restore & apply Nesterov update
    """

    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9,
                 weight_decay: float = 0.0):
        self.lr = learning_rate
        self.beta = beta
        self.wd = weight_decay
        self._v = {}

    def _vel(self, i, layer):
        if i not in self._v:
            self._v[i] = {"W": np.zeros_like(layer.W),
                          "b": np.zeros_like(layer.b)}
        return self._v[i]

    def apply_lookahead(self, layers):
        """Shift weights to look-ahead position before forward/backward pass."""
        for i, layer in enumerate(layers):
            v = self._vel(i, layer)
            layer.W -= self.beta * v["W"]
            layer.b -= self.beta * v["b"]

    def restore_and_update(self, layers):
        """Restore weights from look-ahead and apply Nesterov update."""
        for i, layer in enumerate(layers):
            v = self._vel(i, layer)
            layer.W += self.beta * v["W"]
            layer.b += self.beta * v["b"]
            v["W"] = self.beta * v["W"] + layer.grad_W + self.wd * layer.W
            v["b"] = self.beta * v["b"] + layer.grad_b
            layer.W -= self.lr * v["W"]
            layer.b -= self.lr * v["b"]

    def update(self, layers):
        self.restore_and_update(layers)


# RMSProp


class RMSProp(Optimizer):
    """
    RMSProp.
    s_t = rho * s_(t-1) + (1 - rho) * grad^2
    w   = w - lr * grad / sqrt(s_t + eps)
    """

    def __init__(self, learning_rate: float = 0.001, rho: float = 0.9,
                 epsilon: float = 1e-8, weight_decay: float = 0.0):
        self.lr = learning_rate
        self.rho = rho
        self.eps = epsilon
        self.wd = weight_decay
        self._s = {}

    def _cache(self, i, layer):
        if i not in self._s:
            self._s[i] = {"W": np.zeros_like(layer.W),
                          "b": np.zeros_like(layer.b)}
        return self._s[i]

    def update(self, layers):
        for i, layer in enumerate(layers):
            s = self._cache(i, layer)
            grad_W = layer.grad_W + self.wd * layer.W
            s["W"] = self.rho * s["W"] + (1 - self.rho) * grad_W ** 2
            s["b"] = self.rho * s["b"] + (1 - self.rho) * layer.grad_b ** 2
            layer.W -= self.lr * grad_W / (np.sqrt(s["W"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(s["b"]) + self.eps)


# Factory

OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nag":      NAG,
    "rmsprop":  RMSProp,
}


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Instantiate and return an optimizer by its name."""
    key = name.lower()
    return OPTIMIZERS[key](**kwargs)
