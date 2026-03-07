#objective functions and their derivative defining

import numpy as np
from ann.activations import softmax

#cross entropy error

def cross_entropy_loss(logits: np.ndarray, y_true: np.ndarray) -> float:

    probs = softmax(logits)
    batch_size = logits.shape[0]
    eps = 1e-12
    correct_log_probs = np.log(
        np.clip(probs[np.arange(batch_size), y_true], eps, 1.0)
    )
    return -np.mean(correct_log_probs)


def cross_entropy_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    
    batch_size = logits.shape[0]
    probs = softmax(logits)
    probs[np.arange(batch_size), y_true] -= 1.0
    return probs / batch_size   # normalised

# Mean Squared Error

def mse_loss(logits: np.ndarray, y_true: np.ndarray) -> float:

    y_one_hot = _one_hot(y_true, logits.shape[1])
    return np.mean((logits - y_one_hot) ** 2)


def mse_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:

    batch_size, num_classes = logits.shape
    y_one_hot = _one_hot(y_true, num_classes)
    return 2.0 * (logits - y_one_hot) / (batch_size)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# Dispatching the loss and their derivatives

LOSS_FUNCTIONS = {
    "cross_entropy":      (cross_entropy_loss, cross_entropy_gradient),
    "mean_squared_error":   (mse_loss, mse_gradient)
}


def get_loss(name):
    return LOSS_FUNCTIONS[name]
