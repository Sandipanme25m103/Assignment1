#Neural Layer handles the weight initialization, forward pass, and gradient computation

import numpy as np
from ann.activations import get_activation, get_activation_derivative


class NeuralLayer:
    

    def __init__(self, input_size: int, output_size: int,
                 activation: str = "relu", weight_init: str = "xavier"):
       
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.W, self.b = self._init_weights(weight_init) # Initialializing weights and biases

        # Gradient placeholders – filled by backward()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Cache for backprop
        self._input = None          # a^{l-1} coming in
        self._z = None              # pre-activation: z = a W + b
        self._activation_out = None # post-activation: a^l
 
    # Weight initialisation

    def _init_weights(self, method: str):
        if method == "zeros":
            W = np.zeros((self.input_size, self.output_size))
            b = np.zeros((1, self.output_size))

        elif method == "random":
            W = np.random.randn(self.input_size, self.output_size) * 0.01
            b = np.zeros((1, self.output_size))

        elif method == "xavier":
            # Xavier
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            W = np.random.uniform(-limit, limit, (self.input_size, self.output_size))
            b = np.zeros((1, self.output_size))

        return W, b

    # Forward pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        
        self._input = x
        self._z = x @ self.W + self.b  # linear combination

        if self.activation_name is None or self.activation_name == "linear":
            self._activation_out = self._z
        else:
            act_fn = get_activation(self.activation_name)
            self._activation_out = act_fn(self._z)

        return self._activation_out

    # Backward pass

    def backward(self, delta: np.ndarray) -> np.ndarray:
        
        if self.activation_name is None or self.activation_name == "linear":
            dz = delta
        else:
            act_deriv = get_activation_derivative(self.activation_name)
            dz = delta * act_deriv(self._z)   # element-wise multiplication

        # Accumulating gradients (delta already normalised by N from loss grad)
        self.grad_W = self._input.T @ dz          # (input_size, output_size)
        self.grad_b = np.sum(dz, axis=0, keepdims=True)  # (1, output_size)

        # Propagate to previous layer
        delta_prev = dz @ self.W.T              # (batch_size, input_size)
        return delta_prev

    # Utility

    def get_weights(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_weights(self, weight_dict: dict):
        self.W = weight_dict["W"].copy()
        self.b = weight_dict["b"].copy()
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
