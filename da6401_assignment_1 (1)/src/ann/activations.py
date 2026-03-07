
import numpy as np

#Activation functions and their activations generation respectively

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    '''the z-max(z)shift is necessary,
     without it large value of z can cause overflow of exp(z) to infinity'''

#dispatch maps for the activations & their derivatives respectively

ACTIVATIONS = {
    "relu":    relu,
    "sigmoid": sigmoid,
    "tanh":    tanh,
}

ACTIVATION_DERIVATIVES = {
    "relu":    relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh":    tanh_derivative,
}


def get_activation(name):
    return ACTIVATIONS[name]


def get_activation_derivative(name):
    return ACTIVATION_DERIVATIVES[name]
