

import numpy as np
from sklearn.model_selection import train_test_split


# Dataset loading

def load_dataset(name: str):
    """
    Load MNIST or Fashion-MNIST via Keras and return train/val/test splits.

    Parameters
    name : 'mnist' or 'fashion_mnist'

    Returns
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    All X arrays : float32, normalised to [0, 1], flattened to (N, 784)
    All y arrays : int32 integer class labels
    """
    name = name.lower().replace("-", "_")

    if name == "mnist":
        from keras.datasets import mnist
        (X_full, y_full), (X_test, y_test) = mnist.load_data()
    elif name == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_full, y_full), (X_test, y_test) = fashion_mnist.load_data()
    

    # Flatten 28×28 → 784 and normalise to [0, 1]
    
    X_full = X_full.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_full = y_full.astype(np.int32)
    y_test = y_test.astype(np.int32)

    # 10 % stratified validation split

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full,
        test_size=0.1,
        random_state=42,
        stratify=y_full,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



# Batch iterator

def get_batches(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool = True):
    """
    Yield (X_batch, y_batch) mini-batches.

    Parameters
    ----------
    X          : (N, features)
    y          : (N,)
    batch_size : int
    shuffle    : shuffle indices before each epoch
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, N, batch_size):
        idx = indices[start: start + batch_size]
        yield X[idx], y[idx]


# Class name maps


CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fashion_mnist": [
        "T-shirt/top", "Trouser",  "Pullover", "Dress",    "Coat",
        "Sandal",      "Shirt",    "Sneaker",  "Bag",      "Ankle boot",
    ],
}
