# DA6401 – Assignment 1: Multi-Layer Perceptron for Image Classification

## Links
- **W&B Report**: `<ADD YOUR W&B REPORT LINK HERE>`
- **GitHub Repository**:(https://github.com/Sandipanme25m103/Assignment1.git)

---

## Project Structure

```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py          # ReLU, Sigmoid, Tanh, Softmax + derivatives
│   │   ├── neural_layer.py         # NeuralLayer: forward, backward, grad_W, grad_b
│   │   ├── neural_network.py       # NeuralNetwork: forward, backward, get/set_weights
│   │   ├── objective_functions.py  # Cross-Entropy, MSE + gradients
│   │   └── optimizers.py           # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py          # load_dataset, get_batches
│   ├── train.py                    # Main training script (argparse CLI)
│   ├── inference.py                # Inference & evaluation script
│   ├── gradient_check.py           # Numerical gradient verification
│   ├── wandb_sweep.py              # W&B hyperparameter sweep (100+ runs)
│   ├── best_model.npy              # Best saved weights (by test F1)
│   └── best_config.json            # Best hyperparameter configuration
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train.py \
    -d fashion_mnist \
    -e 15 \
    -b 64 \
    -l cross_entropy \
    -o rmsprop \
    -lr 0.001 \
    -wd 0.0001 \
    -nhl 3 \
    -sz 128 \
    -a relu \
    -w_i xavier \
    -w_p YOUR_WANDB_PROJECT
```

### Inference

```bash
python src/inference.py \
    -d fashion_mnist \
    --model_path best_model.npy \
    --split test
```

### Gradient Check

```bash
python src/gradient_check.py
```

### Hyperparameter Sweep

```bash
python src/wandb_sweep.py --project YOUR_WANDB_PROJECT --count 100
```

---

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `-d` / `--dataset` | `fashion_mnist` | `mnist` or `fashion_mnist` |
| `-e` / `--epochs` | `15` | Training epochs |
| `-b` / `--batch_size` | `64` | Mini-batch size |
| `-lr` / `--learning_rate` | `0.001` | Learning rate |
| `-o` / `--optimizer` | `rmsprop` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-nhl` / `--num_layers` | `3` | Hidden layers |
| `-sz` / `--hidden_size` | `128` | Neurons per hidden layer |
| `-a` / `--activation` | `relu` | `relu`, `sigmoid`, `tanh` |
| `-l` / `--loss` | `cross_entropy` | `cross_entropy`, `mse` |
| `-w_i` / `--weight_init` | `xavier` | `random`, `xavier`, `zeros` |
| `-wd` / `--weight_decay` | `0.0` | L2 regularisation |
| `-w_p` / `--wandb_project` | `da6401-assignment1` | W&B project |
| `--model_save_path` | `best_model.npy` | Relative path to save weights |

---

## Key Design Points

- **Output layer returns logits** — no softmax applied (autograder requirement)
- **`backward()` returns grad_W, grad_b** — index 0 = last layer, as required by skeleton
- **`NeuralLayer.grad_W` and `NeuralLayer.grad_b`** — stored after every `backward()` call
- **`get_weights()` / `set_weights()`** — keyed as `W0, b0, W1, b1, ...`
- Gradient check passes at **~1e-11 absolute error** (requirement: 1e-7)
