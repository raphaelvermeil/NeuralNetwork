# NeuralNetwork

A general-purpose feed-forward neural network implemented from scratch in
[main.py](main.py) using only NumPy. No deep-learning frameworks — forward
propagation, back propagation, and gradient descent are all written out by hand.

The network is **fully generalised**: the input size, number of hidden layers,
size of each hidden layer, and output size are all driven by a handful of
constants. Change them to fit any classification problem — the rest of the code
(initialisation, forward prop, back prop, parameter updates) adapts
automatically. Handwritten-digit recognition (MNIST) is used below only as a
worked example.

## Configuring the network

The architecture is defined by these constants near the top of [main.py](main.py):

```python
input_layer_size        = 784        # number of input features
number_of_hidden_layers = 2          # how many hidden layers
size_of_hidden_layers   = [100, 10]  # one entry per hidden layer
output_layer_size       = 10         # number of output classes
```

Tune these to your problem:

- **`input_layer_size`** — number of features per example.
- **`number_of_hidden_layers`** — network depth.
- **`size_of_hidden_layers`** — a list with one neuron count per hidden layer
  (its length must equal `number_of_hidden_layers`).
- **`output_layer_size`** — number of classes to predict.

For example, a problem with 64 features, three hidden layers, and 5 classes:

```python
input_layer_size        = 64
number_of_hidden_layers = 3
size_of_hidden_layers   = [128, 64, 32]
output_layer_size       = 5
```

### Example: MNIST digits

The defaults shown above are configured for the MNIST dataset: `784` inputs
(28×28 pixels), two hidden layers, and `10` outputs (digits 0–9).

## How it works

| Function | Role |
| --- | --- |
| `init_params()` | Randomly initialises every weight/bias matrix in `[-0.5, 0.5]`. |
| `forward_prop(...)` | Runs input through each layer (ReLU on hidden, softmax on output). |
| `back_prop(...)` | Computes gradients via the chain rule, layer by layer. |
| `update_params(...)` | Applies gradient-descent updates with learning rate `alpha`. |
| `gradient_descent(X, Y, alpha, iterations)` | Training loop; prints accuracy every 10 iterations. |
| `make_predictions(...)` / `test_prediction(index, ...)` | Predict and visualise individual samples. |

- **Hidden layers** use ReLU activation.
- **Output layer** uses softmax to produce class probabilities.
- Weights and biases are stored as Python lists of NumPy matrices, one per layer,
  so the network depth is driven entirely by the constants above.

### Data layout

- `X` — feature matrix where **each column is one example**. The number of rows
  must equal `input_layer_size`.
- `Y` — label row vector, one integer label per example. Labels must range from
  `0` to `output_layer_size - 1`.

## Requirements

```bash
pip install numpy pandas matplotlib
```

> `matplotlib` is only needed for `test_prediction`, which reshapes a sample into
> a 28×28 image. That visualisation is MNIST-specific — adapt or skip it for
> other problems.

## Usage

> **Note:** [main.py](main.py) expects a `data` variable to already exist as a
> 2-D array whose first column is the label and remaining columns are features.
> Loading it is not yet wired up — add a step such as the following before line 6:
>
> ```python
> data = pd.read_csv("your_data.csv").to_numpy()
> ```
>
> For the MNIST example, `train.csv` is available on
> [Kaggle's Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer/data).

Once data loading is in place:

```bash
python main.py
```

This will:

1. Shuffle the data and split it into a 1,000-example dev set and the rest as a
   training set.
2. Scale features to `[0, 1]` by dividing by 255 (suitable for pixel data —
   adjust the normalisation for non-image inputs).
3. Train via `gradient_descent(X_train, Y_train, 0.10, 500)`.
4. Print training accuracy every 10 iterations.

To inspect a single prediction:

```python
test_prediction(0, array_of_weights_matrices, array_of_biases_matrices)
```

## Hyperparameters

| Parameter | Value | Where |
| --- | --- | --- |
| Learning rate (`alpha`) | `0.10` | call to `gradient_descent` |
| Iterations | `500` | call to `gradient_descent` |
| Dev set size | `1000` | `data[0:1000]` split |
