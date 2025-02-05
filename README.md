# Neural Network Library

## Overview 🚀

This **Neural Network Library** was built entirely from scratch and represents my first serious project in machine learning. The goal of this project was to implement the core components of a neural network, including forward propagation, backpropagation, weight updates, and activation functions, without relying on any pre-existing machine learning libraries.

This project was both challenging and highly rewarding, offering valuable insights into the mathematics and algorithms behind neural networks. It significantly enhanced my understanding of **multivariate calculus**, **linear algebra**, and **backpropagation**. 💡

## Challenges 🧠

The hardest part of the project was **backpropagation**. Understanding the gradients and their propagation through the network required me to delve deep into the following mathematical concepts:

- **Multivariate calculus**: for calculating partial derivatives and understanding gradient descent.
- **Linear algebra**: for handling matrix operations involved in the network's weight updates.

These principles were essential for deriving and implementing the **backpropagation** algorithm and the correct gradient updates for weights and biases in the network. This made the project highly complex but also deeply educational.

## Project Features ✨

- **Forward Propagation**: Compute the output of the network by propagating the input through each layer.
- **Backpropagation**: Update weights using the gradients derived via the chain rule.
- **Activation Functions**: Implementations of **ReLU** and **Sigmoid**.
- **Loss Functions**: Includes a **Mean Squared Error (MSE)** loss function for training.

## Prerequisites 📚

This project requires a solid understanding of the following concepts:

- **Multivariate calculus**: Used for deriving gradients and optimizing the network.
- **Linear algebra**: Used for matrix manipulations and operations that drive neural network training.
- **Git**: Version control for managing changes and sharing code through GitHub.

If you're new to neural networks or these mathematical concepts, I highly recommend checking out 3b1b's **DL3** and **DL4** videos on backpropagation, which provided me with a deep understanding of these topics.

## Usage 🚀

To use this neural network library, you'll need to have **Python 3.x** and **NumPy** installed.

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/neural-network-library.git
cd neural-network-library
```

### 2 Example Usage:

Here is a simple example of how to use the library to train a neural network:

```python
import numpy as np
from neural_network import Model, Layer
from activationFuncs import ReLU, Sigmoid
from lossFuncs import MSE

# Define the network architecture
layer1 = Layer(inputs=2, outputs=3, wantBias=True, activationFunc=ReLU())
layer2 = Layer(inputs=3, outputs=1, wantBias=True, activationFunc=Sigmoid())

# Initialize the model
model = Model(layer1, layer2, loss=MSE())

# Define some sample data (XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
for epoch in range(10000):
    output = model.run(X)
    model.back(X, y)

# Test the model
predictions = model.run(X)
print(predictions)
```

## 3. Training the model:

- Define the layers by specifying the number of inputs, outputs, and activation functions.
- Use the Model class to encapsulate the layers and the loss function.
- The back method performs backpropagation to update the weights.
- The run method performs forward propagation to calculate the network’s output.

## Files and Folder Structure 📂

- `neural_network.py`: Contains the `Model` class, `Layer` class, and forward/backpropagation logic.
- `activationFuncs.py`: Contains the activation functions (ReLU, Sigmoid).
- `lossFuncs.py`: Contains the loss functions (MSE).
- `README.md`: This file you're currently reading.

## Key Concepts Explained 🔍

### Backpropagation:

The backpropagation algorithm is at the core of training a neural network. It updates the weights based on the error (calculated via the loss function) and propagates the error backwards through the network. The gradients of the loss with respect to the weights are calculated using the chain rule from calculus, which allows us to update the weights in the direction that minimizes the error.

### Forward Propagation:

Forward propagation is the process of passing the input through the layers of the network to produce an output. The output of each layer is passed through an activation function to introduce non-linearity.

## How to Contribute 🌱

If you want to contribute to the development of this project, feel free to:

- Fork the repository and submit pull requests with improvements.
- Open issues for bugs or feature requests.

## Conclusion 🌟

This project has been an exciting and challenging experience. It reinforced the importance of having a strong foundation in mathematics and algorithms when building machine learning models. It also taught me the power of Git for managing my projects and sharing my work.

I look forward to continuing the development of this project and expanding its capabilities. Stay tuned for future updates! 💥#

```
Let me know if you'd like me to tweak anything else! 😁
```
