# Implementation-of-neural-network-from-scratch


This repository contains a simple feedforward neural network implemented from scratch in Python, without the use of deep learning libraries. The implementation focuses on basic neural network operations such as the forward pass, backpropagation, and gradient descent for training.

# Objective

The goal of this project is to build a neural network to predict target values (regression task) based on input features, demonstrating how neural networks function at a fundamental level.

# Problem Definition

# Dataset
Input Features: A small custom dataset with two features: `temperature` and `humidity`.
Targets: Normalized continuous values representing performance or ratings.

Example Data:

X = [[23.5, 60.0], [25.0, 65.0], [20.0, 50.0], [30.0, 80.0]]  # Input
y = [[0.8], [0.9], [0.7], [0.95]]                             # Target


# Task
The task is a regression problem where the network predicts continuous target values based on input features.

# Methodology

# Neural Network Architecture
Input Layer:2 neurons (representing temperature and humidity).
Hidden Layer: 4 neurons with Sigmoid activation function.
Output Layer: 1 neuron with Sigmoid activation function.

# Forward Pass
The forward pass involves propagating input data through the network:
1. Weighted sums are computed for each layer.
2. The Sigmoid activation function is applied to introduce non-linearity.
3. Outputs are generated in the final layer.

# Backpropagation
Error is propagated backward through the network to adjust weights and biases:
1. Compute the loss using Mean Squared Error (MSE).
2. Compute gradients using the derivative of the Sigmoid function.
3. Update weights and biases using Gradient Descent.

# Loss Function
Mean Squared Error (MSE):
•	MSE=1/N∑(ytrue−ypred)^2

# Optimization
Gradient Descent:Updates weights and biases iteratively to minimize the loss function.

# Usage

# Requirements
- Python 3
- NumPy library

# Running the Code
1. Clone the repository:
   
   git clone https://github.com/yourusername/neural-network-from-scratch.git
   cd neural-network-from-scratch
   
2. Run the script:
 
   python neural_network.py
  
3. Enter custom inputs for predictions (optional):
   
   Enter input (e.g., '23.5 60.0') or type 'exit' to quit:
   

# Example Output

Epoch 0 - Loss: 0.12345
Epoch 1000 - Loss: 0.01023

Predictions after training:
[[0.789]
 [0.900]
 [0.702]
 [0.948]]


# Enhancements
Consider extending the project with:
- Larger datasets (e.g., Iris, MNIST).
- Additional activation functions (e.g., ReLU, Tanh).
- More advanced optimizers (e.g., Adam).
- Integration with real-world data for classification or regression tasks.

## Contributing
Feel free to fork the repository, create a new branch, and submit a pull request with improvements or bug fixes.

## License
This project is open-source and available under the [MIT License](LICENSE).


