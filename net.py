import numpy as np
from typing import List

Vector = List[float]

def neural_net(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return _sigmoid(z)

def perceptron(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias
    return _step_function(z)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))

def _step_function(z):
    if z <= 0:
        return 0
    else:
        return 1

inputs = np.array([0.1, 0.2, 0.37])
weights = np.array([0.55, 0.7, 0.21])
bias = 3

print(neural_net(inputs, weights, bias))
c = 100
print(f'sigmoid: {neural_net(inputs, weights*c, bias*c)}')
print(f'perceptron: {perceptron(inputs, weights, bias)}')
