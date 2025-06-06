#!/usr/bin/env python3
"""
This module defines the NeuralNetwork class for binary classification
with one hidden layer.
"""
import numpy as np


class NeuralNetwork:
    """
    Define a neural network with one hidden layer performing binary
    classification.
    """

    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork class.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for private attribute W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for private attribute b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for private attribute A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for private attribute W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for private attribute b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for private attribute A2."""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network using a
        sigmoid activation function.
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))  # Sigmoid hidden layer.
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))  # Sigmoid output layer.
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                 * np.log(1.0000001 - A))
        return cost
