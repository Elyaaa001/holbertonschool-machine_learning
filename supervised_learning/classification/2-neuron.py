#!/usr/bin/env python3
"""
Implementing the Neuron class for binary classification tasks.

"""
import numpy as np


class Neuron:
    """
    Implements a single neuron for binary classification.
    """

    def __init__(self, nx):
        """
        Initializes a Neuron object.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        Performs forward propagation using the sigmoid activation function.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: The activated output.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    @property
    def W(self):
        """
        Getter method for the weights of the neuron.

        Returns:
            numpy.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias of the neuron.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output of the neuron.

        Returns:
            float: The activated output of the neuron.
        """
        return self.__A
