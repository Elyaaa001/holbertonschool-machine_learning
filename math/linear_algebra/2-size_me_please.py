#!/usr/bin/env python3
"""This module contains a function that returns the shape of a matrix."""


def matrix_shape(matrix):
    """Returns the shape of a matrix as a list of integers."""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
