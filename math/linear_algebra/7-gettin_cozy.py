#!/usr/bin/env python3
"""This module defines a function that concatenates two 2D matrices."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis.
    
    Returns a new matrix, or None if shapes are incompatible.
    """
    if axis == 0:
        # Check if the number of columns is the same
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        # Check if the number of rows is the same
        if len(mat1) != len(mat2):
            return None
        return [mat1[i] + mat2[i] for i in range(len(mat1))]

    return None
