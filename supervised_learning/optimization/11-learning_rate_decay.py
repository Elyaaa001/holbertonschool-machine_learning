#!/usr/bin/env python3
"""
    A function that updates the learning rate using inverse time decay in numpy
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
        Updates the learning rate using inverse time decay in numpy.

        Args:
            alpha (float): the original learning rate.
            decay_rate (float): the weight used to determine the rate at which
                alpha will decay.
            global_step (int): the number of passes of gradient descent that
                have elapsed.
            decay_step (int): the number of passes of gradient descent that
                should occur before alpha is decayed further.

        Returns:
            The updated value for alpha.
    """
    new_alpha = alpha / (1 + decay_rate * (global_step // decay_step))

    return new_alpha
