#!/usr/bin/env python3
"""
Epsilon-greedy action selection for a given state.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action.

    Args:
        Q (numpy.ndarray): Q-table.
        state (int): current state.
        epsilon (float): probability of exploring.

    Returns:
        int: the index of the next action.
    """
    p = np.random.uniform(0, 1)

    # Explore: choose random action
    if p < epsilon:
        action = np.random.randint(Q.shape[1])
    # Exploit: choose best known action
    else:
        action = np.argmax(Q[state])

    return action
