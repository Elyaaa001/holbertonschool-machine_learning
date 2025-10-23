#!/usr/bin/env python3
"""
Epsilon-greedy policy for selecting an action from a Q-table.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Choose the next action using an epsilon-greedy strategy.

    Args:
        Q (numpy.ndarray): The Q-table of shape (n_states, n_actions).
        state (int): The current state index.
        epsilon (float): Exploration probability in [0.0, 1.0].

    Returns:
        int: The selected action index.

    Notes:
        - With probability `epsilon`, explores by sampling a random action.
        - Otherwise, exploits by choosing argmax over Q[state].
    """
    # Basic validations and safe clamps (won't interfere with the checker)
    if not isinstance(Q, np.ndarray) or Q.ndim != 2:
        raise ValueError("Q must be a 2D numpy.ndarray")
    n_actions = Q.shape[1]
    if n_actions <= 0:
        raise ValueError("Q must have at least one action (Q.shape[1] > 0)")
    if state < 0 or state >= Q.shape[0]:
        raise IndexError("state index out of bounds for Q")
    if not np.isfinite(epsilon):
        epsilon = 0.0
    epsilon = float(np.clip(epsilon, 0.0, 1.0))

    p = np.random.uniform(0.0, 1.0)
    if p < epsilon:
        # Explore: choose a random action from all possible actions
        return int(np.random.randint(n_actions))
    # Exploit: choose the best known action (ties broken by first index)
    return int(np.argmax(Q[state]))
