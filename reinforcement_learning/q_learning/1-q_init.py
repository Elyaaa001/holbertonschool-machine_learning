#!/usr/bin/env python3
"""
Initialize the Q-table for a FrozenLake environment.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for a given FrozenLake environment.

    Args:
        env: the FrozenLakeEnv instance.

    Returns:
        numpy.ndarray: the Q-table, a matrix of zeros with shape
                       (number of states, number of actions).
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    return Q
