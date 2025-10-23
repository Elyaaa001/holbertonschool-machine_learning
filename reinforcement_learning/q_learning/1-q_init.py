#!/usr/bin/env python3
import numpy as np

def q_init(env):
    """
    Initialize the Q-table.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        Q-table as a numpy.ndarray of zeros
    """
    # Number of states and actions in the environment
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize Q-table with zeros
    Q = np.zeros((n_states, n_actions))

    return Q
