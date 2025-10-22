#!/usr/bin/env python3
""" Task 0: 0. Load the FrozenLake environment """
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLakeEnv from Gymnasium.

    Args:
        desc (list[list[str]] | None): custom map description.
        map_name (str | None): pre-made map name (e.g., "4x4", "8x8").
        is_slippery (bool): whether transitions are stochastic.

    Returns:
        gym.Env: the FrozenLake environment.
    """
    # Use the current Gymnasium ID
    if desc is not None:
        env = gym.make("FrozenLake-v1", 
                       desc=desc, is_slippery=is_slippery)
    else:
        env = gym.make("FrozenLake-v1", 
                       map_name=map_name, is_slippery=is_slippery)
    return env
