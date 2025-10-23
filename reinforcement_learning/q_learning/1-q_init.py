#!/usr/bin/env python3
"""Load the FrozenLake environment from Gymnasium."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load the pre-made FrozenLake environment from Gymnasium.

    Args:
        desc: either None or a list of lists containing a custom
              description of the map to load for the environment.
        map_name: either None or a string containing the pre-made
                  map to load. If both desc and map_name are None,
                  a randomly generated 8x8 map is used.
        is_slippery: boolean to determine if the ice is slippery.

    Returns:
        The environment.
    """
    if desc is None and map_name is None:
        map_name = "8x8"

    if desc is not None:
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            is_slippery=is_slippery
        )
    else:
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_name,
            is_slippery=is_slippery
        )
    return env
