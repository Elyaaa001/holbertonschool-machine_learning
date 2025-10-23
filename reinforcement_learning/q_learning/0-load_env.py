#!/usr/bin/env python3
"""
Load the FrozenLake environment.
"""

from typing import List, Optional
import gymnasium as gym


def load_frozen_lake(
    desc: Optional[List[List[str]]] = None,
    map_name: Optional[str] = None,
    is_slippery: bool = False
):
    """
    Load the pre-made FrozenLake environment from Gymnasium.

    Args:
        desc: Optional custom map description as a list of list of single-char
            strings (e.g., [['S', 'F', ...], ...]). If provided, map_name is
            ignored.
        map_name: Name of a pre-made map (e.g., "4x4", "8x8"). Ignored when
            desc is provided. If both desc and map_name are None, defaults to
            "8x8".
        is_slippery: If True, transitions are stochastic; otherwise
            deterministic.

    Returns:
        The Gymnasium FrozenLake-v1 environment.
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
