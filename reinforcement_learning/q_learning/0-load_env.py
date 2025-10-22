#!/usr/bin/env python3
"""
Task 0: Load the FrozenLake environment.

Provides a helper to create Gymnasium's FrozenLake-v1 with either a custom
description (desc) or a pre-made map (map_name), and with optional stochastic
transitions (is_slippery).
"""

from typing import List, Optional
import gymnasium as gym


def load_frozen_lake(
    desc: Optional[List[List[str]]] = None,
    map_name: Optional[str] = None,
    is_slippery: bool = False
):
    """
    Create and return a FrozenLake-v1 environment.

    Args:
        desc: Optional custom map description as a list of list of single-char
            strings (e.g., [['S', 'F', ...], ...]). If provided, map_name is
            ignored.
        map_name: Name of a pre-made map (e.g., "4x4", "8x8"). Ignored when
            desc is provided.
        is_slippery: If True, transitions are stochastic; otherwise deterministic.

    Returns:
        The created Gymnasium environment.
    """
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
