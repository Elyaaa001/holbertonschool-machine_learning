#!/usr/bin/env python3
def load_frozen_lake(desc=None, map_name='4x4', is_slippery=True):
    try:
        import gymnasium as gym
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="ansi",
        )
    except Exception:
        import gym
        env = gym.make(
            "FrozenLake-v1",
            desc=desc,
            map_name=map_name,
            is_slippery=is_slippery,
            render_mode="ansi",
        )
    return env
