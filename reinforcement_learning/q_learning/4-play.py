#!/usr/bin/env python3
"""
Module to have a trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        total_reward, rendered_outputs
    """
    # Reset the environment and capture the initial board state
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        # Exploit the Q-table (Greedy policy)
        action = np.argmax(Q[state])

        # Apply the action
        state, reward, terminated, truncated, _ = env.step(action)

        # Capture the current board state as a string
        rendered_outputs.append(env.render())

        total_reward += reward

        # End episode if agent reached goal, fell in hole, or timed out
        if terminated or truncated:
            break

    return total_reward, rendered_outputs
