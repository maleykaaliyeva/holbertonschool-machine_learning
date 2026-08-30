#!/usr/bin/env python3
"""
Module to have a trained agent play an episode
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode using pure exploitation

    Args:
        env: the FrozenLakeEnv instance
        Q: a numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards, rendered_outputs
    """
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_reward = 0

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)

        rendered_outputs.append(env.render())
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, rendered_outputs
