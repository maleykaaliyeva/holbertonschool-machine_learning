#!/usr/bin/env python3
"""
Play module for Q-learning
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode exploiting the Q-table

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: float, total rewards accumulated
        rendered_outputs: list of rendered board states
    """
    state, _ = env.reset()
    rendered_outputs = []
    
    # Capture initial state
    rendered_outputs.append(env.render())

    total_rewards = 0

    for step in range(max_steps):
        # Always exploit Q-table
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

        state = next_state

    return total_rewards, rendered_outputs
