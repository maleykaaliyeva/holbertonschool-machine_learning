#!/usr/bin/env python3
"""
Module to perform Q-learning on a Frozen Lake environment
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning training

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q, total_rewards
    """
    total_rewards = []
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q-value
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

        # Decay epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

    return Q, total_rewards
