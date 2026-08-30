#!/usr/bin/env python3
"""
Module to train an agent using Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains an agent with Q-learning

    Args:
        env: FrozenLakeEnv instance
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

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Standard Bellman Equation Update
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        # Linear decay per episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards
