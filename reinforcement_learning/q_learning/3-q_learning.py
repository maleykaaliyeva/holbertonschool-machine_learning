#!/usr/bin/env python3
"""
Q-learning implementation
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning on FrozenLake environment

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: updated Q-table
        total_rewards: list containing total rewards per episode
    """
    total_rewards = []
    initial_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update reward to -1 if the agent falls in a hole
            if terminated and reward == 0:
                reward = -1

            # Q-learning update equation
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state

        total_rewards.append(episode_reward)

        # Exponential decay of epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)

    return Q, total_rewards
