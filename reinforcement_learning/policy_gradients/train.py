#!/usr/bin/env python3
"""
Train Monte-Carlo policy gradient model
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Implements full training loop for Monte-Carlo policy gradient

    Args:
        env: initial environment instance
        nb_episodes: number of episodes used for training
        alpha: learning rate
        gamma: discount factor

    Returns:
        scores: list of total scores per episode
    """
    # Initialize random weight matrix (state_dim, action_dim)
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        gradients = []
        rewards = []
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            gradients.append(grad)
            rewards.append(reward)
            state = next_state

        score = sum(rewards)
        scores.append(score)

        # Update weights using discounted rewards
        for t in range(len(rewards)):
            G_t = sum([r * (gamma ** i) for i, r in enumerate(rewards[t:])])
            weight += alpha * gradients[t] * G_t

        print(f"Episode: {episode} Score: {score}")

    return scores
