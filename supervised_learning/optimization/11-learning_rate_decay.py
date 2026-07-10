#!/usr/bin/env python3
""" Learning Rate Decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return the updated value for alpha."""
    alpha = alpha / (1 + decay_rate * (global_step // decay_step))
    return alpha
