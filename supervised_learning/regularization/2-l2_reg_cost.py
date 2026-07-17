#!/usr/bin/env python3
"""L2 Regularization Cost."""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Return a tensor containing the cost for each layer
    including its L2 regularization."""
    l2_losses = model.losses

    # a list of new tensors, each representing a "total cost" per layer.
    total_costs_per_layer = [cost + los for los in l2_losses]

    # Finally, we stack this list of tensors into a single tensor to produce
    # the desired output
    return tf.stack(total_costs_per_layer)
