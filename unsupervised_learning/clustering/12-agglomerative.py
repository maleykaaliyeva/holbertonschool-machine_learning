#!/usr/bin/env python3
"""Module that performs agglomerative clustering using scipy."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset.
    """
    # Perform hierarchical clustering using Ward's linkage
    hier = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Plot dendrogram with different colors for each cluster based on dist threshold
    scipy.cluster.hierarchy.dendrogram(hier, color_threshold=dist)
    plt.show()

    # Form flat clusters from the hierarchical clustering
    clss = scipy.cluster.hierarchy.fcluster(hier, t=dist, criterion='distance')

    # fcluster returns 1-based cluster indices, convert them to 0-based
    return clss - 1
