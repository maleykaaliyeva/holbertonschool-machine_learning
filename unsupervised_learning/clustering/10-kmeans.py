#!/usr/bin/env python3
"""Module that performs K-means clustering using sklearn."""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.
    """
    kmeans_model = sklearn.cluster.KMeans(
        n_clusters=k,
        n_init='auto'
    )
    kmeans_model.fit(X)

    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
