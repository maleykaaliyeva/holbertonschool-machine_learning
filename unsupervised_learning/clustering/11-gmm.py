#!/usr/bin/env python3
"""Module that calculates a GMM from a dataset using sklearn."""
import sklearn.mixture


def gmm(X, k):
    """
    Calculates a Gaussian Mixture Model from a dataset.
    """
    gmm_model = sklearn.mixture.GaussianMixture(
        n_components=k
    )
    gmm_model.fit(X)

    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariances_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
