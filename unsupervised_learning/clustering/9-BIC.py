#!/usr/bin/env python3
"""module BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for GMM
    using the Bayesian Information criterion"""
    n, d = X.shape
    if kmax is None:
        kmax = n

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)

    best_bic = float('inf')
    best_k = None
    best_result = None

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None or m is None or S is None or log_likelihood is None:
            continue
        p = (k - 1) + k * d + k * d * (d + 1) // 2

        bic = p * np.log(n) - 2 * log_likelihood

        bic = p * np.log(n) - 2 * log_likelihood

        l[k - kmin] = log_likelihood
        b[k - kmin] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

        if best_k is None:
            return None, None, None, None
        return best_k, best_result, l, b
