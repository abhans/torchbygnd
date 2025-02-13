import numpy as np
from scipy.optimize import minimize

def primal(w, C, xi):
    return 0.5 * np.dot(w, w) + C * np.sum(xi)

def constraints(w, X, y, b, xi):
    return y * (np.dot(w.T, X) + b) + xi - 1

def solve(X):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    # Initialize variables
    w_init = np.zeros(n_features)
    b_init = 0.0
    xi_init = w_init = np.zeros(n_samples)

    bounds = [(None, None) * n_features + (None, None) + (0, None) * n_features]
    
    variables = np.concatenate([])

    constraints = ({'type': 'ineq', 'fun': lambda : })

    pass