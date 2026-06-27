import numpy as np

def standardise(X, eps = 1e-8):
    X = np.asarray(X, dtype = float)

    if  X.ndim != 2:
        raise ValueError("features must have shape (T, D).")

    if not np.all(np.isfinite(X)):
        raise ValueError("features contain NaN or infinite values.")

    X_means = X.mean(axis = 0)
    X_stds = X.std(axis = 0)

    safe_stds = np.maximum(X_stds, eps)

    Z = (X - X_means) / safe_stds

    return Z, X_means, safe_stds