import numpy as np

def holdout_split(X, y, test_size=0.3, seed=42):
    """
    Split holdout train/test.
    Restituisce: X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kfold_indices(n_samples, k=5, seed=42):
    """
    Genera indici per K-Fold cross validation.
    Restituisce una lista di tuple (train_idx, test_idx)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    folds = np.array_split(indices, k)
    splits = []

    for i in range(k):
        test_idx = folds[i]
        train_idx = np.hstack(folds[:i] + folds[i+1:])
        splits.append((train_idx, test_idx))

    return splits


def random_subsampling_indices(n_samples, test_size=0.3, r=30, seed=42):
    """
    Genera indici per Random Subsampling (Repeated Holdout).
    Restituisce una lista di tuple (train_idx, test_idx)
    """
    splits = []
    for i in range(r):
        rng = np.random.default_rng(seed + i)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        split = int(n_samples * (1 - test_size))
        train_idx = indices[:split]
        test_idx = indices[split:]

        splits.append((train_idx, test_idx))

    return splits
