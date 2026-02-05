import numpy as np
from src.evaluation.splits import holdout_split_by_id

def test_holdout_split_by_id_no_leakage():
    # 6 campioni, 3 ID distinti
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([2, 2, 4, 4, 2, 4])
    ids = np.array([1, 1, 2, 2, 3, 3])

    X_tr, X_te, y_tr, y_te = holdout_split_by_id(X, y, ids, test_size=0.33, seed=0)

    # recuperiamo gli ID di train e test
    train_ids = set(ids[np.isin(X, X_tr).all(axis=1)])
    test_ids = set(ids[np.isin(X, X_te).all(axis=1)])

    # nessun ID deve comparire in entrambi
    assert train_ids.isdisjoint(test_ids)
