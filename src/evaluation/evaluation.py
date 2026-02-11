import numpy as np

from .splits import (
    holdout_split_by_id,
    kfold_indices,
    random_subsampling_indices
)
from .metrics import evaluate_classification, auc_score

def evaluate_model_holdout(model, X, y, ids, test_size=0.3, seed=42):
    """
    Valutazione con holdout split by ID.
    """
    X_tr, X_te, y_tr, y_te = holdout_split_by_id(
        X, y, ids, test_size=test_size, seed=seed
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_score = model.predict_scores(X_te)
    
    return evaluate_classification(y_te, y_pred, y_score)


def evaluate_model_kfold(model, X, y, ids, k=5, seed=42):
    """
    Valutazione con k-fold cross-validation (split per ID).
    """
    results = []

    splits = kfold_indices(len(np.unique(ids)), k=k, seed=seed)
    unique_ids = np.unique(ids)

    for train_idx, test_idx in splits:
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        train_mask = np.isin(ids, train_ids)
        test_mask = np.isin(ids, test_ids)

        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        y_score = model.predict_scores(X[test_mask])
        
        results.append(
            evaluate_classification(y[test_mask], y_pred, y_score))
    return aggregate_results(results)


def evaluate_model_subsampling(model, X, y, ids, r=30, test_size=0.3, seed=42):
    """
    Valutazione con random subsampling (repeated holdout).
    """
    results = []
    splits = random_subsampling_indices(
        len(np.unique(ids)), test_size=test_size, r=r, seed=seed
    )
    unique_ids = np.unique(ids)

    for train_idx, test_idx in splits:
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        train_mask = np.isin(ids, train_ids)
        test_mask = np.isin(ids, test_ids)

        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask])
        y_score = model.predict_scores(X[test_mask])
        
        results.append(
            evaluate_classification(y[test_mask], y_pred, y_score))

    return aggregate_results(results)


def aggregate_results(results):
    """
    Calcola media e deviazione standard delle metriche.
    """
    aggregated = {}

    for key in results[0]:
        values = [r[key] for r in results]
        aggregated[key] = {
            "values": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return aggregated
