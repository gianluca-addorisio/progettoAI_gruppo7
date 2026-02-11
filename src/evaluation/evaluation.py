import numpy as np

from .splits import (
    holdout_split_by_id,
    kfold_indices,
    random_subsampling_indices
)
from .metrics import build_metric_strategies, evaluate_with_strategies


def evaluate_model_holdout(model, X, y, ids, metric_names, test_size=0.3, seed=42, positive_label=1):
    """
    Valutazione con holdout split by ID.
    """
    X_tr, X_te, y_tr, y_te = holdout_split_by_id(
        X, y, ids, test_size=test_size, seed=seed
    )

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).astype(int)
    
    #Score continuo per AUC 
    metrics = build_metric_strategies(metric_names)
    need_score = any(m.requires_score for m in metrics)
    y_score = model.predict_scores(X_te, positive_label=positive_label) if need_score else None
    return evaluate_with_strategies(y_te, y_pred, y_score, metrics, positive_label=positive_label)


def evaluate_model_kfold(model, X, y, ids, metric_names, k=5, seed=42, positive_label=1):
    """
    Valutazione con k-fold cross-validation (split per ID).
    """
    results = []
    metrics = build_metric_strategies(metric_names)
    need_score = any(m.requires_score for m in metrics)

    unique_ids = np.unique(ids)
    splits = kfold_indices(len(unique_ids), k=k, seed=seed)

    for train_idx, test_idx in splits:
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        train_mask = np.isin(ids, train_ids)
        test_mask = np.isin(ids, test_ids)

        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask]).astype(int)
        y_score = model.predict_scores(X[test_mask]), positive_label=positive_label) if need_score else None
        
        results.append(
            evaluate_with_strategies(y[test_mask], y_pred, y_score, metrics, positive_label=positive_label))
    return aggregate_results(results)


def evaluate_model_subsampling(model, X, y, ids, metric_names, r=30, test_size=0.3, see=42, positive_label=1):
    """
    Valutazione con random subsampling (repeated holdout).
    """
    results = []
    metrics = build_metrics_strategies(metric_names)
    need_score = any(m.requires_score for m in metrics)
    
    unique_ids = np.unique(ids)
    splits = random_subsampling_indices(
        len(unique_ids), test_size=test_size, r=r, seed=seed
    )
    
    for train_idx, test_idx in splits:
        train_ids = unique_ids[train_idx]
        test_ids = unique_ids[test_idx]

        train_mask = np.isin(ids, train_ids)
        test_mask = np.isin(ids, test_ids)

        model.fit(X[train_mask], y[train_mask])
        y_pred = model.predict(X[test_mask]).astype(int)
        y_score = model.predict_scores(X[test_mask], positive_label=positive_label) if need_score else None
        
        results.append(
            evaluate_with_strategies(y[test_mask], y_pred, y_score, metrics, positive_label=positive_label))

    return aggregate_results(results)


def aggregate_results(results):
    """
    Calcola media e deviazione standard delle metriche.
    """
    aggregated = {}

    for key in results[0]:
        values = [r[key] for r in results]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    return aggregated
