from src.evaluation.metrics import evaluate_classification

def test_metrics_perfect_classifier():
    y_true = [2, 2, 4, 4]
    y_pred = [2, 2, 4, 4]

    results = evaluate_classification(y_true, y_pred)

    assert results["accuracy"] == 1.0
    assert results["error_rate"] == 0.0
    assert results["sensitivity"] == 1.0
    assert results["specificity"] == 1.0
    assert results["gmean"] == 1.0
