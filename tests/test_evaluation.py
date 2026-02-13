import unittest
import numpy as np
from src.evaluation.splits import holdout_split_by_id
from src.evaluation.metrics import evaluate_classification

class TestEvaluation(unittest.TestCase):

    def test_holdout_split_by_id_no_leakage(self):
        # 6 campioni, 3 ID distinti
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([2, 2, 4, 4, 2, 4])
        ids = np.array([1, 1, 2, 2, 3, 3])

        # Esecuzione split
        X_tr, X_te, y_tr, y_te = holdout_split_by_id(X, y, ids, test_size=0.33, seed=0)

        # Recuperiamo gli ID di train e test (usando gli indici per precisione)
        train_indices = [np.where((X == row).all(axis=1))[0][0] for row in X_tr]
        test_indices = [np.where((X == row).all(axis=1))[0][0] for row in X_te]
        
        train_ids = set(ids[train_indices])
        test_ids = set(ids[test_indices])

        # Verifica: l'intersezione degli ID deve essere un set vuoto
        intersection = train_ids.intersection(test_ids)
        self.assertEqual(len(intersection), 0, f"Leakage rilevato per gli ID: {intersection}")

    def test_metrics_perfect_classifier(self):
        """Verifica che le metriche siano corrette per un classificatore perfetto"""
        y_true = [2, 2, 4, 4]
        y_pred = [2, 2, 4, 4]

        results = evaluate_classification(y_true, y_pred)

        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["error_rate"], 0.0)
        self.assertEqual(results["sensitivity"], 1.0)
        self.assertEqual(results["specificity"], 1.0)
        self.assertEqual(results["gmean"], 1.0)

if __name__ == '__main__':
    unittest.main()
