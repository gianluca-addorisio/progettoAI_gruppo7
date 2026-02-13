import unittest
import numpy as np
from src.evaluation.splits import holdout_split_by_id, kfold_indices, random_subsampling_indices
from src.evaluation.metrics import evaluate_classification, auc_score, roc_curve_binary

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

    def test_kfold_indices(self):
        """Verifica che K-Fold crei il numero corretto di fold e non ci siano leakage."""
        n_samples = 12
        k = 3
        splits = kfold_indices(n_samples, k=k)

        self.assertEqual(len(splits), k)
        
        for train_idx, test_idx in splits:
            # Verifica che non ci siano indici duplicati tra train e test
            intersection = np.intersect1d(train_idx, test_idx)
            self.assertEqual(len(intersection), 0)
            # Verifica che la somma degli indici corrisponda al totale
            self.assertEqual(len(train_idx) + len(test_idx), n_samples)

    def test_random_subsampling_indices(self):
        """Verifica le dimensioni e la riproducibilità del Random Subsampling."""
        n_samples = 100
        test_size = 0.2
        r = 5
        seed = 10
        
        splits = random_subsampling_indices(n_samples, test_size=test_size, r=r, seed=seed)
        
        self.assertEqual(len(splits), r)
        
        for train_idx, test_idx in splits:
            # Con 100 campioni e test_size 0.2, test deve essere 20 e train 80
            self.assertEqual(len(test_idx), 20)
            self.assertEqual(len(train_idx), 80)
            
    def test_metrics_perfect_classifier(self):
        """Verifica che le metriche siano corrette per un classificatore perfetto"""
        y_true = [0, 0, 1, 1]  # Invece di 2 e 4
        y_pred = [0, 0, 1, 1]
        y_score = np.array([0.1, 0.2, 0.9, 0.8])

        results = evaluate_classification(y_true, y_pred)

        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["error_rate"], 0.0)
        self.assertEqual(results["sensitivity"], 1.0)
        self.assertEqual(results["specificity"], 1.0)
        self.assertEqual(results["gmean"], 1.0)

        # Test AUC
        auc = auc_score(y_true, y_score)
        self.assertEqual(auc, 1.0, "L'AUC di un classificatore perfetto deve essere 1.0")

    def test_roc_curve_logic(self):
        """Verifica la generazione della curva ROC"""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        
        fpr, tpr, thresholds = roc_curve_binary(y_true, y_score)
        
        # La curva ROC deve sempre iniziare a (0,0) e finire a (1,1)
        self.assertEqual(fpr[0], 0.0)
        self.assertEqual(tpr[0], 0.0)
        self.assertEqual(fpr[-1], 1.0)
        self.assertEqual(tpr[-1], 1.0)
        
if __name__ == '__main__':
    unittest.main()
