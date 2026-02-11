import unittest
import numpy as np
from src.models.knn import KNN, EuclideanDistance


class TestKNN(unittest.TestCase):

    def setUp(self):
        """Inizializzazione dei dati per i test"""
        self.distance_metric = EuclideanDistance()

        # Dataset per il test della distanza
        np.random.seed(42)
        self.X_train_dist = np.random.randint(0, 11, size=(8, 3))
        self.X_test_dist = np.random.randint(0, 11, size=(2, 3))

        # Dataset per il test funzionale del KNN
        self.X_train_knn = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        self.y_train_knn = np.array([0, 0, 0, 1])
        self.X_test_knn = np.array([[1, 1]])

    def test_euclidean_distance(self):
        """Verifica che la distanza calcolata sia matematicamente corretta"""
        for i in self.X_test_dist:
            distances = self.distance_metric.compute(i, self.X_train_dist)

            for k, p in enumerate(self.X_train_dist):
                expected = np.sqrt(np.sum((i - p) ** 2))
                self.assertEqual(distances[k], expected, msg=f"Distanza non coincidente all'indice {k}")

    def test_knn_prediction(self):
        """Verifica che il KNN restituisca l'etichetta corretta (maggioranza)"""
        k_neighbors = 2
        knn = KNN(k_neighbors, self.distance_metric)
        knn.fit(self.X_train_knn, self.y_train_knn)

        prediction = knn.predict(self.X_test_knn)

        # Verifica che la predizione sia quella attesa
        self.assertEqual(prediction[0], 0, "Il KNN doveva scegliere la classe 0!")

    def test_k_equals_one(self):
        """Verifica che con K=1 il punto prenda l'etichetta del vicino più vicino"""
        knn = KNN(1, self.distance_metric)
        knn.fit(self.X_train_knn, self.y_train_knn)
        # Testiamo un punto identico al primo campione del train
        prediction = knn.predict(np.array([self.X_train_knn[0]]))
        self.assertEqual(prediction[0], self.y_train_knn[0])

if __name__ == '__main__':
    unittest.main()