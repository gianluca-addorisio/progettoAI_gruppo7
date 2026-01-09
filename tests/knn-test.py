from src.models.knn import KNN, EuclideanDistance
import numpy as np

# generazione dataset di prova (10 campioni, 3 features)
np.random.seed(42)
X_train = np.random.randint(0, 11, size=(8, 3))  # valori interi da 1 a 10
X_test = np.random.randint(0, 11, size=(2, 3))
y_train = np.random.choice([2, 4], size=8)

# controllo se il metodo compute restituisce le giuste distanze
e_distance = EuclideanDistance()
for i in X_test:
    distance = e_distance.compute(i, X_train) # array distanze tra il campione di test e tuitti i campioni di training
    for k, p in enumerate(X_train):
        control = np.sqrt(np.sum((i - p) ** 2))
        assert control == distance[k], "Errore: distanza non coincidente con il controllo!"

# controllo se il knn restituisce l'etichetta giusta
X_train_debug = np.array([
    [0, 0], [1, 0], [0, 1],  # molto vicini
    [10, 10]                 # molto lontano
])
y_train_debug = np.array([2, 2, 2, 4])

# punto di test proprio in mezzo ai primi tre
X_test_debug = np.array([[1, 1]])

knn_debug = KNN(2, EuclideanDistance())
knn_debug.fit(X_train_debug, y_train_debug)

# predizione
Y_pred_debug = knn_debug.predict(X_test_debug)
print(f"Predizione: {Y_pred_debug}")
assert Y_pred_debug[0] == 2, "Errore: il KNN doveva scegliere la classe 2!"
