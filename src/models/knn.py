import numpy as np
from abc import ABC, abstractmethod

class DistanceStrategy(ABC):
    """
    Classe che definisce una regola comune per tutte le classi di distanza
    Il KNN può usare la distanza Euclidea o qualsiasi altra
    distanza senza dover cambiare il suo codice interno
    """
    @abstractmethod
    def compute(self, point1, point2):
        pass

class EuclideanDistance(DistanceStrategy):
    """
    Implementazione della distanza Euclidea.
    """
    def compute(self, point1, point2):
        """
        Parametri:
        point1 -> conterrà il singolo campione di test
        point2 -> conterrà tutti i campioni del training set
        ----
        Valore restituito:
        Restituirà in un array tutte le distanze del singolo campione di test
        da tutti i campioni di training sfruttando il broadcasting di numpy
        """
        return np.linalg.norm(point1 - point2, axis=1)

class KNN:
    """
    Classe principale che implementa l'algoritmo KNN
    Utilizza la modularizzazione tramite classi e il pattern strategy.
    """
    def __init__(self, k: int, distance_strategy: DistanceStrategy):
        """
        Parametri costruttore:
        k -> inizializza il numero di vicini da considerare
        distance_strategy -> inizializza il tipo di distanza da utilizzare
        """
        if k < 1:
            raise ValueError("Il valore di k deve essere almeno 1.")
        self.k = k
        self.distance_strategy = distance_strategy
        self.x_train = None
        self.y_train = None
        self.x_test = None

    def fit(self, x_train, y_train):
        """
        Parametri:
        x_train -> matrice delle features del training set
        y_train -> array delle classi/etichette corrispondenti
        """
        # Controllo di sicurezza finale
        n_samples = x_train.shape[0]
        if self.k > n_samples:
            raise ValueError(
                f"k={self.k} è maggiore del numero di campioni disponibili ({n_samples})."
            )
        # Memorizzazione dei dati di training
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

    def predict(self, x_test):
        """
        Parametri:
        x_test -> matrice dei campioni da testare
        ----
        Valore restituito:
        y_pred -> l'array delle predizioni che conterrà le classi/etichette predette per ogni campione di test
        """
        # Memorizzazione dei dati di test e allocazione dell'array delle predizioni
        self.x_test = np.array(x_test)
        y_pred = np.zeros(self.x_test.shape[0], dtype=int)  # Shape: (numero campioni test)

        # Matrice per memorizzare tutte le distanze tra ogni campione di test e ogni campione di training
        distances_tot = np.zeros((self.x_test.shape[0], self.x_train.shape[0]))  # Shape: (numero campioni test, numero campioni training)

        # Prima parte: calcolo delle distanze
        for i, test in enumerate(x_test):
            # Vengono calcolate le distanze tra ogni campione del test set e ogni campione del trainng set
            distance = self.distance_strategy.compute(test, self.x_train)
            distances_tot[i] = distance

        # Seconda parte: ricerca dei k-vicini e assegnazione della classe/etichetta al campione di test
        for i, dis in enumerate(distances_tot):
            index_sorted = np.argsort(dis)  # Memorizzazione degli indici delle distanze in ordine crescente
            k_index_nearest = index_sorted[:self.k]  # Selezione dei k-indici più vicini
            k_classes_nearest = self.y_train[k_index_nearest]  # Vengono recuperate le classi/etichette relative a quegli indici

            # In values ci sono i valori delle classi/etichette
            # In counts ci sono le occorrenze di ogni classe/etichetta trovata tra i vicini
            values, counts = np.unique(k_classes_nearest, return_counts=True)

            # Gestione del pareggio
            # Se più classi/etichette hanno lo stesso numero massimo di occorrenze avremo che la somma sarà > 1
            if np.sum(counts == np.max(counts)) > 1:
                classes = values[counts == np.max(counts)]  # Seleziona solo le classi/etichette che hanno pareggiato
                y_pred[i] = int(np.random.choice(classes))  # Viene scelta una classe/etichetta a caso tra quelle che pareggiano
            else:
                # Altrimenti viene assegnata la classe/etichetta con il numero massimo di occorrenze
                y_pred[i] = int(values[np.argmax(counts)])

        return y_pred
        
    def predict_scores(self, x_test, positive_label=1):
        """
        Calcola uno score continuo per ogni campione di test.
        Lo score è definito come la frazione di vicini appartenenti
        alla classe positiva (maligno).

        Parametri:
        x_test -> matrice dei campioni da testare
        positive_label -> label della classe positiva 

        Restituisce:
        scores -> array di float in [0,1]
        """
        x_test = np.array(x_test)
        scores = np.zeros(x_test.shape[0])

        # Calcolo distanze come in predict
        distances_tot = np.zeros((x_test.shape[0], self.x_train.shape[0]))

        for i, test in enumerate(x_test):
            distance = self.distance_strategy.compute(test, self.x_train)
            distances_tot[i] = distance

        # Calcolo score per ogni campione
        for i, dis in enumerate(distances_tot):
            index_sorted = np.argsort(dis)
            k_index_nearest = index_sorted[:self.k]
            k_classes_nearest = self.y_train[k_index_nearest]

            # Frazione di vicini positivi
            scores[i] = np.sum(k_classes_nearest == positive_label) / self.k

        return scores
