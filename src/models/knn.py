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
        # Memorizzazione dei dati di training
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
