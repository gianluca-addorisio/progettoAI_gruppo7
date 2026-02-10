import pandas as pd
import numpy as np


class LabelEncoder:
    """
    La classe gestisce la codifica delle label .

    La convenzione adottata è:

    - benigno(2) -> 0
     -maligno(4) -> 1

    La mappatura è esplicita e fissata a priori.

     """

    def __init__(self):

        #dizionario di mapping da label originarie a label binarie
        self.label_to_int = {2: 0, 4: 1}

        #dizionario inverso per ricostruire le label originali
        self.int_to_label = {0: 2, 1: 4}

        #l'encoder è settato su True perchè  la mappatura è fissa
        self.fitted = True

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Metodo che trasforma le label originali in label binarie

        Parametri
       -------------------
        y : series
            Serie contenente le label originali

       Valore restiuito
       -------------------
       pd.Series
            Serie delle label binarie.

       Eccezioni
      -------------------
      ValueError
            Solleva un'eccezione se sono presenti label mancanti o non valide

        """
        #controllo presenza di label mancanti
        if y.isna().any():
            raise ValueError("Sono presenti label mancanti (NaN)")

        #controllo presenza di label diverse da quelle definite nel dominio
        invalid_labels = set(y.unique()) - set(self.label_to_int.keys())
        if invalid_labels:
            raise ValueError(f"Label non valida: {invalid_labels}")

        #mappatura delle label originarie in label binarie
        return  y.map(self.label_to_int)


    def inverse_transform(self, y_encoded: pd.Series) -> pd.Series:
        """
        Metodo di controllo: riporta le label binarie ai valori originali

        Parametri
       -------------------
        y_encoded : pd.Series
              Serie contenente le label binarie

       Valore restiuito
       -------------------
       pd.Series
            Serie delle label originarie.

        """
        return y_encoded.map(self.int_to_label)

class RawDatasetCleaner:
    """
    Classe responsabile della pulizia del dataset grezzo prima del preprocessing.

    Operazioni:
    - Conversione delle feature a numerico (se vi sono errori -> NaN)
    - Correzione di scala per valori x > 10:
          se x > 10 e x è multiplo di 10 e x/10 ricade nel range di dominio [1,10] allora x == x/10
    - Eliminazione delle righe che dopo la correzione contengono valori fuori dal range
    - Arrotondamento dei numeri decimale all'intero più vicino attraverso round
    - Gestione duplicati (record con stesso id):
          se per uno stesso id una label è mancante e una è presente, si tiene l'osservazione con la label
          se per uno stesso id esistono label contrastanti, si eliminano tutte le osservazioni con quell'id

    La classe produce anche un report con i conteggi delle righe modificate

    """

    def __init__(self, feature_cols: list[str], label_col: str, id_col: str, valid_min: float = 1.0, valid_max: float = 10.0):
        """
        Parametri
        -------------------
        - feature_cols: list[str]
              Lista dei nomi delle colonne delle feature
        - label_col: str
            Nome della colonna della label
        - id_col: str
            Nome della colonna identificativa (Sample code number nel caso specifico)
        - valid_min: float
            Valore minimo ammesso per le feature (default è 1)
        - valid_max: float
            Valore massimo ammesso per le feature (default è 10)

        """

        self.feature_cols = feature_cols
        self.label_col = label_col
        self.id_col = id_col
        self.valid_min = valid_min
        self.valid_max = valid_max


class DataPreprocessor:
    """
    Classe responsabile del preprocessing dei dati.

    Si occupa di:
    -separare features e label
    -convertire le feature in formato numerico per prevenire errori
    -gestire i valori mancanti nelle features tramite imputazione (mediana)
    -standardizzare le feature (opzionale)

    """

    def __init__(self, feature_cols: list[str], label_col: str, do_scaling: bool=True):
        """
        Parametri
       -------------------
       - feature_cols: list[str]
             Lista dei nomi delle colonne delle feature
       - label_col: str
              Nome della colonna della label
      - do_scaling: bool
               Se True applica la standardizzazione delle feature

        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.do_scaling = do_scaling

        #parametri appresi nel fit()
        self.impute_values = None
        self.means= None
        self.dev_std = None

        self.fitted = False  # impedisce l'uso di transform() prima di fit()


    def split_features_label(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa il DataFrame in feature (X) e label (y)

        Valore Restituito
        -------------------
        - X:  pd.DataFrame
               DataFrame contenente solo le feature
        - y: pd.Series
             Serie contenente la label

        """
        #conversione delle features in valori numerici
        X = df[self.feature_cols].apply(pd.to_numeric, errors = "coerce")

        #estrazione della colonna delle labels
        y = df[self.label_col]

        return X, y


    def  fit(self, df: pd.DataFrame):
        """
        Apprende i parametri utilizzando esclusivamente il training set.

        Calcola le mediane per imputare i valori mancanti delle feature

        Calcola la media e la deviazione standard per ogni feature

        """

        #utilizzo delle feature
        X, _ = self.split_features_label(df)

        # calcolo della mediana per ciascuna feature
        self.impute_values = X.median(numeric_only = True)
        X = X.fillna(self.impute_values)

        #calcolo della media e della deviazione standard
        if self.do_scaling:
            self.means = X.mean(numeric_only= True)
            self.dev_std = X.std(numeric_only = True, ddof = 0)

            #se una feature ha lo stesso valore per tutti i campioni la std = 0, quindi si sostituisce con 1 per evitare divisione per 0
            self.dev_std = self.dev_std.replace(0, 1.0)

        self.fitted = True
        return self


    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Applica al dataset di input le trasformazioni apprese nel fit().

        Converte le feature a numerico.

        Imputa i valori mancanti usando le mediane apprese nel training

        Standardizza (se True) usando media e deviazione standard apprese nel training

        Eccezioni
      -------------------
      RuntimeError
            Solleva un'eccezione se chiamato prima di fit()

        """

        if not self.fitted:
            raise RuntimeError("Devi chiamare fit() prima di transform()")


        X, y = self.split_features_label(df)

        #imputazione usando le mediane apprese nel fit()
        X = X.fillna(self.impute_values)

        #standardizzazione delle feature con parametri appresi nel fit()
        if self.do_scaling:
            X = (X - self.means) / self.dev_std

        return X, y



    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Combina con una sola chiamata i metodi fit() e transform() sullo stesso dataset (training set).

        """

        self.fit(df)
        return self.transform(df)


