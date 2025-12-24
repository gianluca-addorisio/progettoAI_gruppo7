import pandas as pd
import numpy as np


class LabelEncoder:
    """""
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
       Una serie delle label binarie.

       Eccezioni
      -------------------
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
        y_encoded : series
       Serie contenente le label binarie

       Valore restiuito
       -------------------
       Una serie delle label originarie.

        """
        return y_encoded.map(self.int_to_label)



class DataPreprocessor:
    """
    Classe responsabile del preprocessing dei dati.

    Si occupa di:
    -separare features e label
    -convertire le feature in formato numerico per prevenire errori
    -gestire i valori mancanti nelle features tramite imputazione

    """

    def __init__(self, feature_cols: list[str], label_col: str):
        """
        Parametri
       -------------------
       - feature_cols: lista dei nomi delle colonne delle feature
       - label_col: nome della colonna della label

        """
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.impute_values = None
        self.fitted = False   #impedisce l'uso di transform() prima di fit()

    def split_features_label(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separa il DataFrame in feature (X) e label (y)

        Valore Restituito
        -------------------
        - X: DataFrame contenente solo le feature
        - y: Series contenente la label

        """
        #conversione delle features in valori numerici
        X = df[self.feature_cols].apply(pd.to_numeric, errors = "coerce")

        #estrazione della colonna delle labels
        y = df[self.label_col]

        return X, y


    def  fit(self, df: pd.DataFrame):
        """
        Apprende i parametri utilizzando esclusivamente il training set.

        Calcola il valore di imputazione per i valori mancanti delle feature utilizzano la mediana

        """

        #utilizzo delle feature
        X, _ = self.split_features_label(df)

        # calcolo della mediana per ciascuna feature
        self.impute_values = X.median(numeric_only = True)

        self.fitted = True
        return self


    def transform(self, df):
        #applica le trasformazioni apprese nel fit a training set o test set
        pass

    def fit_transform(self, df):
        # combina fit() e transform() sul training set
        pass


