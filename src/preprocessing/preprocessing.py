import pandas as pd
import numpy as np

class DataPreprocessor:

    def __init__(self, feature_cols: list[str], label_col: str):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.fitted = False   #impedisce l'uso di transform() prima di fit()

    def  fit(selfself, df):
        #apprende i parametri di preprocessing usando esclusivamente il training set
        pass

    def transform(self, df):
        #applica le trasformazioni apprese nel fit a training set o test set
        pass

    def fit_transform(self, df):
        # combina fit() e transform() sul training set
        pass


