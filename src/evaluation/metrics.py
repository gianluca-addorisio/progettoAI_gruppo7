import numpy as np

# ============================================================
# NOTE:
# Questo progetto utilizza label {2, 4} come specificato
# nella traccia:
#   - 2 = benigno
#   - 4 = maligno (classe positiva)
#
# Tutte le metriche seguenti sono quindi definite in modo
# coerente rispetto a questa codifica.
# ============================================================


def confusion_matrix_binary(y_true, y_pred, positive_label=4):
    """
    Calcola i valori della confusion matrix binaria per il dataset.

    Parametri:
    - y_true: array delle etichette reali (2 o 4)
    - y_pred: array delle etichette predette (2 o 4)
    - positive_label: label considerata positiva (default = 4)

    Restituisce:
    - TP, FP, TN, FN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = np.sum((y_true == positive_label) & (y_pred == positive_label))
    TN = np.sum((y_true != positive_label) & (y_pred != positive_label))
    FP = np.sum((y_true != positive_label) & (y_pred == positive_label))
    FN = np.sum((y_true == positive_label) & (y_pred != positive_label))

    return TP, FP, TN, FN


def accuracy_rate(TP, FP, TN, FN):
    return (TP + TN) / (TP + FP + TN + FN)


def error_rate(TP, FP, TN, FN):
    return (FP + FN) / (TP + FP + TN + FN)


def sensitivity(TP, FN):
    """Recall della classe positiva (maligno)."""
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def specificity(TN, FP):
    """Recall della classe negativa (benigno)."""
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0


def geometric_mean(sens, spec):
    return np.sqrt(sens * spec)


def evaluate_classification(y_true, y_pred):
    """
    Valuta il classificatore secondo le metriche richieste dalla traccia.

    Restituisce:
    - accuracy
    - error_rate
    - sensitivity
    - specificity
    - geometric mean
    """
    TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred)

    acc = accuracy_rate(TP, FP, TN, FN)
    err = error_rate(TP, FP, TN, FN)
    sens = sensitivity(TP, FN)
    spec = specificity(TN, FP)
    gmean = geometric_mean(sens, spec)

    return {
        "accuracy": acc,
        "error_rate": err,
        "sensitivity": sens,
        "specificity": spec,
        "gmean": gmean
    }


