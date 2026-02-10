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

def _roc_curve_binary(y_true, y_score, positive_label=4):
    """
    Costruisce la curva ROC (FPR, TPR) per un classificatore binario.

    Parametri:
    - y_true: etichette vere {2,4}
    - y_score: score continuo (es. frazione di vicini maligni)
    - positive_label: label positiva (default = 4)

    Restituisce:
    - fpr: array False Positive Rate
    - tpr: array True Positive Rate
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Converto le etichette in binarie: 1 = positivo, 0 = negativo
    y_true_bin = (y_true == positive_label).astype(int)

    # Ordino i punti per score decrescente
    order = np.argsort(-y_score)
    y_true_bin = y_true_bin[order]

    # Numero totale di positivi e negativi
    P = np.sum(y_true_bin)
    N = len(y_true_bin) - P

    TPR = []
    FPR = []

    tp = 0
    fp = 0

    # Scorro la soglia implicitamente
    for label in y_true_bin:
        if label == 1:
            tp += 1
        else:
            fp += 1

        TPR.append(tp / P if P > 0 else 0.0)
        FPR.append(fp / N if N > 0 else 0.0)

    # Aggiungo punto (0,0) all'inizio
    TPR = np.array([0.0] + TPR)
    FPR = np.array([0.0] + FPR)

    return FPR, TPR


def auc_score(y_true, y_score, positive_label=4):
    """
    Calcola l'AUC (Area Under the ROC Curve).

    Parametri:
    - y_true: etichette vere {2,4}
    - y_score: score continuo ∈ [0,1]

    Restituisce:
    - auc: valore AUC
    """
    fpr, tpr = _roc_curve_binary(y_true, y_score, positive_label)

    # Area sotto la curva ROC (regola dei trapezi)
    auc = np.trapz(tpr, fpr)

    return auc


