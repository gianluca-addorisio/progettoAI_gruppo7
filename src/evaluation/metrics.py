import numpy as np

def confusion_matrix(y_true, y_pred):
    """
    Restituisce confusion matrix binaria:
    [[TN, FP],
     [FN, TP]]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TP = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[TN, FP],
                     [FN, TP]])


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    if TN + FP == 0:
        return 0.0
    return TN / (TN + FP)


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

