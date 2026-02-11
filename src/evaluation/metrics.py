"""
Questo modulo contiene:
1) Funzioni base per confusion matrix e metriche classiche (accuracy, sensitivity, etc.)
2) Funzioni ROC/AUC (per label binarie 0/1)
3) Implementazione di Strategy Pattern per le metriche:
   - tutte le metriche espongono la stessa interfaccia: MetricStrategy.compute(...)
   - alcune metriche richiedono solo (y_true, y_pred), AUC richiede anche y_score
   - il main/evaluation può selezionare metriche via stringhe senza if/else sparsi

Assunzioni
----------
Label binarie:
- 0 = benigno (negativo)
- 1 = maligno (positivo)
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


# ==============================
# 1) FUNZIONI BASE 
# ==============================

def confusion_matrix_binary(y_true, y_pred, positive_label: int = 1) -> Tuple[int, int, int, int]:
    """
    Confusion matrix binaria: TP, FP, TN, FN.

    Parametri:
    - y_true: etichette vere (0/1)
    - y_pred: etichette predette (0/1)
    - positive_label: classe positiva (default=1)

    Ritorna:
    - TP, FP, TN, FN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    TP = int(np.sum((y_true == positive_label) & (y_pred == positive_label)))
    TN = int(np.sum((y_true != positive_label) & (y_pred != positive_label)))
    FP = int(np.sum((y_true != positive_label) & (y_pred == positive_label)))
    FN = int(np.sum((y_true == positive_label) & (y_pred != positive_label)))

    return TP, FP, TN, FN


def accuracy_rate(TP: int, FP: int, TN: int, FN: int) -> float:
    den = TP + FP + TN + FN
    return (TP + TN) / den if den > 0 else 0.0


def error_rate(TP: int, FP: int, TN: int, FN: int) -> float:
    den = TP + FP + TN + FN
    return (FP + FN) / den if den > 0 else 0.0


def sensitivity(TP: int, FN: int) -> float:
    """Recall della classe positiva."""
    den = TP + FN
    return TP / den if den > 0 else 0.0


def specificity(TN: int, FP: int) -> float:
    """Recall della classe negativa."""
    den = TN + FP
    return TN / den if den > 0 else 0.0


def geometric_mean(sens: float, spec: float) -> float:
    return float(np.sqrt(sens * spec))


def evaluate_classification(y_true, y_pred, positive_label: int = 1) -> Dict[str, float]:
    """
    Valuta il classificatore con metriche base richieste tipicamente dal progetto.

    Ritorna:
    - accuracy
    - error_rate
    - sensitivity
    - specificity
    - gmean
    """
    TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)

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


# ==============================
# 2) ROC / AUC (score continuo)
# ==============================

def roc_curve_binary(y_true, y_score, positive_label: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Costruisce ROC (FPR, TPR) usando soglie uniche dello score.

    Parametri:
    - y_true: etichette vere (0/1)
    - y_score: score continuo in [0,1] (es. frazione di vicini positivi del KNN)
    - positive_label: classe positiva (default=1)

    Ritorna:
    - fpr, tpr, thresholds
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    y_true_bin = (y_true == positive_label).astype(int)

    P = int(np.sum(y_true_bin))
    N = int(len(y_true_bin) - P)
    if P == 0 or N == 0:
        raise ValueError("ROC non definibile: manca una classe (P=0 o N=0).")

    thresholds = np.unique(y_score)[::-1]
    thresholds = np.r_[np.inf, thresholds, -np.inf]

    tpr = []
    fpr = []

    for thr in thresholds:
        y_pred_pos = (y_score >= thr)

        TP = int(np.sum(y_pred_pos & (y_true_bin == 1)))
        FP = int(np.sum(y_pred_pos & (y_true_bin == 0)))

        tpr.append(TP / P)
        fpr.append(FP / N)

    return np.array(fpr), np.array(tpr), thresholds


def auc_score(y_true, y_score, positive_label: int = 1) -> float:
    """
    Calcola AUC come area sotto ROC (trapezi).
    """
    fpr, tpr, _ = roc_curve_binary(y_true, y_score, positive_label=positive_label)
    return float(np.trapezoid(tpr, fpr))


# ==========================================
# 3) STRATEGY PATTERN PER SELEZIONE METRICHE
# ==========================================

@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float


class MetricStrategy(ABC):
    """
    Strategy Pattern per le metriche.

    Idea:
    - Tutte le metriche condividono la stessa interfaccia compute(...)
    - Alcune metriche richiedono solo y_true/y_pred
    - Altre (es. ROC) richiedono anche y_score (score continuo)
    """
    name: str

    @property
    def requires_score(self) -> bool:
        """True se la metrica richiede y_score (es. AUC)."""
        return False

    @abstractmethod
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        positive_label: int = 1
    ) -> float:
        raise NotImplementedError


class AccuracyMetric(MetricStrategy):
    name = "accuracy"

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
        return accuracy_rate(TP, FP, TN, FN)


class ErrorRateMetric(MetricStrategy):
    name = "error_rate"

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
        return error_rate(TP, FP, TN, FN)


class SensitivityMetric(MetricStrategy):
    name = "sensitivity"

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
        return sensitivity(TP, FN)


class SpecificityMetric(MetricStrategy):
    name = "specificity"

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
        return specificity(TN, FP)


class GMeanMetric(MetricStrategy):
    name = "gmean"

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        TP, FP, TN, FN = confusion_matrix_binary(y_true, y_pred, positive_label=positive_label)
        sens = sensitivity(TP, FN)
        spec = specificity(TN, FP)
        return geometric_mean(sens, spec)


class AUROCMetric(MetricStrategy):
    name = "auc"

    @property
    def requires_score(self) -> bool:
        return True

    def compute(self, y_true, y_pred, y_score=None, positive_label: int = 1) -> float:
        if y_score is None:
            raise ValueError("AUROCMetric richiede y_score (score continuo).")
        return auc_score(y_true, y_score, positive_label=positive_label)


# Registro minimale delle metriche disponibili.
_METRIC_REGISTRY: Dict[str, MetricStrategy] = {
    "accuracy": AccuracyMetric(),
    "error_rate": ErrorRateMetric(),
    "sensitivity": SensitivityMetric(),
    "specificity": SpecificityMetric(),
    "gmean": GMeanMetric(),
    "auc": AUROCMetric(),
}


def build_metric_strategies(metric_names: Iterable[str]) -> List[MetricStrategy]:
    """
    Crea la lista di metriche (Strategy) a partire da nomi stringa.

    Supporta:
    - ["all"] -> tutte
    - lista esplicita -> solo quelle richieste
    """
    names = [m.strip().lower() for m in metric_names]
    if len(names) == 1 and names[0] == "all":
        return list(_METRIC_REGISTRY.values())

    strategies = []
    for n in names:
        if n not in _METRIC_REGISTRY:
            raise ValueError(f"Metrica non supportata: '{n}'. Supportate: {list(_METRIC_REGISTRY.keys())} o 'all'.")
        strategies.append(_METRIC_REGISTRY[n])
    return strategies


def evaluate_with_strategies(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray],
    metrics: List[MetricStrategy],
    positive_label: int = 1
) -> Dict[str, float]:
    """
    Valuta un set di metriche Strategy in modo uniforme.

    - Se una metrica richiede score e y_score è None -> errore (scelta intenzionale: evita risultati sbagliati).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score_arr = None if y_score is None else np.asarray(y_score, dtype=float)

    out: Dict[str, float] = {}
    for m in metrics:
        out[m.name] = float(m.compute(y_true, y_pred, y_score=y_score_arr, positive_label=positive_label))
    return out


