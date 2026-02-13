from abc import ABC, abstractmethod
from .evaluation import (
    evaluate_model_holdout,
    evaluate_model_kfold,
    evaluate_model_subsampling,
)


class ValidationStrategy(ABC):
    """
    Classe astratta che definisce l'interfaccia comune
    per tutte le strategie di validazione.
    """

    @abstractmethod
    def evaluate(self, model, X, y, ids, metric_names, **kwargs):
        pass


class HoldoutStrategy(ValidationStrategy):
    """
    Strategia di validazione Holdout.
    """

    def evaluate(self, model, X, y, ids, metric_names, **kwargs):
        return evaluate_model_holdout(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            test_size=kwargs.get("test_size") or 0.3,
            seed=kwargs.get("seed"),
            positive_label=kwargs.get("positive_label"),
        )


class KFoldStrategy(ValidationStrategy):
    """
    Strategia di validazione K-Fold (modalità B).
    """

    def evaluate(self, model, X, y, ids, metric_names, **kwargs):
        return evaluate_model_kfold(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            k=kwargs.get("K") or 5,
            seed=kwargs.get("seed"),
            positive_label=kwargs.get("positive_label"),
        )


class SubsamplingStrategy(ValidationStrategy):
    """
    Strategia di validazione Random Subsampling (modalità C).
    """

    def evaluate(self, model, X, y, ids, metric_names, **kwargs):
        return evaluate_model_subsampling(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            r=kwargs.get("r") or 10,
            test_size=kwargs.get("test_size") or 0.3,
            seed=kwargs.get("seed"),
            positive_label=kwargs.get("positive_label"),
        )


class ValidationFactory:
    """
    Factory per creare la strategia di validazione a partire dalla modalità scelta.
    """

    @staticmethod
    def create(mode: str) -> ValidationStrategy:
        mode = mode.lower()

        if mode == "holdout":
            return HoldoutStrategy()

        if mode == "b":
            return KFoldStrategy()

        if mode == "c":
            return SubsamplingStrategy()

        raise ValueError(f"Modalità di validazione non riconosciuta: {mode}")
