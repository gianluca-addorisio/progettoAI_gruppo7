"""
 Main:


    1) Parsing degli argomenti da linea di comando (CLI);
    2) Caricamento del dataset tramite factory in base all’estensione;
    3) Pulizia del dataset grezzo (range valori, duplicati, label valide, ecc.);
    4) Preprocessing (imputazione, standardizzazione);
    5) Codifica delle label in formato binario (0/1);
    6) Addestramento e valutazione del modello KNN secondo la modalità scelta:
          - holdout
          - B (k-fold cross validation)
          - C (random subsampling)
    7) Calcolo delle metriche richieste;
    8) Salvataggio dei risultati e dei plot nella cartella "results/".


"""


import json
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from src.preprocessing.CLI_args import parse_args
from src.dataset.dataset import get_loader
from src.preprocessing.preprocessing import RawDatasetCleaner, DataPreprocessor, LabelEncoder
from src.models.knn import KNN, EuclideanDistance
from src.evaluation.evaluation import (
    evaluate_model_holdout,
    evaluate_model_kfold,
    evaluate_model_subsampling
)
from results.plots.plot import plot_metric_summary, plot_metric_distribution


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    """
    Determina (id_col, label_col, feature_cols):
    - prova prima nomi comuni
    - altrimenti assume: prima colonna = id, ultima = label
    """
    cols = list(df.columns)

    id_candidates = ["Sample code number", "sample_code_number", "ID", "id"]
    label_candidates = ["Class", "class", "Label", "label", "Target", "target"]

    id_col = next((c for c in id_candidates if c in cols), cols[0])
    label_col = next((c for c in label_candidates if c in cols), cols[-1])

    feature_cols = [c for c in cols if c not in (id_col, label_col)]
    return id_col, label_col, feature_cols


def save_json(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    # 1) CLI
    ns = parse_args()

    # 2) LOAD dataset (factory per estensione)
    dataset_path = Path(ns.dataset)
    loader = get_loader(str(dataset_path))
    df = loader(str(dataset_path))

    # 3) Colonne
    id_col, label_col, feature_cols = infer_columns(df)

    # 4) RAW CLEANING (range, scala, decimali, duplicati, label valide)
    cleaner = RawDatasetCleaner(
        feature_cols=feature_cols,
        label_col=label_col,
        id_col=id_col,
        valid_min=1.0,
        valid_max=10.0
    )
    df_clean, cleaning_report = cleaner.clean(df)

    # 5) PREPROCESS (impute + scaling)
    pre = DataPreprocessor(feature_cols=feature_cols, label_col=label_col, do_scaling=True)
    X_df, y_raw = pre.fit_transform(df_clean)

    # 6) LABEL ENCODING: 2/4 -> 0/1
    le = LabelEncoder()
    y = le.transform(y_raw).to_numpy(dtype=int)

    # 7) IDs per split by-id
    ids = df_clean[id_col].to_numpy()

    # 8) Model
    model = KNN(k=ns.k_neighbors, distance_strategy=EuclideanDistance())

    # 9) Evaluation
    X = X_df.to_numpy(dtype=float)
    metric_names = ns.metriche  # già normalizzate (["all"] oppure lista)

    # Coerenza: in 0/1 la classe positiva è 1
    positive_label = 1
    seed = 42

    if ns.eval_mode == "holdout":
        results = evaluate_model_holdout(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            test_size=ns.test_size,
            seed=seed,
            positive_label=positive_label
        )

    elif ns.eval_mode == "B":
        results = evaluate_model_kfold(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            k=ns.k_folds,
            seed=seed,
            positive_label=positive_label
        )

    elif ns.eval_mode == "C":
        results = evaluate_model_subsampling(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            r=ns.n_ripetizioni,
            test_size=ns.test_size,
            seed=seed,
            positive_label=positive_label
        )

    else:
        raise SystemExit(f"Modalità non riconosciuta: {ns.eval_mode}")

    # 10) Output a video
    print("\nCleaning Report:")
    for k, v in cleaning_report.items():
        print(f"{k}: {v}")

    print("\nRisultati:")
    print(json.dumps(results, indent=2))

    # 11) Salvataggio risultati
    payload = {
        "cli_args": vars(ns),
        "columns": {"id": id_col, "label": label_col, "features": feature_cols},
        "cleaning_report": cleaning_report,
        "results": results
    }
    save_json(Path("../results") / "results.json", payload)
    print("\n[OK] Salvato: results/results.json")

    # Plot solo delle metriche richieste (se "all" → tutte)
    metriche_da_plottare = results.keys()

    outdir = Path("../results")

    for m in metriche_da_plottare:
        plot_metric_summary(results, m, outdir)
        plot_metric_distribution(results, m, outdir)

    print("\n[OK] Plot salvati in results/")


if __name__ == "__main__":
    main()

