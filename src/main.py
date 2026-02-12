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
import matplotlib.pyplot as plt
from .evaluation.metrics import roc_curve_binary

from .preprocessing.CLI_args import parse_args
from .dataset.dataset import get_loader
from .preprocessing.preprocessing import RawDatasetCleaner, DataPreprocessor, LabelEncoder
from .models.knn import KNN, EuclideanDistance
from .evaluation.evaluation import (
    evaluate_model_holdout,
    evaluate_model_kfold,
    evaluate_model_subsampling
)
from .evaluation.plot import plot_metric_summary, plot_metric_distribution


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

    # 3) COLONNE
    id_col = "Sample code number"
    label_col = "classtype_v1"
    feature_cols = ["Mitoses", "Normal Nucleoli", "Single Epithelial Cell Size", "uniformity_cellsize_xx",
                    "clump_thickness_ty", "Marginal Adhesion", "Bland Chromatin", "Uniformity of Cell Shape",
                    "bareNucleix_wrong"]

    # 4) RAW CLEANING (range, scala, decimali, duplicati, label valide)
    cleaner = RawDatasetCleaner(
        feature_cols=feature_cols,
        label_col=label_col,
        id_col=id_col,
        valid_min=1.0,
        valid_max=10.0
    )
    print("Before cleaning:", df.shape)
    df_clean, cleaning_report = cleaner.clean(df)
    print("After cleaning:", df_clean.shape)
    print("Cleaning report:", cleaning_report)

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
        holdout_out = evaluate_model_holdout(
            model=model,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            test_size=ns.test_size,
            seed=seed,
            positive_label=positive_label
        )

        results = holdout_out["metrics"]
        y_true_holdout = holdout_out["y_true"]
        y_pred_holdout = holdout_out["y_pred"]
        y_score_holdout = holdout_out["y_score"]

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

    # --- Confronto Holdout vs Media K-Fold (solo se eval_mode == "B") ---
    if ns.eval_mode == "B":
        # Holdout "di riferimento" per confronto (test_size fisso a 0.3)
        model_holdout = KNN(k=ns.k_neighbors, distance_strategy=EuclideanDistance())
        holdout_out_cmp = evaluate_model_holdout(
            model=model_holdout,
            X=X,
            y=y,
            ids=ids,
            metric_names=metric_names,
            test_size=0.3,
            seed=seed,
            positive_label=positive_label
        )
        holdout_metrics = holdout_out_cmp["metrics"]

        print("\n=== Confronto Holdout vs Media K-Fold ===")
        for metric, agg in results.items():
            if isinstance(agg, dict) and "mean" in agg and metric in holdout_metrics:
                kfold_mean = agg["mean"]
                holdout_value = holdout_metrics[metric]
                try:
                    print(f"{metric}:")
                    print(f"  Holdout      = {float(holdout_value):.4f}")
                    print(f"  K-Fold mean  = {float(kfold_mean):.4f}")
                    print(f"  Differenza   = {float(kfold_mean) - float(holdout_value):.4f}")
                except (TypeError, ValueError):
                    # Se qualche metrica non è numerica, salta senza rompere l'esecuzione
                    continue

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

    # Directory base del progetto
    base_dir = Path(__file__).resolve().parent.parent
    results_dir = base_dir / "results"

    # Salvataggio JSON
    save_json(results_dir / "results.json", payload)
    print("\n[OK] Salvato: results/results.json")

    # Plot solo delle metriche richieste
    metriche_da_plottare = results.keys()
    is_holdout = ns.eval_mode == "holdout"

    outdir = results_dir

    for m in metriche_da_plottare:
        plot_metric_summary(results, m, outdir)
        if not is_holdout:
            plot_metric_distribution(results, m, outdir)

    print("\n[OK] Plot salvati in results/")

    # ROC curve SOLO per holdout (al momento, poi ampliamo per B e C)
    if ns.eval_mode == "holdout" and y_score_holdout is not None:
        fpr, tpr, _ = roc_curve_binary(y_true_holdout, y_score_holdout)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {results['auc']:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Holdout)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plots_dir = results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(plots_dir / "roc_curve.png")
        plt.close()

        print("[OK] ROC salvata in results/plots/")

        from .evaluation.metrics import confusion_matrix_binary
        if ns.eval_mode == "holdout":
            TP, FP, TN, FN = confusion_matrix_binary(
                y_true_holdout,
                y_pred_holdout,
                positive_label=positive_label
            )

            cm = np.array([[TN, FP],
                           [FN, TP]])

            # Percentuali per riga (normalizzazione su true label)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_percent = cm / row_sums * 100

            plt.figure(figsize=(6, 5))
            im = plt.imshow(cm, cmap="Blues")

            plt.title("Confusion Matrix (Holdout)", fontsize=12)
            plt.colorbar(im, fraction=0.046, pad=0.04)

            plt.xticks([0, 1], ["Pred 0 (Benigno)", "Pred 1 (Maligno)"])
            plt.yticks([0, 1], ["True 0 (Beningo)", "True 1 (Maligno)"])

            # Scrittura valori dentro celle
            for i in range(2):
                for j in range(2):
                    value = cm[i, j]
                    perc = cm_percent[i, j]
                    text_color = "white" if value > cm.max() / 2 else "black"

                    plt.text(
                        j, i,
                        f"{value}\n({perc:.1f}%)",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=11
                    )

            plt.ylabel("Classe Reale")
            plt.xlabel("Classe Predetta")
            plt.tight_layout()

            plots_dir = results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
            plt.close()

            print("[OK] Confusion Matrix salvata in results/plots/")


if __name__ == "__main__":
    main()
