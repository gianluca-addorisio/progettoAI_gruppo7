import pandas as pd
import numpy as np

# 1. Import dei moduli del progetto
from src.dataset.dataset import get_loader
from src.preprocessing.preprocessing import DataPreprocessor, LabelEncoder
from src.models.knn import KNN, EuclideanDistance
from src.evaluation.evaluation import (
    evaluate_model_holdout,
    evaluate_model_kfold,
    evaluate_model_subsampling
)

def main():
    # Percorso del dataset
    dataset_path = "data/raw/version_1.csv"

    # Definizione colonne
    column_names = [
        "Sample code number", "clump_thickness_ty", "uniformity_cellsize_xx",
        "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size",
        "bareNucleix_wrong", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "classtype_v1"
    ]

    feature_cols = column_names[1:-1]  # Esclude id e la label finale
    label_col = "classtype_v1"
    id_col = "Sample code number"

    # --- 1. CARICAMENTO DATASET ---
    print(f"Caricamento dataset: {dataset_path}...")
    loader = get_loader(dataset_path)
    df = loader(dataset_path)

    # --- 2. PREPROCESSING ---
    print("Inizio Preprocessing...")
    preprocessor = DataPreprocessor(feature_cols=feature_cols, label_col=label_col)

    #!!! Elimina le righe dove la colonna delle label è NaN
    df = df.dropna(subset=[label_col])

    # Adattiamo il preprocessor
    X, y_raw = preprocessor.fit_transform(df)

    # Estraiamo gli ID per lo split corretto
    ids = df[id_col].values
    print("Fine Preprocessing...")

    # --- 3. CONFIGURAZIONE MODELLO k-NN ---
    print("Configurazione k-NN...")
    k_value = 5
    dist_strategy = EuclideanDistance()
    model = KNN(k=k_value, distance_strategy=dist_strategy)

    # --- 4. VALUTAZIONE ---
    print("\n--- RISULTATI VALUTAZIONE ---")

    # A. Holdout
    print("\n>>> METRICHE HOLDOUT (Singolo Split):")
    res_holdout = evaluate_model_holdout(model, X.values, y_raw.values, ids, test_size=0.3)
    for metrica, valore in res_holdout.items():
        print(f"{metrica.capitalize():<15}: {valore:.4f}")

    # B. K-Fold (Split per ID)
    k_folds = 5
    print(f"\n>>> METRICHE {k_folds}-FOLD (Media +/- DevStd):")
    res_kfold = evaluate_model_kfold(model, X.values, y_raw.values, ids, k=k_folds)
    for metrica, stats in res_kfold.items():
        # stats è un dizionario con {'mean': ..., 'std': ...}
        print(f"{metrica.capitalize():<15}: {stats['mean']:.4f} (+/- {stats['std']:.4f})")

    # C. Random Subsampling
    iterations = 30
    print(f"\n>>> METRICHE RANDOM SUBSAMPLING ({iterations} iterazioni):")
    res_sub = evaluate_model_subsampling(model, X.values, y_raw.values, ids, r=iterations, test_size=0.3)
    for metrica, stats in res_sub.items():
        print(f"{metrica.capitalize():<15}: {stats['mean']:.4f}")

    #!!! AUC mancante?

if __name__ == "__main__":
    main()


