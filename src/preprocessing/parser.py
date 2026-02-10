
import argparse
import os


metriche_consentite = set (["accuracy", "error rate", "sensitivity", "specificity", "geometric mean", "AUC", "all"])

#definizione del parser
def add_args(p):

    p.add_argument("--dataset", required=True, help="Inserire il path del dataset")
    p.add_argument("--k", dest= "k_neighbors", required = True, type = int, help = "Numero di vicini k per il KNN")
    p.add_argument("--metrics", nargs = '+', required = True, default = ["all"], help= "Metriche da validare: accuracy, error rate, sensitivity, specificity, geometric mean, AUC oppure all")



def build_parser():

    parser = argparse.ArgumentParser(
        prog="progettoAI_gruppo7",
        description="Pipeline KNN e valutazione"
    )
    sub = parser.add_subparsers(dest="eval_mode", required=True)

    # Metodo HOLDOUT
    p_hold = sub.add_parser("holdout", help = "Valutazione holdout.")
    add_args(p_hold)
    p_hold.add_argument("--test_size", type= float, required = True, default = 0.3, help = "Percentuale del test set. Il default è 0.3. Il train è 1 - test_size")

    # Metodo B: K-FOLD CROSS VALIDATION
    p_B = sub.add_parser("B", help = "Validazione B: K-fold cross validation.")
    add_args(p_B)
    p_B.add_argument("--K", dest = "k_folds", type = int, help = "Numero di fold K per la k-fold cross validation." )

    # Metodo C: RANDOM SUBSAMPLING
    p_C = sub.add_parser("C", help = "Validazione C: random subsampling.")
    add_args(p_C)
    p_C.add_argument("--K", dest = "n_ripetizioni", type = int, required = True, help = "Numero di ripetizioni K per random subsampling.")
    p_C.add_argument("--test_size", type = float ,defult = 0.3, help = "Percentuale del test set per ogni ripetizione. Default = 0.3")


    return parser


def validate_metrics(metrics):
    m = [x.strip() for x in metrics]

    if "all" in m:
        return ["all"]

    for x in m:
        if x not in metriche_consentite:
            raise SystemExit("Errore: metrica non valida")

def validate_args(ns):

    # ci saranno le eccezioni in caso di argomenti non validi
        pass
