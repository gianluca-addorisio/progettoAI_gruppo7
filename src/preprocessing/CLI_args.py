"""

Questo modulo gestisce il parsing degli argomenti da linea di comando attraverso la libreria argparse.

Il modulo definisce:
- parametri comuni: dataset, numero vicini (k) e metriche
- subcomandi: holdout, K-fold cross validation (B) e Random Subsampling (C)
- validazione per gli argomenti inseriti
"""
import argparse


# insieme delle metriche accettate da linea di comando
metriche_consentite = set (["accuracy", "error_rate", "sensitivity", "specificity", "geometric_mean", "auc", "all"])


def add_args(p):

    """

    Aggiunge al parser e al subparser gli argomenti comuni a tutte le modalità di valytazione, cioè:
    - --dataset: path del dataset
    - --k_neighbors: numero di k vicini del classificatore
    - --metriche: elenco metriche da calcolare metrics

    Parametri:
    ----------
    p : argparse.ArgumentParser
     Il parser o subparser su cui vengono registrati gli argomenti.

    """
    p.add_argument("--dataset", required=True, help="Inserire il path del dataset")
    p.add_argument("--k", dest= "k_neighbors", required = True, type = int, help = "Numero di vicini k per il KNN")
    p.add_argument("--metriche", nargs = '+', default = ["all"], help= "Metriche da validare: accuracy, error_rate, sensitivity, specificity, geometric_mean, auc oppure all")


def build_parser():
    """

    Costruisce e restituisce il parser principale della CLI.

    Struttura:
    - comando holdout: valutazione holdout che richiede test_size
    - comando B: validazione K-fold cross validation che richiede numero di folds k
    - comando C: validazione random subsampling che richiede numero di ripetizioni K e test_size

    Restituisce:
    ----------
    parser : argparse.ArgumentParser
           Parser configurato con subcomandi e argomenti.
    """

    parser = argparse.ArgumentParser(
        prog="progettoAI_gruppo7",
        description="Pipeline KNN, metriche e validazione (Holdout, B= K-fold, C = subsampling)"
    )

    # i subparser gestiscono le tre diverse modalità di validazione
    sub = parser.add_subparsers(dest="eval_mode", required=True)

    # Metodo HOLDOUT
    p_hold = sub.add_parser("holdout", help = "Valutazione holdout.")
    add_args(p_hold)
    p_hold.add_argument("--test_size", type= float, default = 0.3, help = "Percentuale del test set. Il default è 0.3. Il train è 1 - test_size")

    # Metodo B: K-FOLD CROSS VALIDATION
    p_B = sub.add_parser("B", help = "Validazione B: K-fold cross validation.")
    add_args(p_B)
    p_B.add_argument("--K", dest = "k_folds", required= True, type = int, help = "Numero di fold K per la k-fold cross validation." )

    # Metodo C: RANDOM SUBSAMPLING
    p_C = sub.add_parser("C", help = "Validazione C: random subsampling.")
    add_args(p_C)
    p_C.add_argument("--K", dest = "n_ripetizioni", type = int, required = True, help = "Numero di ripetizioni K per random subsampling.")
    p_C.add_argument("--test_size", type = float ,default= 0.3, help = "Percentuale del test set per ogni ripetizione. Default = 0.3")


    return parser


def validate_metrics(metriche):

    """

    Valida e normalizza le metriche inserite dall'utente.
    In particolare:

    - tutte le metriche inserite vengono trasformate in lowercase e ripulite con strip()
    - se è presente "all" viene restituito solo ["all"]
    - se una metrica non è presente nel set consentito, termina con SystemExit che solleva errore

    Parametri:
    ----------
    metriche : list[str]
        Lista delle metriche passate da riga di comando.

    Restituisce:
    -----------
    list[str]
        Lista normalizzata delle metriche

    """

    m = [x.strip().lower() for x in metriche]

    if "all" in m:
        return ["all"]

    for x in m:
        if x not in metriche_consentite:
            raise SystemExit("Errore: metrica non valida")

    return m

def validate_args(ns):

    """
    Valida la coerenza degli argomenti per la modalità selezionata.

    In particolare:
      - k_neighbors >= 1, cioè numero vicini deve essere >= 1
      - test_size deve essere in (0,1) quando esiste (holdout e C)
      - per B: k_folds >= 2
      - per C: n_ripetizioni >= 1

      Parametri
      ----------
      ns : argparse.Namespace
          namespace prodotto dal parser (contiene i campi degli argomenti).

      Restituisce
      -----------
      argparse.Namespace
          Lo stesso namespace con eventuali campi normalizzati.
    """
    # controllo numero vicini (k)
    if ns.k_neighbors < 1:
        raise SystemExit("Errore: k deve essere maggiore o uguale a 1")

    # validazione metriche inserite
    ns.metriche = validate_metrics(ns.metriche)

    # validazione test-size solo quando esiste
    if hasattr(ns, "test_size"):
        if ns.test_size < 0 or ns.test_size > 1:
            raise SystemExit("Errore: --test_size deve essere compreso tra 0 e 1")

    # validazione k_folds nel k-fold cross validation; è opportuno avere un numero superiore a 2
    if ns.eval_mode == "B":
        if ns.k_folds < 2:
            raise SystemExit("Errore: per k-fold serve --K > 2")

    # validazione numero ripetizioni e test-size in random subsampling
    if ns.eval_mode == "C":
        if ns.n_ripetizioni < 1:
            raise SystemExit("Errore: deve valere --K >= 1")
        if ns.test_size < 0 or ns.test_size > 1:
            raise SystemExit("Errore: --test_size deve essere compreso tra 0 e 1")

    return ns

def parse_args():
    """

    Funzione di chiamata CLI dal main:
    - costruisce il parser
    - legge gli argomenti
    - valida e restituisce il Namespace

    Restituisce:
    -----------
    argparse.Namespace
        Configurazione completa estratta da riga di comando.

    """

    parser = build_parser()
    ns = parser.parse_args()
    return validate_args(ns)









