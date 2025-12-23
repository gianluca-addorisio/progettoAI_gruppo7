import pandas as pd
from pathlib import Path


def get_loader(path: str):
    """"
    La funzione restituisce la funzione di caricamento appropriata per il dataset indicato dal percorso del file.

    La funzione di caricamento viene selezionata in base all'estensione del file e restituisce un DataFrame di pandas.

    Parametri
    -------------------
    path: str
      Percorso del file contenente il dataset.

    Valore restiuito
     -------------------
     Una funzione di caricamento che prende come input il percorso di un file e restituisce un Dataframe di pandas.

    Eccezioni
     -------------------
     Solleva un'eccezione se il formato del file non è supportato

    """
    ext = Path(path).suffix.lower()
    match ext:
        case ".csv":
            # standard file CSV
            return pd.read_csv
        case ".tsv":
            # file TSV:  come file CSV ma con separatore tabulare
            return lambda p: pd.read_csv(p, sep = "\t")    #sarebbe come scrivere def load_tsv(p):
                                                                                        #                                          return pd.read_csv(p, sep = "\t")
        case ".json":
            # standard JSON file
            return pd.read_json
        case ".xlsx":
            #standard excel file
            return pd.read_excel
        case ".txt":
            # si assume che sia formattato come CSV
            return pd.read_csv
        case _:
            # eccezione esplicita nel caso di formati non supportati
            raise ValueError(f"Unsupported format: {ext}")


