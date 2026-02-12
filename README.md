# Progetto AI – Classificazione Tumori con k-NN (Gruppo 7)

Progetto per il corso di **Fondamenti di Intelligenza Artificiale**.

Il progetto implementa una pipeline completa per la classificazione binaria di tumori (benigno vs maligno) utilizzando un classificatore **k-Nearest Neighbors (k-NN)** sviluppato interamente da zero, senza l'utilizzo di librerie di machine learning come scikit-learn.

---

## Obiettivo

Realizzare un sistema modulare che consenta di:

1. Caricare ed effettuare il preprocessing del dataset assegnato (`version_1.csv`).
2. Implementare e addestrare un classificatore k-NN con distanza euclidea.
3. Validare il modello tramite:
   - Holdout
   - K-Fold Cross Validation (B)
   - Random Subsampling (C)
4. Calcolare e analizzare le metriche di performance selezionabili da linea di comando:
   - Accuracy Rate
   - Error Rate
   - Sensitivity 
   - Specificity  
   - Geometric Mean    
   - AUC (Area Under the Curve)
   - `all` (tutte le precedenti)
   Sono inoltre generati come output grafici: 
      - Confusion Matrix
      - ROC Curve (quando è richiesta AUC)
5. Salvare risultati numerici e grafici nella cartella results/.

---

## Architettura del Progetto

Il codice è organizzato in moduli separati per garantire chiarezza strutturale, riproducibilità e separazione delle responsabilità.

```
progettoAI_gruppo7/
│
├── data/
│   ├── raw/                # Dataset originali
│   └── processed/          # Dataset preprocessati
│
├── src/
│   ├── dataset/            # Caricamento dati
│   ├── preprocessing/      # Pulizia e trasformazioni
│   ├── models/             # Implementazione k-NN (Strategy Pattern)
│   ├── evaluation/         # Validazione, metriche, split
│   ├── utils/              # Funzioni di supporto
│   └── main.py             # Punto di ingresso della pipeline
│
├── results/
│   └── plots/              # Grafici generati
│
├── tests/                  # Test automatici (unittest)
│
├── requirements.txt
└── README.md
```

---

## Scelte di Preprocessing

Sono state adottate le seguenti operazioni per garantire correttezza sperimentale:

- **Split per ID**: prevenzione del data leakage assicurando che campioni dello stesso paziente non siano distribuiti tra training e test set.
- **Gestione duplicati**: aggregazione dei record basata sull'identificativo del paziente.
- **Valori mancanti**: imputazione tramite mediana.
- **Valori anomali**: limitazione dei valori superiori alla soglia prevista dal dominio (>10) per mantenere coerenza con la codifica originale del dataset.

---

## Esecuzione

Con Docker:

```bash
docker build -t gruppo7-knn .
```
"build": ordina a Docker di creare l'immagine seguendo il Dockerfile  
"-t gruppo7-knn": assegna un nome all'immagine  
".": Docker cercherà il Dockerfile qui

```bash
docker run \-v "$(pwd)/data/raw:/app/data/raw" \-v "$(pwd)/results:/app/results" \gruppo7-knn <modalità> [parametri]
```
"run": dice a Docker di creare e avviare un nuovo contenitore basato su un'immagine specifica  
"$(pwd)/data/raw": è il percorso sul computer reale, dove $(pwd) identifica la cartella attuale  
":/app/data/raw": è il percorso "virtuale" dentro il container  
Quindi il programma dentro Docker vedrà il dataset come se fosse al suo interno, ma in realtà lo sta leggendo dalla cartella reale  
"\-v "$(pwd)/results:/app/results": quando lo script salva un grafico o un report in /app/results, quel file apparirà nella cartella reale  
"\gruppo7-knn": è il nome dell'immagine che è stata costruita con il comando build. Contiene Python e tutte le librerie (pandas, numpy, ecc.)

### Esempi di "docker run"

1. Modalità Holdout

Suddivisione del dataset in training e test set secondo una percentuale specificata.

```bash
docker run \-v "$(pwd)/data/raw:/app/data/raw" \-v "$(pwd)/results:/app/results" \gruppo7-knn holdout --dataset data/raw/version_1.csv --k 5 --test_size 0.3 --metriche all
```

In questo esempio k = 5, test_size = 0.3 (30% dei dati utilizzati per il test) e vengono calcolate tutte le metriche disponibili.

2. Modalità B – K-Fold Cross Validation

Esecuzione della validazione incrociata con K fold.

```bash
docker run \-v "$(pwd)/data/raw:/app/data/raw" \-v "$(pwd)/results:/app/results" \gruppo7-knn B --dataset data/raw/version_1.csv --k 5 --K 5 --metriche accuracy auc
```

In questo caso k = 5 indica il numero di vicini del classificatore, K = 5 indica il numero di fold e vengono calcolate solo le metriche accuracy e auc.

3. Modalità C – Random Subsampling

Esecuzione di più esperimenti di holdout (repeated holdout).

```bash
docker run \-v "$(pwd)/data/raw:/app/data/raw" \-v "$(pwd)/results:/app/results" \gruppo7-knn C --dataset data/raw/version_1.csv --k 5 --K 10 --test_size 0.3 --metriche all
```

In questo esempio k = 5, K = 10 indica il numero di ripetizioni, test_size = 0.3 viene applicato a ogni ripetizione e vengono calcolate tutte le metriche disponibili.

---
Con ambiente virtuale:  
Creare, attivare l’ambiente virtuale e installare le dipendenze:

Spostati nella cartella del progetto e crea il venv:
```bash
python3 -m venv venv
```

Attivazione:
```bash
source venv/bin/activate
```

Installazione dipendenze:
```bash
pip install -r requirements.txt
```

Il programma deve essere eseguito come modulo Python dalla directory principale del progetto:

```bash
python3 -m src.main <modalita> [parametri]
```

### Esempi di esecuzione

Il programma viene eseguito da linea di comando specificando la modalità di validazione, il dataset, il numero di vicini `k` e le metriche da calcolare.

1. Modalità Holdout

Suddivisione del dataset in training e test set secondo una percentuale specificata.

```bash
python3 -m src.main holdout --dataset data/raw/version_1.csv --k 5 --test_size 0.3 --metriche all
```

In questo esempio k = 5, test_size = 0.3 (30% dei dati utilizzati per il test) e vengono calcolate tutte le metriche disponibili.

2. Modalità B – K-Fold Cross Validation

Esecuzione della validazione incrociata con K fold.

```bash
python3 -m src.main B --dataset data/raw/version_1.csv --k 5 --K 5 --metriche accuracy auc
```

In questo caso k = 5 indica il numero di vicini del classificatore, K = 5 indica il numero di fold e vengono calcolate solo le metriche accuracy e auc.

3. Modalità C – Random Subsampling

Esecuzione di più esperimenti di holdout (repeated holdout).

```bash
python3 -m src.main C --dataset data/raw/version_1.csv --k 5 --K 10 --test_size 0.3 --metriche all
```

In questo esempio k = 5, K = 10 indica il numero di ripetizioni, test_size = 0.3 viene applicato a ogni ripetizione e vengono calcolate tutte le metriche disponibili.

---

## Output

Al termine dell’esecuzione, il programma genera automaticamente i risultati nella cartella `results/`.

### File generati

1. `results/results.json`  

Contiene:
- argomenti passati da linea di comando
- report di pulizia del dataset
- risultati delle metriche calcolate

La struttura del file consente di tracciare in modo riproducibile:
- configurazione dell’esperimento
- modalità di validazione utilizzata
- performance ottenute

2. `results/plots/`

Contiene i grafici generati automaticamente:

- Grafici riassuntivi delle metriche (per modalità Holdout)
- Distribuzione delle metriche (per modalità B e C)
- Confusion Matrix (in modalità Holdout)
- ROC Curve (solo quando richiesta AUC)

I grafici sono salvati in formato immagine e consentono una valutazione visiva delle prestazioni del modello.

### Interpretazione dei risultati

- In modalità **Holdout**, viene restituito un singolo valore per ciascuna metrica.
- In modalità **B (K-Fold)** e **C (Random Subsampling)**, vengono salvati:
  - valori per ogni esperimento
  - media
  - deviazione standard

Questo consente di analizzare la stabilità del modello rispetto alla variabilità dello split.

---

## Test Automatici

Il progetto include test automatici sviluppati tramite il modulo `unittest` per verificare la correttezza delle componenti principali della pipeline.

In particolare vengono testati:

- correttezza degli split (holdout, k-fold, random subsampling)
- correttezza del calcolo delle metriche
- coerenza delle aggregazioni (media e deviazione standard)

Per eseguire i test dalla directory principale del progetto:

```bash
python3 -m unittest discover tests
```

Il superamento dei test garantisce la correttezza logica delle funzioni di validazione e valutazione implementate.

---

## Stato Attuale

Il progetto è funzionalmente completo e strutturalmente coerente con le specifiche assegnate.

Attualmente risultano implementati:

- Pipeline completa end-to-end (caricamento dati → preprocessing → training → validazione → output)
- Classificatore k-NN sviluppato da zero con distanza euclidea
- Strategie di validazione: Holdout, K-Fold (B) e Random Subsampling (C)
- Sistema modulare per il calcolo delle metriche tramite Strategy Pattern
- Generazione automatica di file di output e grafici
- Test automatici per verifica di split e metriche

La struttura del repository è organizzata in modo modulare e versionata tramite Git.

---

## Prossimi Passi

2. Implementazione di una Confusion Matrix aggregata per K-Fold.
3. Eventuale calcolo e visualizzazione della ROC media in K-Fold.

---

## Team

- **Erika** – Preprocessing del dataset, sviluppo del parser CLI, supporto all'integrazione della pipeline nel `main`.  
- **Mattia** – Implementazione del classificatore k-NN e della struttura Strategy per la distanza, unit testing, sviluppo iniziale del `main`, configurazione Docker.
- **Gianluca** – Implementazione metriche e strategie di validazione, debugging e finalizzazione del `main`, gestione dell'output dei risultati.