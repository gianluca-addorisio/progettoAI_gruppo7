# progettoAI_gruppo7

Progetto per il corso di **Fondamenti di Intelligenza Artificiale**.

## Obiettivo
Sviluppare una pipeline di classificazione supervisionata basata su un classificatore
**k-Nearest Neighbors (k-NN)** implementato da zero, applicata alla classificazione di tumori
benigni e maligni.

Il progetto consente di:
- caricare un dataset fornito
- applicare diverse strategie di validazione
- valutare le prestazioni tramite metriche di classificazione

## Struttura del progetto
- `data/raw/` : dataset originali
- `data/processed/` : dataset preprocessati 
- `src/` : codice sorgente
  - `dataset/` : caricam dei dati
  - `preprocessing/` : pulizia e normalizzazione
  - `models/` : implementazione del k-NN
  - `validation/` : strategie di validazione #(holdout, cross-validation)
  - `evaluation/` : calcolo delle metriche
  - `utils/` : funzioni di supporto
- `results/metrics/` : metriche salvate
- `results/plots/` : grafici generati
- `tests/` : test automatici
- `main.py` : punto di ingresso del programma

## Stato del progetto
Struttura iniziale del repo
