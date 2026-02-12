# Progetto AI – Classificazione Tumori con k-NN (Gruppo 7)

Progetto per il corso di **Fondamenti di Intelligenza Artificiale**.

Il progetto sviluppa una pipeline completa di machine learning per la classificazione binaria di tumori (benigno vs maligno) utilizzando un classificatore **k-Nearest Neighbors (k-NN)** implementato interamente da zero, senza l’uso di librerie di alto livello come scikit-learn.

---

## Obiettivo

Realizzare un sistema modulare che consenta di:

1. Effettuare il preprocessing del dataset assegnato (`version_1.csv`).
2. Implementare e addestrare un classificatore k-NN con distanza euclidea.
3. Validare il modello tramite:
   - Hold-out
   - K-Fold Cross Validation
   - Random Subsampling (Repeated Hold-out)
4. Calcolare e analizzare le metriche di performance:
   - Accuracy  
   - Sensitivity (Recall positivo)  
   - Specificity  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion Matrix  
   - ROC Curve  
   - AUC (Area Under the Curve)

---

## Architettura del Progetto

Il codice è organizzato in moduli separati per garantire chiarezza strutturale, riusabilità e separazione delle responsabilità.

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

Sono state adottate le seguenti strategie per garantire correttezza sperimentale:

- **Split per ID**: prevenzione del data leakage assicurando che campioni dello stesso paziente non siano distribuiti tra training e test set.
- **Gestione duplicati**: aggregazione dei record basata sull'identificativo del paziente.
- **Valori mancanti**: imputazione tramite mediana.
- **Valori anomali**: limitazione dei valori superiori alla soglia prevista dal dominio (>10) per mantenere coerenza con la codifica originale del dataset.

---

## Stato Attuale

Il progetto è strutturalmente completo.

- I moduli di preprocessing, modello k-NN, validazione e metriche sono implementati.
- I test automatici obbligatori sono stati superati.
- L'integrazione end-to-end tramite `main.py` è in fase di stabilizzazione per garantire esecuzione completa e robusta.

---

## Esecuzione

Attivare l’ambiente virtuale e installare le dipendenze:

```bash
pip install -r requirements.txt
```

Eseguire il programma:

```bash
python src/main.py
```

---

## Prossimi Passi

1. Stabilizzazione definitiva della pipeline end-to-end.
2. Finalizzazione dell’interfaccia da linea di comando (argparse) per la gestione parametrica di:
   - valore di k
   - tecnica di validazione
   - selezione metriche
3. Dockerizzazione del progetto per garantire riproducibilità completa dell’ambiente.

---

## Team

- **Erika** – Preprocessing, parser parametri, integrazione main  
- **Mattia** – Implementazione k-NN, unit testing, setup Docker  
- **Gianluca** – Metriche, validazione, integrazione main, gestione output, documentazione  

Sviluppo in corso. Monitoraggio continuo dei commit per il docente incaricato.
