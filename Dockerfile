# Definisce l'immagine di base e usa una versione leggera di Python 3.10
FROM python:3.10-slim 

# Imposta la cartella di lavoro(app) nel container 
WORKDIR /app  

# Installazione dipendenze
COPY requirements.txt .  # Copia il file requirements nella cartella corrente del container (.)
RUN pip install -r requirements.txt  # Installazione librere

COPY . .  # Copia tutto il resto del progetto (cartelle, main, ecc.)

# Crea la struttura delle cartelle per l'input e l'output
RUN mkdir -p /app/data/raw /app/results/plots  # -p serve a creare anche le sottocartelle

# Permessi di scrittura per tutta la cartella results (da inserire?)
RUN chmod -R 777 /app/results

# Questo dice a Python di cercare i moduli partendo dalla cartella /app
ENV PYTHONPATH=/app

# Definisce il comando di avvio
# Il container una volta avviato lancerà automaticamente python -m src.main ecc.
ENTRYPOINT ["python", "-m", "src.main"]