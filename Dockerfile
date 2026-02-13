# Definisce l'immagine di base e usa una versione leggera di Python 3.11
FROM python:3.11-slim 

# Imposta la cartella di lavoro(app) nel container 
WORKDIR /app  

# Crea la struttura delle cartelle per l'input e l'output
# -p serve a creare anche le sottocartelle
RUN mkdir -p /app/data/raw /app/results/plots  

# Permessi di scrittura per tutta la cartella results
RUN chmod -R 777 /app/results

# Installazione dipendenze
# Copia il file requirements nella cartella corrente del container (.)
# Installazione librere
COPY requirements.txt .  
RUN pip install -r requirements.txt  

# Copia tutto il resto del progetto (cartelle, main, ecc.)
COPY . .  

# Questo dice a Python di cercare i moduli partendo dalla cartella /app
ENV PYTHONPATH=/app

# Definisce il comando di avvio
# Il container una volta avviato lancerà automaticamente python -m src.main ecc.
ENTRYPOINT ["python", "-m", "src.main"]