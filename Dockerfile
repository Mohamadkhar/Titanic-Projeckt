# Basis-Image mit Miniconda
FROM continuumio/miniconda3:latest

# Arbeitsverzeichnis setzen
WORKDIR /workspace

# System-Abhängigkeiten installieren
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Environment-Datei kopieren und Conda-Umgebung erstellen
COPY environment.yml .
RUN conda env create -f environment.yml

# Shell aktivieren, damit conda activate funktioniert
SHELL ["conda", "run", "-n", "ds", "/bin/bash", "-c"]

# Projektstruktur vorbereiten
RUN mkdir -p /workspace/data /workspace/notebooks /workspace/src

# Port für Jupyter Notebook freigeben
EXPOSE 8888

# Conda-Umgebung aktivieren und Jupyter Notebook starten
CMD ["conda", "run", "--no-capture-output", "-n", "ds", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
