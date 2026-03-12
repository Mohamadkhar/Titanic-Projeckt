# Titanic Data Analysis - Data Science Labor Projekt

Dieses Repository enthält eine vollständige Datenanalyse des Titanic-Datasets für das Data Science Labor.

##  Projektübersicht

Das Projekt analysiert das berühmte Titanic-Dataset und untersucht folgende Forschungsfragen:
- [Hier deine Forschungsfragen einfügen]
- [Forschungsfrage 2]
- [Forschungsfrage 3]

**Hypothesen:**
- [Hypothese 1]
- [Hypothese 2]
- [Hypothese 3]

##  Projektstruktur

```
titanic-project/
├── data/                    # Datensätze
│   ├── raw/                # Original-Rohdaten
│   ├── processed/          # Vorverarbeitete Daten
│   └── README.md           # Daten-Dokumentation
├── notebooks/              # Jupyter Notebooks für Projekt
│   ├── 01-data-exploration.ipynb
│   ├── 02-data-cleaning.ipynb
│   ├── 03-analysis.ipynb
│   ├── 04-visualization.ipynb
│   └── README.md           # Notebook-Dokumentation
├── src/                    # Python-Quelldateien
│   ├── __init__.py        # Package Initialisierung
│   ├── utils.py           # Hilfsfunktionen
│   └── download_data.py   # Daten-Download-Skript
├── Files/                  # Lab-Notebooks und PDFs (Referenz)
│   ├── 02-jupyter.ipynb bis 17-pytorch.ipynb
│   └── PDF-Dokumentationen
├── environment.yml        # Conda-Environment-Definition
├── requirements.txt       # Python-Pakete (für pip)
├── Dockerfile            # Docker-Image-Konfiguration
├── docker-compose.yml    # Docker-Compose-Konfiguration
├── .dockerignore         # Docker-Ignore-Datei
├── .gitignore           # Git-Ignore-Datei
└── README.md            # Diese Datei
```

##  Quick Start mit Docker

Das Projekt verwendet Miniconda mit einer isolierten Conda-Umgebung (`ds` mit Python 3.11), wie in den Lab-Anforderungen spezifiziert.

### 1. Repository klonen

```bash
git clone <repository-url>
cd titanic-project
```

### 2. Docker Container starten

```bash
docker-compose up -d
```

Das Image wird automatisch gebaut und der Container gestartet.

### 3. Jupyter Notebook öffnen

Öffne deinen Browser und gehe zu:
```
http://localhost:8888
```

Jupyter startet automatisch im `notebooks/` Verzeichnis.

### 4. Container stoppen

```bash
docker-compose down
```

##  Entwicklungsumgebung

### Mit Docker (empfohlen)

Alle Schritte oben durchführen.

### Lokale Installation (Alternative)

Falls du ohne Docker arbeiten möchtest:

```bash
# Conda-Umgebung erstellen
conda env create -f environment.yml

# Umgebung aktivieren
conda activate ds

# Jupyter Notebook starten
jupyter notebook
```

### In den Container zugreifen

Falls du direkt im Container arbeiten möchtest:

```bash
# Container Shell öffnen
docker exec -it titanic-project /bin/bash

# Conda-Umgebung aktivieren
conda activate ds

# Python-Version prüfen
python --version
```

##  Verwendete Bibliotheken

- **NumPy** (2.1.2): Numerische Berechnungen
- **Pandas** (2.2.3): Datenanalyse und -manipulation
- **Matplotlib** (3.9.2): Datenvisualisierung
- **Seaborn** (0.13.2): Statistische Visualisierungen
- **Scikit-learn** (1.5.2): Machine Learning
- **SciPy** (1.14.0): Wissenschaftliche Berechnungen und Hypothesentests
- **Jupyter Notebook** (6.5.7): Interaktive Entwicklungsumgebung

Vollständige Liste siehe [environment.yml](environment.yml) oder [requirements.txt](requirements.txt).

##  Arbeiten auf verschiedenen Geräten

Da das Projekt in einem Docker-Container läuft, kannst du nahtlos zwischen Geräten wechseln:

1. **Desktop**: `docker-compose up -d` → `http://localhost:8888`
2. **Laptop**: Repository pullen → `docker-compose up -d` → `http://localhost:8888`

Alle Änderungen werden automatisch synchronisiert (Volume-Mount).

##  Projektanforderungen

Das finale Projekt erfüllt folgende Kriterien:

-  Als Git-Repository eingereicht
-  In sich geschlossen (alle Skripte und Daten-Download enthalten)
-  README mit Projektbeschreibung und Ausführungsanleitung
-  Definierte Forschungsfragen und Hypothesen
-  Vollständige Datenanalyse mit Visualisierungen

##  Troubleshooting

### Port 8888 bereits belegt

Ändere in [docker-compose.yml](docker-compose.yml):

```yaml
ports:
  - "8889:8888"  # Verwende Port 8889
```

Dann über `http://localhost:8889` zugreifen.

### Container startet nicht

```bash
docker-compose logs
```

oder

```bash
docker logs titanic-project
```

### Packages fehlen

Im Container:

```bash
docker exec -it titanic-project /bin/bash
conda activate ds
pip install <package-name>
```

Dann zu `environment.yml` oder `requirements.txt` hinzufügen.

##  Workflow

1. **Daten vorbereiten**: Dataset in `data/raw/` ablegen oder Download-Skript ausführen
2. **Exploration**: Notebooks in `notebooks/` erstellen und analysieren
3. **Hilfsfunktionen**: Wiederverwendbaren Code in `src/` auslagern
4. **Dokumentation**: Erkenntnisse in README und Notebooks dokumentieren
5. **Git**: Regelmäßig committen und pushen

##  Kontakt

Bei Fragen zum Projekt: [Deine Kontaktinformationen]

---

**Tipp**: Jupyter-Notebooks sollten regelmäßig gespeichert werden. Der Docker-Container mountet das Verzeichnis, sodass alle Änderungen persistent sind.

