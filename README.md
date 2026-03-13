# Titanic Data Analysis - Data Science Labor Projekt

Dieses Repository enthält eine vollständige Datenanalyse des Titanic-Datasets für das Data Science Labor.

##  Datensatz

Für das Abschlussprojekt wird der öffentlich zugängliche Kaggle-Datensatz **"Titanic: Machine Learning from Disaster"** verwendet [[1]](#literatur).

### Quelle und Zugriff
Der Datensatz ist über Kaggle verfügbar und liegt als tabellarische CSV-Dateien (`train.csv`, `test.csv`) vor. Er wird häufig als Referenzdatensatz für Klassifikationsaufgaben im maschinellen Lernen verwendet.

- **Kaggle Competition**: https://www.kaggle.com/competitions/titanic/data
- **Seaborn Dataset**: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv

### Kurzbeschreibung
Der Datensatz enthält Informationen zu Passagieren der Titanic sowie den jeweiligen Überlebensstatus als Zielvariable. Die Daten eignen sich für explorative Datenanalyse, statistische Hypothesentests sowie für überwachte Machine-Learning-Modelle zur Vorhersage des Überlebens [[1, 2]](#literatur).

### Beispielhafte Merkmale (Features)

| Feature | Beschreibung |
|---------|--------------|
| **Survived** | Zielvariable (0 = nicht überlebt, 1 = überlebt) |
| **Pclass** | Ticket-Klasse (1, 2, 3) |
| **Sex** | Geschlecht (male, female) |
| **Age** | Alter in Jahren |
| **SibSp** | Anzahl Geschwister/Ehepartner an Bord |
| **Parch** | Anzahl Eltern/Kinder an Bord |
| **Fare** | Ticketpreis |
| **Embarked** | Einschiffungshafen (C = Cherbourg, Q = Queenstown, S = Southampton) |

##  Forschungsfragen (Research Questions)

1. **RQ1**: Welche demografischen Merkmale (z. B. Geschlecht, Alter, Klasse) stehen in Zusammenhang mit der Überlebenswahrscheinlichkeit der Passagiere?

2. **RQ2**: Wie stark erklären sozioökonomische Eigenschaften (z. B. Ticketklasse, Fahrpreis) das Überleben im Vergleich zu demografischen Merkmalen?

3. **RQ3**: Können überwachte Machine-Learning-Modelle basierend auf den Passagiermerkmalen die Überlebenswahrscheinlichkeit zuverlässig vorhersagen?

##  Hypothesen (empirisch prüfbar)

Die folgenden Hypothesen beziehen sich direkt auf Merkmale aus dem Titanic-Datensatz und sind empirisch überprüfbar:

- **H1**: Weibliche Passagiere hatten eine höhere Überlebenswahrscheinlichkeit als männliche Passagiere.

- **H2**: Passagiere der 1. Klasse hatten eine höhere Überlebenswahrscheinlichkeit als Passagiere der 3. Klasse.

- **H3**: Jüngere Passagiere hatten eine höhere Überlebenswahrscheinlichkeit als ältere Passagiere.

### Geplante methodische Überprüfung

- **H1**: Chi-Quadrat-Test zur Prüfung der Unabhängigkeit zwischen `Sex` und `Survived`, ergänzt durch logistische Regression.

- **H2**: Vergleich der Überlebensraten nach `Pclass` (Chi-Quadrat-Test) sowie Analyse von `Fare`; zusätzlich Einsatz eines Random-Forest-Modells zur Bewertung der Feature-Wichtigkeit.

- **H3**: Vergleich von Altersgruppen mittels t-Test oder ANOVA (bzw. nichtparametrischer Alternativen) sowie Regressionsanalyse.

### Warum diese Hypothesen geeignet sind

 Sie sind empirisch überprüfbar mittels statistischer Tests und Machine-Learning-Methoden  
 Sie beziehen sich direkt auf vorhandene Merkmale des Datensatzes  
 Sie sind hypothesengetrieben und erfüllen die Anforderungen der Projektvorlage

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
│   └── utils.py           # Hilfsfunktionen
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

### 5. Datensatz verwenden

Die Datendateien liegen bereits im Projekt unter `data/raw/` und koennen direkt in den Notebooks verwendet werden.

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

## Projektanforderungen

Das finale Projekt erfüllt folgende Kriterien:

-  Als Git-Repository eingereicht → https://github.com/Mohamadkhar/Titanic-Projeckt
-  In sich geschlossen (alle benoetigten Dateien und Ausfuehrungsschritte enthalten)
-  README mit Projektbeschreibung und Ausführungsanleitung
-  Definierte Forschungsfragen (RQ1-RQ3) und Hypothesen (H1-H3)
-  Vollständige Datenanalyse mit Visualisierungen (in Arbeit)

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

1. **Daten vorbereiten**: Dataset in `data/raw/` ablegen (falls noch nicht vorhanden)
2. **Exploration**: Notebooks in `notebooks/` erstellen und analysieren
3. **Hilfsfunktionen**: Wiederverwendbaren Code in `src/` auslagern
4. **Dokumentation**: Erkenntnisse in README und Notebooks dokumentieren
5. **Git**: Regelmäßig committen und pushen

##  Literatur

[1] Kaggle. *Titanic: Machine Learning from Disaster*. https://www.kaggle.com/competitions/titanic/data, Zugriff: Februar 2026.

[2] Mukund Sharma. *Working with Titanic Dataset using Keras: Solving a Simple Classification Problem*. Medium, 2018. https://medium.com/@mukundsharma1995/working-with-titanic-dataset-using-keras-solving-a-simple-classification-problem-440e3860e8fd, Zugriff: Februar 2026.

##  Kontakt

Bei Fragen zum Projekt: Mohamad Kharboutli

---



