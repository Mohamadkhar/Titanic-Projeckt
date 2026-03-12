# Datenverzeichnis

Hier werden die Datensätze für das Projekt gespeichert.

## Struktur

```
data/
├─ raw/          # Original-Rohdaten
├─ processed/    # Vorverarbeitete Daten
└─ README.md     # Diese Datei
```

## Titanic-Dataset herunterladen

Du kannst das Titanic-Dataset von Kaggle herunterladen:
- https://www.kaggle.com/c/titanic/data

Oder ein Skript zum automatischen Download erstellen (siehe `src/download_data.py`).

## Datenschutz

**Wichtig:** Große Datendateien sollten nicht ins Git-Repository gepusht werden.
Siehe `.gitignore` für ausgeschlossene Dateitypen.
