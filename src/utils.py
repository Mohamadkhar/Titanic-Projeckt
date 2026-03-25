"""
Hilfsfunktionen für das Titanic-Projekt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_titanic_data(filepath=None):
    """
    Lädt Titanic-Daten aus einer CSV-Datei.

    Wenn kein Pfad übergeben wird, liest die Funktion standardmäßig aus
    `data/raw/titanic.csv` (Projektwurzel).

    Args:
        filepath (str or Path, optional): Pfad zur CSV-Datei.

    Returns:
        pd.DataFrame: Titanic-Datensatz
    """
    if filepath is None:
        filepath = Path("data/raw/titanic.csv")
    return pd.read_csv(filepath)


def resolve_column(df, preferred, fallback):
    """
    Liefert den passenden Spaltennamen aus alternativen Bezeichnern.

    Args:
        df (pd.DataFrame): Input DataFrame
        preferred (str): Bevorzugter Spaltenname
        fallback (str): Alternativer Spaltenname

    Returns:
        str: Gefundener Spaltenname

    Raises:
        KeyError: Wenn keine der Spalten existiert
    """
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise KeyError(f"Spalte nicht gefunden: '{preferred}' oder '{fallback}'")


def check_missing_values(df, columns=None):
    """
    Prüft fehlende Werte im DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Spalten zu prüfen. Wenn None, alle Spalten.
        
    Returns:
        pd.Series: Pro Spalte die Anzahl der fehlenden Werte
    """
    if columns is None:
        columns = df.columns
    missing = df[columns].isnull().sum()
    return missing[missing > 0]


def encode_categorical(df, categorical_cols=None, drop_first=True):
    """
    One-Hot-Encoding für kategorische Spalten.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        categorical_cols (list): Spalten zum Encodieren
        drop_first (bool): Erste Kategorie droppen, um perfekte Multikollinearitaet zu vermeiden
        
    Returns:
        pd.DataFrame: DataFrame mit encoded Spalten
    """
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=drop_first)
    return df_encoded


def prepare_features_target(df, target_col, drop_cols=None):
    """
    Trennt Features und Target für ML-Modelle.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Spaltenname des Targets
        drop_cols (list, optional): Zusätzliche Spalten zum Droppen
        
    Returns:
        tuple: (X, y) - Features und Target
    """
    default_drop_cols = {"PassengerId", "Name", "Ticket", "Cabin"}
    y = df[target_col]
    X = df.drop([target_col], axis=1)

    if drop_cols:
        default_drop_cols.update(drop_cols)

    cols_to_drop = [c for c in default_drop_cols if c in X.columns]
    if cols_to_drop:
        X = X.drop(cols_to_drop, axis=1)

    return X, y


def train_test_split_titanic(df, test_size=0.2, random_state=42):
    """
    Erstellt einen standardisierten Train-Test-Split für Titanic-Modelle.

    Args:
        df (pd.DataFrame): Modellbereiter DataFrame
        test_size (float): Anteil Testdaten
        random_state (int): Seed für Reproduzierbarkeit

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    target_col = resolve_column(df, "Survived", "survived")
    X, y = prepare_features_target(df, target_col)
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def _build_preprocessor(X, scale_numeric=True):
    """
    Baut einen robusten Preprocessor für numerische und kategorische Features.

    - Numerisch: Median-Imputation (+ optional Standardisierung)
    - Kategorisch: Most-Frequent-Imputation + OneHotEncoder(handle_unknown='ignore')
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in bool_features]
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist() + bool_features
    categorical_features = list(dict.fromkeys(categorical_features))

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def compare_baseline_models(X_train, X_test, y_train, y_test):
    """
    Trainiert drei Baseline-Klassifikatoren und berechnet Kernmetriken.

    Args:
        X_train (pd.DataFrame): Trainingsfeatures
        X_test (pd.DataFrame): Testfeatures
        y_train (pd.Series): Trainingslabels
        y_test (pd.Series): Testlabels

    Returns:
        tuple: (results_df, predictions)
            results_df: DataFrame mit Accuracy/Precision/Recall/F1
            predictions: dict[str, np.ndarray] mit Vorhersagen je Modell
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    bool_cols = X_train.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        X_train[col] = X_train[col].astype(int)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(int)

    # KNN und Logistic Regression profitieren stark von skalierten numerischen Features.
    preprocessor_scaled = _build_preprocessor(X_train, scale_numeric=True)
    preprocessor_unscaled = _build_preprocessor(X_train, scale_numeric=False)
    knn_neighbors = max(1, min(5, len(y_train)))

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor_scaled),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor_unscaled),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocessor", preprocessor_scaled),
                ("model", KNeighborsClassifier(n_neighbors=knn_neighbors)),
            ]
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    return results_df, predictions


def basic_statistics(df, columns=None):
    """
    Gibt grundlegende Statistiken des DataFrames aus und liefert sie strukturiert zurueck.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Spalten zu prüfen. Wenn None, alle.
    Returns:
        dict: Strukturierte Statistikdaten
    """
    selected_columns = columns if columns is not None else df.columns.tolist()
    subset = df[selected_columns]

    stats = {
        "shape": subset.shape,
        "columns": selected_columns,
        "missing_values": subset.isnull().sum(),
        "dtypes": subset.dtypes,
        "describe": subset.describe(include="all"),
    }

    print("="*50)
    print("DATASET INFORMATIONEN")
    print("="*50)
    print(f"\nAnzahl Zeilen: {stats['shape'][0]}")
    print(f"Anzahl Spalten: {stats['shape'][1]}")
    print(f"\nSpalten: {stats['columns']}")
    print("\n" + "="*50)
    print("FEHLENDE WERTE")
    print("="*50)
    print(stats["missing_values"])
    print("\n" + "="*50)
    print("DATENTYPEN")
    print("="*50)
    print(stats["dtypes"])

    return stats
