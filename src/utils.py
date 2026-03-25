"""
Hilfsfunktionen für das Titanic-Projekt
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_titanic_data(filepath):
    """
    Lädt die Titanic-Daten aus einer CSV-Datei.
    
    Args:
        filepath (str or Path): Pfad zur CSV-Datei
        
    Returns:
        pd.DataFrame: Titanic-Datensatz
    """
    return pd.read_csv(filepath)


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
        drop_first (bool): Erste Kategorie droppen (für multicollinearity)
        
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
    y = df[target_col]
    X = df.drop([target_col], axis=1)
    
    if drop_cols:
        X = X.drop([c for c in drop_cols if c in X.columns], axis=1)
    
    # PassengerId ist oft im Datensatz, aber für Vorhersage nicht nutzbar
    if 'PassengerId' in X.columns:
        X = X.drop(['PassengerId'], axis=1)
    
    return X, y


def basic_statistics(df, columns=None):
    """
    Gibt grundlegende Statistiken des DataFrames aus.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        columns (list, optional): Spalten zu prüfen. Wenn None, alle.
    """
    print("="*50)
    print("DATASET INFORMATIONEN")
    print("="*50)
    print(f"\nAnzahl Zeilen: {len(df)}")
    print(f"Anzahl Spalten: {len(df.columns)}")
    print(f"\nSpalten: {list(df.columns)}")
    print("\n" + "="*50)
    print("FEHLENDE WERTE")
    print("="*50)
    print(df.isnull().sum())
    print("\n" + "="*50)
    print("DATENTYPEN")
    print("="*50)
    print(df.dtypes)
