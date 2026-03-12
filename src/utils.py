"""
Hilfsfunktionen für das Titanic-Projekt
"""

import pandas as pd
import numpy as np


def load_titanic_data(filepath):
    """
    Lädt die Titanic-Daten aus einer CSV-Datei.
    
    Args:
        filepath (str): Pfad zur CSV-Datei
        
    Returns:
        pd.DataFrame: Titanic-Datensatz
    """
    return pd.read_csv(filepath)


def basic_statistics(df):
    """
    Gibt grundlegende Statistiken des DataFrames aus.
    
    Args:
        df (pd.DataFrame): Input DataFrame
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
