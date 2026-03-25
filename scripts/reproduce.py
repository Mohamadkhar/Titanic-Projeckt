"""Kleines Reproduktionsskript für den Modellvergleich (RQ3).

Das Skript nutzt eine robuste Preprocessing-Pipeline aus src.utils:
- fehlende Werte werden imputiert,
- kategorische Spalten werden One-Hot-encodiert,
- numerische Features werden für KNN/LogReg skaliert.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_titanic_data, train_test_split_titanic, compare_baseline_models


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "processed" / "titanic_model_ready.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {data_path}. Fuehre zuerst das Cleaning-Notebook aus."
        )

    df = load_titanic_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split_titanic(df)
    results_df, _ = compare_baseline_models(x_train, x_test, y_train, y_test)

    print("RQ3 - Modellvergleich (F1 absteigend):")
    print(results_df.round(4).to_string(index=False))
    best = results_df.iloc[0]
    print(f"\nBestes Modell: {best['Model']} (F1={best['F1']:.4f})")


if __name__ == "__main__":
    main()
