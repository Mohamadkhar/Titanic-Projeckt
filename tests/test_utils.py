import unittest
import io
from contextlib import redirect_stdout

import pandas as pd

from src.utils import (
    resolve_column,
    check_missing_values,
    prepare_features_target,
    train_test_split_titanic,
    compare_baseline_models,
    basic_statistics,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "PassengerId": [1, 2, 3, 4, 5, 6],
                "Age": [22, None, 24, 30, 35, 28],
                "Fare": [7.25, 71.83, 8.05, 13.0, 53.1, 10.5],
                "Sex": ["male", "female", "female", "male", "female", "male"],
                "Survived": [0, 1, 1, 0, 1, 0],
            }
        )

    def test_resolve_column(self):
        self.assertEqual(resolve_column(self.df, "Survived", "survived"), "Survived")

    def test_check_missing_values(self):
        missing = check_missing_values(self.df)
        self.assertIn("Age", missing.index)
        self.assertEqual(int(missing["Age"]), 1)

    def test_prepare_features_target_drops_passenger_id(self):
        x, y = prepare_features_target(self.df, "Survived")
        self.assertNotIn("PassengerId", x.columns)
        self.assertEqual(len(y), len(self.df))

    def test_prepare_features_target_drops_text_columns(self):
        df = self.df.copy()
        df["Name"] = ["A", "B", "C", "D", "E", "F"]
        df["Ticket"] = ["T1", "T2", "T3", "T4", "T5", "T6"]
        x, _ = prepare_features_target(df, "Survived")
        self.assertNotIn("Name", x.columns)
        self.assertNotIn("Ticket", x.columns)

    def test_train_test_split_titanic(self):
        x_train, x_test, y_train, y_test = train_test_split_titanic(self.df, test_size=0.33)
        self.assertEqual(len(x_train) + len(x_test), len(self.df))
        self.assertEqual(len(y_train) + len(y_test), len(self.df))

    def test_compare_baseline_models_handles_missing_and_categorical(self):
        x_train, x_test, y_train, y_test = train_test_split_titanic(self.df, test_size=0.33)
        results_df, predictions = compare_baseline_models(x_train, x_test, y_train, y_test)
        self.assertEqual(results_df.shape[0], 3)
        self.assertIn("F1", results_df.columns)
        self.assertEqual(set(predictions.keys()), {"Logistic Regression", "Random Forest", "KNN"})
        self.assertTrue(results_df["F1"].is_monotonic_decreasing)

        for metric in ["Accuracy", "Precision", "Recall", "F1"]:
            self.assertTrue(((results_df[metric] >= 0.0) & (results_df[metric] <= 1.0)).all())

    def test_basic_statistics_returns_dict(self):
        with io.StringIO() as buf, redirect_stdout(buf):
            stats = basic_statistics(self.df)
        self.assertIn("shape", stats)
        self.assertIn("missing_values", stats)
        self.assertIn("dtypes", stats)
        self.assertIn("describe", stats)
        self.assertEqual(stats["shape"][0], len(self.df))


if __name__ == "__main__":
    unittest.main()
