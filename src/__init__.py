"""
Titanic Project - Source Code Package
"""

__version__ = '0.1.0'

from .utils import (
	load_titanic_data,
	resolve_column,
	check_missing_values,
	encode_categorical,
	prepare_features_target,
	train_test_split_titanic,
	compare_baseline_models,
	basic_statistics,
)
