"""
loader.py

This module is responsible for loading and validating CSV datasets.
"""

from pathlib import Path

import pandas as pd

from src.exceptions import (
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidDatasetError,
)


class DatasetLoader:
    """
    Loads CSV datasets and performs validation.
    """

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate a CSV dataset.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded and validated dataset.

        Raises:
            DatasetNotFoundError:
                If the file does not exist.

            EmptyDatasetError:
                If the dataset contains no rows.

            InvalidDatasetError:
                If the dataset does not contain the required
                x column and at least one y column.
        """

        path = Path(filepath)

        if not path.exists():
            raise DatasetNotFoundError(
                f"Dataset not found: {filepath}"
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise EmptyDatasetError(
                f"Dataset '{filepath}' is empty."
            )

        # Dataset must contain an x column
        if "x" not in dataframe.columns:
            raise InvalidDatasetError(
                f"Dataset '{filepath}' must contain an 'x' column."
            )

        # Dataset must contain at least one y column
        y_columns = [
            column
            for column in dataframe.columns
            if column.startswith("y")
        ]

        if not y_columns:
            raise InvalidDatasetError(
                f"Dataset '{filepath}' must contain at least one 'y' column."
            )

        return dataframe