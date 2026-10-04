import pandas as pd
import pytest

from src.loader import DatasetLoader
from src.exceptions import (
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidDatasetError,
)


def test_valid_csv_loads_correctly(tmp_path):

    csv_file = tmp_path / "valid.csv"

    csv_file.write_text(
        "x,y1,y2\n"
        "1,10,20\n"
        "2,30,40\n"
    )

    loader = DatasetLoader()

    dataframe = loader.load_dataset(str(csv_file))

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["x", "y1", "y2"]
    assert len(dataframe) == 2


def test_missing_file_raises_dataset_not_found_error(tmp_path):

    csv_file = tmp_path / "missing.csv"

    loader = DatasetLoader()

    with pytest.raises(DatasetNotFoundError):
        loader.load_dataset(str(csv_file))


def test_empty_csv_raises_empty_dataset_error(tmp_path):

    csv_file = tmp_path / "empty.csv"

    csv_file.write_text(
        "x,y1\n"
    )

    loader = DatasetLoader()

    with pytest.raises(EmptyDatasetError):
        loader.load_dataset(str(csv_file))


def test_malformed_csv_raises_invalid_dataset_error(tmp_path):

    csv_file = tmp_path / "invalid.csv"

    csv_file.write_text(
        "a,b,c\n"
        "1,2,3\n"
        "4,5,6\n"
    )

    loader = DatasetLoader()

    with pytest.raises(InvalidDatasetError):
        loader.load_dataset(str(csv_file))