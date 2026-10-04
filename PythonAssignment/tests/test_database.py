import pandas as pd
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.database import DatabaseManager
from src.exceptions import DatabaseError
from src.models import Base, TrainingData


@pytest.fixture
def database():
    """
    Create an in-memory SQLite database for testing.

    Nothing is written to assignment.db.
    """

    database = DatabaseManager(":memory:")

    # Replace the manager's engine with an in-memory SQLite engine.
    database.engine = create_engine(
        "sqlite:///:memory:",
        echo=False
    )

    database.Session = sessionmaker(
        bind=database.engine
    )

    return database


def test_create_tables(database):
    """
    Verify that create_tables creates the database tables.
    """

    database.create_tables()

    inspector = inspect(database.engine)

    tables = inspector.get_table_names()

    assert "training_data" in tables
    assert "ideal_data" in tables
    assert "test_mapping" in tables


def test_insert_dataframe(database):
    """
    Verify that a DataFrame can be inserted successfully.
    """

    database.create_tables()

    dataframe = pd.DataFrame({
        "x": [1.0, 2.0],
        "y1": [10.0, 20.0],
        "y2": [30.0, 40.0],
        "y3": [50.0, 60.0],
        "y4": [70.0, 80.0]
    })

    database.insert_dataframe(
        dataframe,
        TrainingData
    )

    session = database.get_session()

    try:
        records = session.query(TrainingData).all()

        assert len(records) == 2
        assert records[0].x == 1.0
        assert records[0].y1 == 10.0

    finally:
        session.close()


def test_insert_dataframe_failure_rolls_back(database):
    """
    Verify that a failed insertion raises DatabaseError
    and does not partially commit the data.
    """

    database.create_tables()

    valid_dataframe = pd.DataFrame({
        "x": [1.0],
        "y1": [10.0],
        "y2": [20.0],
        "y3": [30.0],
        "y4": [40.0]
    })

    database.insert_dataframe(
        valid_dataframe,
        TrainingData
    )

    # Force a database insertion failure by supplying
    # a column that does not exist in the ORM model.
    invalid_dataframe = pd.DataFrame({
        "x": [2.0],
        "y1": [20.0],
        "y2": [30.0],
        "y3": [40.0],
        "y4": [50.0],
        "invalid_column": ["bad"]
    })

    with pytest.raises(DatabaseError):
        database.insert_dataframe(
            invalid_dataframe,
            TrainingData
        )

    session = database.get_session()

    try:
        records = session.query(TrainingData).all()

        # Only the original valid record should remain.
        assert len(records) == 1
        assert records[0].x == 1.0

    finally:
        session.close()