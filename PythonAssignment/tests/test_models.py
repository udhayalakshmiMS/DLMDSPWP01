from sqlalchemy import create_engine, inspect

from src.models import (
    Base,
    TrainingData,
    IdealData,
    TestMapping as TestMappingModel,
)


def test_models_create_tables():
    """
    Verify that all ORM models create their tables correctly.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        echo=False
    )

    Base.metadata.create_all(engine)

    inspector = inspect(engine)

    tables = inspector.get_table_names()

    assert "training_data" in tables
    assert "ideal_data" in tables
    assert "test_mapping" in tables


def test_training_data_model():
    """
    Verify that TrainingData can be instantiated.
    """

    record = TrainingData(
        x=1.0,
        y1=10.0,
        y2=20.0,
        y3=30.0,
        y4=40.0
    )

    assert record.x == 1.0
    assert record.y1 == 10.0
    assert record.y2 == 20.0
    assert record.y3 == 30.0
    assert record.y4 == 40.0


def test_ideal_data_model():
    """
    Verify that IdealData can be instantiated.
    """

    record = IdealData(
        x=1.0,
        y1=10.0,
        y2=20.0
    )

    assert record.x == 1.0
    assert record.y1 == 10.0
    assert record.y2 == 20.0


def test_test_mapping_model():
    """
    Verify that TestMapping can be instantiated.
    """

    record = TestMappingModel(
        x=1.0,
        y=10.0,
        ideal_function="y13",
        delta=0.5
    )

    assert record.x == 1.0
    assert record.y == 10.0
    assert record.ideal_function == "y13"
    assert record.delta == 0.5