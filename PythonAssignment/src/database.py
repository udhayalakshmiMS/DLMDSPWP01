"""
database.py

Handles SQLite database creation and ORM operations.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models import (
    Base,
    TrainingData,
    IdealData,
    TestMapping,
)

from src.exceptions import DatabaseError


class DatabaseManager:
    """
    Handles all database-related operations.
    """

    def __init__(self, database_name: str = "assignment.db"):
        """
        Initialize database connection.
        """

        self.engine = create_engine(
            f"sqlite:///{database_name}",
            echo=False
        )

        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """
        Drop and recreate all tables so every run starts clean.
        """

        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)

    def get_session(self):
        """
        Create a new database session.
        """

        return self.Session()

    def insert_dataframe(
        self,
        dataframe,
        model
    ):
        """
        Generic DataFrame insertion method.
        """

        session = self.get_session()

        try:

            for _, row in dataframe.iterrows():

                orm_object = model(
                    **row.to_dict()
                )

                session.add(orm_object)

            session.commit()

        except Exception as error:

            session.rollback()

            raise DatabaseError(
                f"Database insertion failed: {error}"
            ) from error

        finally:

            session.close()

    def insert_training(
        self,
        dataframe
    ):
        """
        Insert training dataset.
        """

        self.insert_dataframe(
            dataframe,
            TrainingData
        )

    def insert_ideal(
        self,
        dataframe
    ):
        """
        Insert ideal dataset.
        """

        self.insert_dataframe(
            dataframe,
            IdealData
        )

    def insert_mapping(
        self,
        dataframe
    ):
        """
        Insert mapping results.
        """

        self.insert_dataframe(
            dataframe,
            TestMapping
        )