"""
exceptions.py

Custom exceptions used throughout the project.
"""


class DatasetNotFoundError(Exception):
    """Raised when a dataset file cannot be found."""
    pass


class EmptyDatasetError(Exception):
    """Raised when a dataset contains no rows."""
    pass


class InvalidDatasetError(Exception):
    """Raised when a dataset has an invalid structure."""
    pass


class DatabaseError(Exception):
    """Raised when a database operation fails."""
    pass


class MappingError(Exception):
    """Raised when test data mapping fails."""
    pass


class VisualizationError(Exception):
    """Raised when visualization generation fails."""
    pass