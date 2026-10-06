"""
exceptions.py

Custom exceptions used throughout the project.
"""


class AssignmentError(Exception):
    """Base class for all project-specific exceptions."""
    pass


class DatasetNotFoundError(AssignmentError):
    """Raised when a required dataset file cannot be found."""
    pass


class EmptyDatasetError(AssignmentError):
    """Raised when a dataset is empty."""
    pass


class InvalidDatasetError(AssignmentError):
    """Raised when a dataset has an invalid structure or format."""
    pass


class DatabaseError(AssignmentError):
    """Raised when a database operation fails."""
    pass


class MappingError(AssignmentError):
    """Raised when test-data mapping fails."""
    pass


class VisualizationError(AssignmentError):
    """Raised when visualization generation fails."""
    pass