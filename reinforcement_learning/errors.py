"""Custom exception classes for numpy array validations."""
from typing import Self, final


class NumpyValidationError(Exception):
    """Exception raised when encountering a validation error in numpy arrays.

    Attributes:
        message (str): Description of the validation error.
    """

    def __init__(self: Self, *, message: str) -> None:
        """Initialize a NumpyValidationError object with the given message.

        Args:
            message (str): The error message associated with the exception.

        Returns:
            None

        """
        super().__init__(message)


@final
class NumpyDimError(NumpyValidationError):
    """NumpyDimError Class.

    Raised when the dimensionality of a numpy array does not match the expected dimensionality.
    """

    def __init__(self: Self, expected_dim: int, actual_dim: int) -> None:
        """Initialize the instance with expected and actual dimensions.

        Args:
        expected_dim (int): The expected dimensionality.
        actual_dim (int): The actual dimensionality provided.
        """
        self.expected_dim: int = expected_dim
        self.actual_dim: int = actual_dim
        super().__init__(message=f"{self.expected_dim}-dim is expected, but {self.actual_dim}-dim is passed.")


@final
class NotInitializedError(Exception):
    """An Exception that is raised when an attribute is accessed before it has been initialized."""

    def __init__(self: Self, instance_name: str, attribute_name: str) -> None:
        """Initialize the instance."""
        message = f"{attribute_name} of {instance_name} is not initialized."
        super().__init__(message)


@final
class InvalidMemoryError(Exception):
    """Exception raised for invalid memory access."""
