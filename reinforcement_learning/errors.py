from typing import Self


class NumpyValidationError(Exception):
    def __init__(self: Self, *, message: str) -> None:
        super().__init__(message)


class NumpyDimError(NumpyValidationError):
    def __init__(self: Self, expected_dim: int, actual_dim: int) -> None:
        self.expected_dim: int = expected_dim
        self.actual_dim: int = actual_dim
        super().__init__(message=f"{self.expected_dim}-dim is expected, but {self.actual_dim}-dim is passed.")
