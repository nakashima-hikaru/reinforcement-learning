from typing import Self


class NumpyValidationError(Exception):
    def __init__(self: Self, *, message: str) -> None:
        super().__init__(message)


class NumpyDimError(NumpyValidationError):
    def __init__(self: Self, expected_dim: int, actual_dim: int) -> None:
        super().__init__(message=f"{expected_dim}-dim is expected, but {actual_dim}-dim is passed.")
