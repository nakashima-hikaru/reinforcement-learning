"""Utility functions that may be used in other files."""
from typing import TypeVar

T = TypeVar("T")


def argmax(d: dict[T, float]) -> T:
    """Find the key with the highest value.

    Args:
        d: A dictionary with keys of type T and values of type float.

    Returns:
        T: The key from the dictionary with the highest corresponding value.

    """
    return max(d, key=lambda key: d[key])
