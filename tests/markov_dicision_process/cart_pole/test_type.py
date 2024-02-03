import numpy as np
import pytest
import torch
from pydantic_core import PydanticCustomError

from reinforcement_learning.markov_decision_process.cart_pole.type import numpy_validator, tensor_validator  # Change this to your actual import


def test_tensor_validator() -> None:
    # Test that the validator accepts valid inputs
    correct_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    validator = tensor_validator(1, torch.float32)
    assert torch.equal(validator(correct_tensor), correct_tensor)

    # Test that the validator raises errors for incorrect tensor dimensions
    incorrect_dim_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    with pytest.raises(PydanticCustomError):
        validator(incorrect_dim_tensor)

    # Test that the validator raises errors for incorrect tensor data type
    incorrect_dtype_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    with pytest.raises(PydanticCustomError):
        validator(incorrect_dtype_tensor)

    # Test that the validator raises errors if the input is not a tensor
    not_a_tensor = [1, 2, 3]
    with pytest.raises(PydanticCustomError):
        validator(not_a_tensor)


def test_numpy_validator() -> None:
    # Test that the validator accepts valid inputs
    correct_array = np.array([1, 2, 3]).astype(np.float32)
    validator = numpy_validator((3,), np.dtype(np.float32))
    assert np.array_equal(validator(correct_array), correct_array)

    # Test that the validator raises errors for incorrect tensor dimensions
    incorrect_shape_array = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    with pytest.raises(PydanticCustomError):
        validator(incorrect_shape_array)

    # Test that the validator raises errors for incorrect tensor data type
    incorrect_dtype_array = np.array([1, 2, 3]).astype(np.int32)
    with pytest.raises(PydanticCustomError):
        validator(incorrect_dtype_array)

    # Test that the validator raises errors if the input is not a tensor
    not_an_array = [1, 2, 3]
    with pytest.raises(PydanticCustomError):
        validator(not_an_array)
